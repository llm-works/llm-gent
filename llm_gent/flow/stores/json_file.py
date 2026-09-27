# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Filesystem-backed :class:`CheckpointStore` — content-addressed object + ref store.

Layout under the caller-owned root::

    <root>/
      names/
        <encoded-client-flow-id>          # text: the history's flow_id
      histories/<encoded-flow-id>/
        _client_flow_id                   # text: the name bound to this history
        _seq                              # ref sequence counter
        objects/
          blob/<content_hash>             # raw payload bytes
          tree/<content_hash>             # canonical JSON bytes
          commit/<content_hash>           # canonical JSON bytes
        refs/
          <encoded-node-path>/<iteration>.json    # {"commit_hash": "...", "seq": N}

``client_flow_id``, ``flow_id`` and ``node_path`` are URL-quoted
(``quote(..., safe="")``) so arbitrary strings survive round-trip as
single directory / file names.
``.`` and ``..`` are rejected up front; a resolved-path containment check
locks the invariant as defense in depth.

Objects are history-scoped — each ``flow_id`` owns its own
``objects/`` tree. Same content bytes across two histories store
twice; the trade-off buys trivial :meth:`gc_history` (a single
``shutil.rmtree`` of the history directory); blobs are deliberately
not shared across histories.

Writes are atomic within a filesystem (write to ``.tmp`` sibling, then
``os.replace``); a partial write cannot leave a truncated file the next
load would misread. Puts are idempotent — the same
``(flow_id, kind, content_hash)`` re-put is a no-op when the
file already exists with the same bytes.

Concurrent access safety is left to the caller: a single-writer
contract per history holds at the framework level. Intended for
local dev / small-scale ops; for a shared-fleet setup use
:class:`llm_gent.flow.stores.PgCheckpointStore`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import quote, unquote

from appinfra.log import Logger

from ..checkpoint import Kind, Retention


_ITER_RE = re.compile(r"^(\d+)\.json$")


def _atomic_write_text(target: Path, text: str) -> None:
    """Write ``text`` to ``target`` via a ``.tmp`` sibling + ``os.replace``."""
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


class JsonFileCheckpointStore:
    """File-per-object :class:`CheckpointStore` under a caller-owned root.

    See module docstring for on-disk layout and durability semantics.
    """

    retention: Retention

    def __init__(
        self,
        lg: Logger,
        root: str | os.PathLike[str],
        *,
        retention: Retention = "retain",
    ) -> None:
        """Bind a root directory + retention policy.

        Args:
            lg: Logger for load-time diagnostics (unreadable file
                warnings).
            root: Directory to place per-history subdirectories
                under. Created lazily on first put; not required to
                exist at construction time.
            retention: ``"retain"`` (default) keeps successful
                histories on disk for audit / diff / provenance;
                ``"gc_on_success"`` calls :meth:`gc_history` on a
                fully-successful :meth:`Flow.run` completion.
        """
        self._lg = lg
        self._root = Path(root)
        self.retention = retention

    # ------------------------------------------------------------------
    # Name map
    # ------------------------------------------------------------------

    def get_flow_id(self, client_flow_id: str) -> str | None:
        """Return the ``flow_id`` bound to ``client_flow_id``, or ``None``."""
        path = self._name_file(client_flow_id)
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None

    def put_flow_id(self, client_flow_id: str, flow_id: str) -> None:
        """Bind ``client_flow_id`` → ``flow_id``; raise if the name is already bound.

        Also records the name inside the history directory so
        :meth:`gc_history` can drop the binding from the ``flow_id`` side.
        """
        name_file = self._name_file(client_flow_id)
        if name_file.exists():
            raise ValueError(f"client_flow_id {client_flow_id!r} is already bound")
        history_dir = self._history_dir(flow_id)
        history_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(history_dir / "_client_flow_id", client_flow_id)
        name_file.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(name_file, flow_id)

    # ------------------------------------------------------------------
    # Object store
    # ------------------------------------------------------------------

    def put_object(
        self,
        flow_id: str,
        kind: Kind,
        content_hash: str,
        payload: bytes,
    ) -> None:
        """Persist ``payload`` under ``objects/{kind}/{content_hash}``.

        Idempotent — a same-hash re-put of the same bytes is a no-op.
        Atomic via write-to-``.tmp`` + ``os.replace``.

        Concurrency: single-writer per history (see module docstring).
        No ``fcntl.flock`` — the framework's CheckpointStore contract is
        single-writer, and content-addressed puts are naturally idempotent
        under same-hash re-puts. A caller that lets two processes save
        into the same ``flow_id`` is violating the contract; the
        physical atomic write here prevents torn files but the caller
        remains responsible for keyspace ordering.
        """
        obj_dir = self._objects_dir(flow_id, kind)
        obj_dir.mkdir(parents=True, exist_ok=True)
        target = obj_dir / content_hash
        if target.exists():
            return
        fd, tmp_path = tempfile.mkstemp(dir=obj_dir, prefix=f"{content_hash}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            os.replace(tmp_path, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def get_object(
        self,
        flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bytes | None:
        """Return the payload bytes for the object, or ``None`` on miss."""
        path = self._objects_dir(flow_id, kind) / content_hash
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as e:
            self._lg.warning(
                "object file unreadable; treating as absent",
                extra={"exception": e, "path": str(path)},
            )
            return None

    def has_object(
        self,
        flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bool:
        """Return ``True`` when the object file exists on disk."""
        return (self._objects_dir(flow_id, kind) / content_hash).is_file()

    # ------------------------------------------------------------------
    # Ref store
    # ------------------------------------------------------------------

    def put_ref(
        self,
        flow_id: str,
        node_path: str,
        iteration: int,
        commit_hash: str,
    ) -> None:
        """Write ``refs/{node_path}/{iteration}.json`` with the commit hash.

        Also stamps a strictly-increasing per-history sequence number
        (``seq``) into the JSON so :meth:`_latest_across_history` can
        pick the newest ref deterministically without relying on mtime
        ties or clock adjustments.
        """
        ref_dir = self._refs_dir(flow_id, node_path)
        ref_dir.mkdir(parents=True, exist_ok=True)
        seq = self._next_ref_seq(flow_id)
        target = ref_dir / f"{iteration}.json"
        payload = json.dumps({"commit_hash": commit_hash, "seq": seq})
        fd, tmp_path = tempfile.mkstemp(dir=ref_dir, prefix=f"{iteration}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def _next_ref_seq(self, flow_id: str) -> int:
        """Return the next per-history ref sequence.

        Single-writer contract per history (see module docstring), so
        read+increment+write without file locking is safe. The seq
        counter file lives at ``<history>/_seq``.
        """
        history_dir = self._history_dir(flow_id)
        history_dir.mkdir(parents=True, exist_ok=True)
        seq_file = history_dir / "_seq"
        current = 0
        if seq_file.is_file():
            try:
                current = int(seq_file.read_text(encoding="utf-8").strip() or "0")
            except (OSError, ValueError):
                current = 0
        next_seq = current + 1
        fd, tmp_path = tempfile.mkstemp(dir=history_dir, prefix="_seq.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(str(next_seq))
            os.replace(tmp_path, seq_file)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        return next_seq

    def resolve_ref(
        self,
        flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> str | None:
        """Return the commit hash for the history key, or ``None``.

        See :class:`~llm_gent.flow.checkpoint.CheckpointStore.resolve_ref`
        for the (``None``, ``iteration``) contract — invalid, raises
        :class:`ValueError`.
        """
        if node_path is None and iteration is not None:
            raise ValueError("iteration requires node_path; use both or neither")
        history_refs = self._history_dir(flow_id) / "refs"
        if not history_refs.is_dir():
            return None

        if node_path is None:
            by_seq = self._latest_across_history_by_seq(history_refs)
            if by_seq is not None:
                return by_seq
            # Fallback for legacy refs without a persisted seq.
            return self._latest_across_history(history_refs)

        ref_dir = self._refs_dir(flow_id, node_path)
        if not ref_dir.is_dir():
            return None

        if iteration is not None:
            return self._read_ref(ref_dir / f"{iteration}.json")

        return self._latest_under_node_path(ref_dir)

    # ------------------------------------------------------------------
    # History cleanup
    # ------------------------------------------------------------------

    def gc_history(self, flow_id: str) -> None:
        """Remove the history directory and its name binding. Idempotent."""
        history_dir = self._history_dir(flow_id)
        if not history_dir.is_dir():
            return
        name_record = history_dir / "_client_flow_id"
        if name_record.is_file():
            name_file = self._name_file(name_record.read_text(encoding="utf-8"))
            with contextlib.suppress(FileNotFoundError):
                if name_file.read_text(encoding="utf-8").strip() == flow_id:
                    name_file.unlink()
        shutil.rmtree(history_dir)

    # ------------------------------------------------------------------
    # Path helpers + traversal guards
    # ------------------------------------------------------------------

    def _history_dir(self, flow_id: str) -> Path:
        """Return ``<root>/histories/<encoded-flow-id>``."""
        return self._encoded_child(self._root / "histories", flow_id, "flow_id")

    def _name_file(self, client_flow_id: str) -> Path:
        """Return ``<root>/names/<encoded-client-flow-id>``."""
        return self._encoded_child(self._root / "names", client_flow_id, "client_flow_id")

    def _encoded_child(self, base: Path, raw: str, field_name: str) -> Path:
        """URL-encode ``raw`` into one path segment under ``base``, path-safely.

        :func:`urllib.parse.quote` encodes every path-relevant char that
        isn't in the RFC-3986 unreserved set (alphanum + ``-._~``). The
        one gap is ``.`` and ``..``, which pass through verbatim — reject
        them up front. A resolved-path containment check locks the
        invariant in as defense in depth.
        """
        if not raw:
            raise ValueError(f"{field_name} must not be empty")
        if raw in (".", ".."):
            raise ValueError(f"{field_name} must not be {raw!r} (path-traversal risk)")
        return self._checked(base / quote(raw, safe=""), base, field_name, raw)

    def _objects_dir(self, flow_id: str, kind: Kind) -> Path:
        """Return ``<history>/objects/<kind>`` (``kind`` is a fixed enum, no encode)."""
        return self._history_dir(flow_id) / "objects" / kind

    def _refs_dir(self, flow_id: str, node_path: str) -> Path:
        """Return ``<history>/refs/<encoded-node-path>``.

        ``node_path`` is the ``"/"``-joined content-addressed node-id chain from
        the run root down to the saving iterate. Encode it as a single directory
        name so slashes don't create a deeper hierarchy on disk.
        """
        if not node_path:
            raise ValueError("node_path must not be empty")
        if node_path in (".", ".."):
            raise ValueError(f"node_path must not be {node_path!r} (path-traversal risk)")
        history_refs = self._history_dir(flow_id) / "refs"
        candidate = history_refs / quote(node_path, safe="")
        return self._checked(candidate, history_refs, "node_path", node_path)

    @staticmethod
    def _checked(candidate: Path, base: Path, field_name: str, raw: str) -> Path:
        """Defense-in-depth containment check for ``candidate`` under ``base``."""
        base_resolved = base.resolve() if base.exists() else base.absolute()
        candidate_resolved = candidate.resolve() if candidate.exists() else candidate.absolute()
        if base_resolved != candidate_resolved and base_resolved not in candidate_resolved.parents:
            raise ValueError(f"{field_name} resolves outside store root; got {raw!r}")
        return candidate

    # ------------------------------------------------------------------
    # Ref resolution helpers
    # ------------------------------------------------------------------

    def _latest_across_history(self, history_refs: Path) -> str | None:
        """Return the newest ref (by mtime) across every ``node_path``."""
        newest_path: Path | None = None
        newest_mtime = -1.0
        for node_dir in history_refs.iterdir():
            if not node_dir.is_dir():
                continue
            for entry in node_dir.iterdir():
                if not _ITER_RE.match(entry.name):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime > newest_mtime:
                    newest_mtime = mtime
                    newest_path = entry
        if newest_path is None:
            return None
        return self._read_ref(newest_path)

    def _latest_across_history_by_seq(self, history_refs: Path) -> str | None:
        """Return the ref with the highest persisted ``seq`` under a history.

        Falls back to :meth:`_latest_across_history` (mtime-based) when
        no ref in the history carries a ``seq`` — e.g., pre-``seq``
        refs written before this change.
        """
        best_seq = -1
        best_path: Path | None = None
        for node_dir in history_refs.iterdir():
            if not node_dir.is_dir():
                continue
            for entry in node_dir.iterdir():
                if not _ITER_RE.match(entry.name):
                    continue
                seq = self._read_ref_seq(entry)
                if seq is None:
                    continue
                if seq > best_seq:
                    best_seq = seq
                    best_path = entry
        if best_path is None:
            return None
        return self._read_ref(best_path)

    @staticmethod
    def _read_ref_seq(path: Path) -> int | None:
        """Return the ``seq`` value stored in a ref file, or ``None`` if absent/unreadable."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        raw = payload.get("seq")
        if not isinstance(raw, int):
            return None
        return raw

    def _latest_under_node_path(self, ref_dir: Path) -> str | None:
        """Return the highest-iteration ref under ``ref_dir``, or ``None``."""
        best_iter = -1
        best_path: Path | None = None
        for entry in ref_dir.iterdir():
            m = _ITER_RE.match(entry.name)
            if m is None:
                continue
            n = int(m.group(1))
            if n > best_iter:
                best_iter = n
                best_path = entry
        if best_path is None:
            return None
        return self._read_ref(best_path)

    def _read_ref(self, path: Path) -> str | None:
        """Load one ref file; warn and skip on parse failure."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return str(payload["commit_hash"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            self._lg.warning(
                "ref file unreadable; treating as absent",
                extra={"exception": e, "path": str(path)},
            )
            return None

    @staticmethod
    def _decode_id(dirname: str) -> str:
        """Inverse of :func:`urllib.parse.quote`; exposed for tooling that walks the root."""
        return unquote(dirname)
