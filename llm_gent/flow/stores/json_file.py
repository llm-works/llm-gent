# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Filesystem-backed :class:`CheckpointStore` — content-addressed object + ref store.

Layout under the caller-owned root::

    <root>/
      names/
        <encoded-client-flow-id>          # text: the history's flow_id
      histories/<encoded-flow-id>/
        _client_flow_id                   # text: the name bound to this history
        objects/
          blob/<content_hash>             # raw payload bytes
          tree/<content_hash>             # canonical JSON bytes
          commit/<content_hash>           # canonical JSON bytes
        refs/
          <encoded-ref-name>              # text: commit hash

``client_flow_id``, ``flow_id`` and ref names are URL-quoted
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

A history has one writer at a time: :meth:`set_ref` is a compare-and-set
under an exclusive ``flock`` on ``refs/.lock``, so a second writer's move
of ``HEAD`` fails instead of forking the history. Intended for local dev /
small-scale ops; for a shared-fleet setup use
:class:`llm_gent.flow.stores.PgCheckpointStore`.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import get_args
from urllib.parse import quote, unquote

from appinfra.log import Logger

from ..checkpoint import Kind, Retention


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


@contextlib.contextmanager
def _locked(lock_file: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``lock_file`` (created if absent) for the block."""
    fd = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _create_exclusive_text(target: Path, text: str) -> bool:
    """Create ``target`` holding ``text`` only if absent; ``False`` when it already exists.

    Writes a ``.tmp`` sibling, then ``os.link`` — an atomic create-if-absent:
    of concurrent creators exactly one wins, and readers never see a
    partially written file.
    """
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.link(tmp_path, target)
        return True
    except FileExistsError:
        return False
    finally:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)


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

    def bind_flow_id(self, client_flow_id: str, flow_id: str) -> str:
        """Bind ``client_flow_id`` → ``flow_id`` unless already bound; return the bound id.

        The history directory and its ``_client_flow_id`` record (which lets
        :meth:`gc_history` drop the binding from the ``flow_id`` side) are
        written first; the name file is then created exclusively. A losing
        concurrent bind removes its unused directory and returns the winner.
        """
        existing = self.get_flow_id(client_flow_id)
        if existing is not None:
            return existing
        history_dir = self._claim_history_dir(flow_id, client_flow_id)
        name_file = self._name_file(client_flow_id)
        name_file.parent.mkdir(parents=True, exist_ok=True)
        if _create_exclusive_text(name_file, flow_id):
            return flow_id
        winner = self.get_flow_id(client_flow_id)
        if winner is None:
            raise RuntimeError(f"binding for {client_flow_id!r} vanished during bind")
        if winner != flow_id:
            shutil.rmtree(history_dir, ignore_errors=True)
        return winner

    def _claim_history_dir(self, flow_id: str, client_flow_id: str) -> Path:
        """Create ``flow_id``'s directory recording its name; reject a ``flow_id`` owned elsewhere."""
        history_dir = self._history_dir(flow_id)
        record = history_dir / "_client_flow_id"
        if record.is_file():
            owner = record.read_text(encoding="utf-8")
            if owner != client_flow_id:
                raise ValueError(f"flow_id {flow_id!r} already names {owner!r}")
        history_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(record, client_flow_id)
        return history_dir

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

    def list_objects(self, flow_id: str) -> list[tuple[Kind, str]]:
        """Return the ``(kind, content_hash)`` of every object file; ``.tmp`` files are skipped."""
        keys: list[tuple[Kind, str]] = []
        for kind in get_args(Kind):
            obj_dir = self._objects_dir(flow_id, kind)
            if obj_dir.is_dir():
                names = (p.name for p in obj_dir.iterdir())
                keys.extend((kind, name) for name in names if not name.endswith(".tmp"))
        return keys

    def delete_objects(self, flow_id: str, keys: list[tuple[Kind, str]]) -> None:
        """Delete the object files ``keys``; missing files are skipped."""
        for kind, content_hash in keys:
            with contextlib.suppress(FileNotFoundError):
                (self._objects_dir(flow_id, kind) / content_hash).unlink()

    # ------------------------------------------------------------------
    # Refs
    # ------------------------------------------------------------------

    def get_ref(self, flow_id: str, name: str) -> str | None:
        """Return the commit hash ref ``name`` points at, or ``None``."""
        try:
            return self._ref_file(flow_id, name).read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None

    def set_ref(self, flow_id: str, name: str, commit_hash: str, expected: str | None) -> bool:
        """Point ref ``name`` at ``commit_hash`` if it points at ``expected`` (``None``: absent).

        The compare and the atomic replace run under an exclusive
        ``flock`` on ``refs/.lock``, so the pair is atomic across threads
        and processes.
        """
        if not commit_hash:
            raise ValueError("commit_hash must not be empty")
        ref_file = self._ref_file(flow_id, name)
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        with _locked(ref_file.parent / ".lock"):
            if self.get_ref(flow_id, name) != expected:
                return False
            _atomic_write_text(ref_file, commit_hash)
            return True

    def list_refs(self, flow_id: str) -> dict[str, str]:
        """Return every ref under ``flow_id``: name → commit hash."""
        refs_dir = self._history_dir(flow_id) / "refs"
        if not refs_dir.is_dir():
            return {}
        refs: dict[str, str] = {}
        for path in refs_dir.iterdir():
            if path.name == ".lock" or path.name.endswith(".tmp"):
                continue
            commit_hash = path.read_text(encoding="utf-8").strip()
            if commit_hash:
                refs[unquote(path.name)] = commit_hash
        return refs

    # ------------------------------------------------------------------
    # History cleanup
    # ------------------------------------------------------------------

    def gc_history(self, flow_id: str) -> None:
        """Remove the history directory and its name binding. Idempotent.

        Order: history contents, then the name binding, then the directory
        with its ``_client_flow_id`` record. A gc that fails partway leaves
        the name (and the record naming it) in place, so it can be retried.
        """
        history_dir = self._history_dir(flow_id)
        if not history_dir.is_dir():
            return
        name_record = history_dir / "_client_flow_id"
        for child in history_dir.iterdir():
            if child == name_record:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
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

    def _ref_file(self, flow_id: str, name: str) -> Path:
        """Return ``<history>/refs/<encoded-name>``; a ``/`` in the name stays one segment."""
        if name == ".lock":
            raise ValueError("ref name '.lock' is reserved")
        return self._encoded_child(self._history_dir(flow_id) / "refs", name, "ref name")

    def _objects_dir(self, flow_id: str, kind: Kind) -> Path:
        """Return ``<history>/objects/<kind>`` (``kind`` is a fixed enum, no encode)."""
        return self._history_dir(flow_id) / "objects" / kind

    @staticmethod
    def _checked(candidate: Path, base: Path, field_name: str, raw: str) -> Path:
        """Defense-in-depth containment check for ``candidate`` under ``base``."""
        base_resolved = base.resolve() if base.exists() else base.absolute()
        candidate_resolved = candidate.resolve() if candidate.exists() else candidate.absolute()
        if base_resolved != candidate_resolved and base_resolved not in candidate_resolved.parents:
            raise ValueError(f"{field_name} resolves outside store root; got {raw!r}")
        return candidate

    @staticmethod
    def _decode_id(dirname: str) -> str:
        """Inverse of :func:`urllib.parse.quote`; exposed for tooling that walks the root."""
        return unquote(dirname)
