# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Filesystem-backed :class:`CheckpointStore` — content-addressed object + ref store.

Layout under the caller-owned root::

    <root>/<encoded-client-flow-id>/
      objects/
        blob/<content_hash>       # raw payload bytes
        tree/<content_hash>       # canonical JSON bytes
        commit/<content_hash>     # canonical JSON bytes
      refs/
        <encoded-node-path>/<iteration>.json    # {"commit_hash": "..."}

``client_flow_id`` and ``node_path`` are URL-quoted (``quote(..., safe="")``)
so arbitrary caller strings survive round-trip as single directory names.
``.`` and ``..`` are rejected up front; a resolved-path containment check
locks the invariant as defense in depth.

Objects are trajectory-scoped — each ``client_flow_id`` owns its own
``objects/`` tree. Same content bytes across two trajectories store
twice; the trade-off buys trivial :meth:`gc_trajectory` (a single
``shutil.rmtree`` of the trajectory directory) and matches the arc's
non-goal on cross-trajectory blob sharing.

Writes are atomic within a filesystem (write to ``.tmp`` sibling, then
``os.replace``); a partial write cannot leave a truncated file the next
load would misread. Puts are idempotent — the same
``(client_flow_id, kind, content_hash)`` re-put is a no-op when the
file already exists with the same bytes.

Concurrent access safety is left to the caller: a single-writer
contract per trajectory holds at the framework level. Intended for
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
            root: Directory to place per-trajectory subdirectories
                under. Created lazily on first put; not required to
                exist at construction time.
            retention: ``"retain"`` (default) keeps successful
                trajectories on disk for audit / diff / provenance;
                ``"gc_on_success"`` calls :meth:`gc_trajectory` on a
                fully-successful :meth:`Flow.run` completion.
        """
        self._lg = lg
        self._root = Path(root)
        self.retention = retention

    # ------------------------------------------------------------------
    # Object store
    # ------------------------------------------------------------------

    def put_object(
        self,
        client_flow_id: str,
        kind: Kind,
        content_hash: str,
        payload: bytes,
    ) -> None:
        """Persist ``payload`` under ``objects/{kind}/{content_hash}``.

        Idempotent — a same-hash re-put of the same bytes is a no-op.
        Atomic via write-to-``.tmp`` + ``os.replace``.
        """
        obj_dir = self._objects_dir(client_flow_id, kind)
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
        client_flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bytes | None:
        """Return the payload bytes for the object, or ``None`` on miss."""
        path = self._objects_dir(client_flow_id, kind) / content_hash
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
        client_flow_id: str,
        kind: Kind,
        content_hash: str,
    ) -> bool:
        """Return ``True`` when the object file exists on disk."""
        return (self._objects_dir(client_flow_id, kind) / content_hash).is_file()

    # ------------------------------------------------------------------
    # Ref store
    # ------------------------------------------------------------------

    def put_ref(
        self,
        client_flow_id: str,
        node_path: str,
        iteration: int,
        commit_hash: str,
    ) -> None:
        """Write ``refs/{node_path}/{iteration}.json`` with the commit hash."""
        ref_dir = self._refs_dir(client_flow_id, node_path)
        ref_dir.mkdir(parents=True, exist_ok=True)
        target = ref_dir / f"{iteration}.json"
        payload = json.dumps({"commit_hash": commit_hash})
        fd, tmp_path = tempfile.mkstemp(dir=ref_dir, prefix=f"{iteration}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def resolve_ref(
        self,
        client_flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> str | None:
        """Return the commit hash for the trajectory key, or ``None``.

        See :class:`~llm_gent.flow.checkpoint.CheckpointStore.resolve_ref`
        for the (``None``, ``iteration``) contract — invalid, raises
        :class:`ValueError`.
        """
        if node_path is None and iteration is not None:
            raise ValueError("iteration requires node_path; use both or neither")
        traj_refs = self._trajectory_dir(client_flow_id) / "refs"
        if not traj_refs.is_dir():
            return None

        if node_path is None:
            return self._latest_across_trajectory(traj_refs)

        ref_dir = self._refs_dir(client_flow_id, node_path)
        if not ref_dir.is_dir():
            return None

        if iteration is not None:
            return self._read_ref(ref_dir / f"{iteration}.json")

        return self._latest_under_node_path(ref_dir)

    # ------------------------------------------------------------------
    # Trajectory cleanup
    # ------------------------------------------------------------------

    def gc_trajectory(self, client_flow_id: str) -> None:
        """Remove the trajectory directory and everything under it. Idempotent."""
        traj_dir = self._trajectory_dir(client_flow_id)
        if traj_dir.is_dir():
            shutil.rmtree(traj_dir)

    # ------------------------------------------------------------------
    # Path helpers + traversal guards
    # ------------------------------------------------------------------

    def _trajectory_dir(self, client_flow_id: str) -> Path:
        """URL-encode the id so arbitrary caller strings are path-safe.

        :func:`urllib.parse.quote` encodes every path-relevant char that
        isn't in the RFC-3986 unreserved set (alphanum + ``-._~``). The
        one gap is ``.`` and ``..``, which pass through verbatim — reject
        them up front. A resolved-path containment check locks the
        invariant in as defense in depth.
        """
        if not client_flow_id:
            raise ValueError("client_flow_id must not be empty")
        if client_flow_id in (".", ".."):
            raise ValueError(f"client_flow_id must not be {client_flow_id!r} (path-traversal risk)")
        candidate = self._root / quote(client_flow_id, safe="")
        return self._checked(candidate, self._root, "client_flow_id", client_flow_id)

    def _objects_dir(self, client_flow_id: str, kind: Kind) -> Path:
        """Return ``<traj>/objects/<kind>`` (``kind`` is a fixed enum, no encode)."""
        return self._trajectory_dir(client_flow_id) / "objects" / kind

    def _refs_dir(self, client_flow_id: str, node_path: str) -> Path:
        """Return ``<traj>/refs/<encoded-node-path>``.

        ``node_path`` is the ``"/"``-joined content-addressed node-id chain from
        the run root down to the saving iterate. Encode it as a single directory
        name so slashes don't create a deeper hierarchy on disk.
        """
        if not node_path:
            raise ValueError("node_path must not be empty")
        if node_path in (".", ".."):
            raise ValueError(f"node_path must not be {node_path!r} (path-traversal risk)")
        traj_refs = self._trajectory_dir(client_flow_id) / "refs"
        candidate = traj_refs / quote(node_path, safe="")
        return self._checked(candidate, traj_refs, "node_path", node_path)

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

    def _latest_across_trajectory(self, traj_refs: Path) -> str | None:
        """Return the newest ref (by mtime) across every ``node_path``."""
        newest_path: Path | None = None
        newest_mtime = -1.0
        for node_dir in traj_refs.iterdir():
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
