# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Filesystem-backed :class:`CheckpointStore` — one JSON file per save.

Layout::

    <root>/<encoded-client-flow-id>/save-<N>.json

``client_flow_id`` is URL-quoted for directory-safety (arbitrary caller
strings survive round-trip). ``N`` is a monotonically-increasing save
sequence within the trajectory directory — every save (across every
``node_path`` under this ``client_flow_id``) gets the next integer. Each
file contains a single JSON object ``{"state": state_json, "metadata":
metadata_json, "node_path": <str>, "iteration": <int>}``: ``node_path``
+ ``iteration`` are recorded inside the file so the store can serve
lookups keyed on either; the ``save-N.json`` filename is the total-order
signal for load-latest. Writes are atomic within a filesystem (write to
``.tmp`` sibling, then ``os.replace``); a partial write cannot leave a
truncated file the next load would misread. Same
``(node_path, iteration)`` re-save allocates a new ``N`` — the older
record still exists on disk but load-by-key returns the newer one
(highest ``N`` wins on tie).

Intended for local dev / small-scale ops. For a shared-fleet setup use
:class:`llm_gent.flow.stores.PgCheckpointStore`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from appinfra.log import Logger


_SAVE_RE = re.compile(r"^save-(\d+)\.json$")


class JsonFileCheckpointStore:
    """File-per-iteration :class:`CheckpointStore` under a caller-owned root.

    See module docstring for on-disk layout and durability semantics.
    """

    def __init__(
        self,
        lg: Logger,
        root: str | os.PathLike[str],
    ) -> None:
        """Bind a root directory.

        Args:
            lg: Logger for load-time diagnostics (unreadable file
                warnings).
            root: Directory to place per-trajectory subdirectories
                under. Created lazily on first save; not required to
                exist at construction time.
        """
        self._lg = lg
        self._root = Path(root)

    def save_checkpoint(
        self,
        client_flow_id: str,
        node_path: str,
        iteration: int,
        state_json: dict[str, Any],
        metadata_json: dict[str, Any],
    ) -> None:
        """Persist one record as ``save-<N>.json``. Atomic via write-then-replace."""
        traj_dir = self._trajectory_dir(client_flow_id)
        traj_dir.mkdir(parents=True, exist_ok=True)
        next_seq = self._next_save_seq(traj_dir)
        target = traj_dir / f"save-{next_seq}.json"
        payload = {
            "state": state_json,
            "metadata": metadata_json,
            "node_path": node_path,
            "iteration": iteration,
        }
        fd, tmp_path = tempfile.mkstemp(dir=traj_dir, prefix=f"save-{next_seq}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload))
            os.replace(tmp_path, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        self._prune_superseded(traj_dir, next_seq, node_path, iteration)

    def load_checkpoint(
        self,
        client_flow_id: str,
        node_path: str | None = None,
        iteration: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Read the latest record matching the filter (both ``None`` → latest overall)."""
        if node_path is None and iteration is not None:
            raise ValueError("iteration requires node_path; use both or neither")
        traj_dir = self._trajectory_dir(client_flow_id)
        if not traj_dir.is_dir():
            return None
        for _seq, path in self._saves_desc(traj_dir):
            record = self._read_record(path)
            if record is None:
                continue
            state_json, metadata_json, rec_node_path, rec_iteration = record
            if node_path is not None and rec_node_path != node_path:
                continue
            if iteration is not None and rec_iteration != iteration:
                continue
            return state_json, metadata_json
        return None

    def delete_checkpoint(self, client_flow_id: str) -> None:
        """Remove every save file under this trajectory. Idempotent."""
        traj_dir = self._trajectory_dir(client_flow_id)
        if not traj_dir.is_dir():
            return
        for entry in traj_dir.iterdir():
            if entry.is_file() and (
                entry.name.endswith(".json") or entry.name.endswith(".json.tmp")
            ):
                entry.unlink()
        # Non-empty (unrelated files present) — leave the directory in place.
        with contextlib.suppress(OSError):
            traj_dir.rmdir()

    def _trajectory_dir(self, client_flow_id: str) -> Path:
        """URL-encode the id so arbitrary caller strings are path-safe.

        :func:`urllib.parse.quote` encodes every path-relevant char
        that isn't in the RFC-3986 unreserved set (alphanum + ``-._~``).
        Slashes / colons / spaces / NUL bytes all round-trip safely
        through percent-encoding into a single directory name inside
        the root. The one gap is ``.`` and ``..``, which are unreserved
        and pass through verbatim — those would resolve to a traversal
        into the root's parent, so reject them (plus empty strings) up
        front. A resolved-path containment check locks the invariant
        in as defense in depth.
        """
        if not client_flow_id:
            raise ValueError("client_flow_id must not be empty")
        if client_flow_id in (".", ".."):
            raise ValueError(f"client_flow_id must not be {client_flow_id!r} (path-traversal risk)")
        candidate = self._root / quote(client_flow_id, safe="")
        root_resolved = self._root.resolve()
        candidate_resolved = candidate.resolve()
        if root_resolved != candidate_resolved and root_resolved not in candidate_resolved.parents:
            raise ValueError(f"client_flow_id resolves outside store root; got {client_flow_id!r}")
        return candidate

    def _next_save_seq(self, traj_dir: Path) -> int:
        """Return one plus the highest existing ``save-<N>.json`` in ``traj_dir``.

        Starts at ``1`` when the directory has no matching files. Non-
        matching entries (``.tmp`` residues, foreign files) are ignored.
        """
        best = 0
        for entry in traj_dir.iterdir():
            m = _SAVE_RE.match(entry.name)
            if m is None:
                continue
            n = int(m.group(1))
            if n > best:
                best = n
        return best + 1

    def _saves_desc(self, traj_dir: Path) -> list[tuple[int, Path]]:
        """Return ``(seq, path)`` for every ``save-<N>.json`` in descending ``N`` order."""
        saves: list[tuple[int, Path]] = []
        for entry in traj_dir.iterdir():
            m = _SAVE_RE.match(entry.name)
            if m is None:
                continue
            saves.append((int(m.group(1)), entry))
        saves.sort(key=lambda pair: pair[0], reverse=True)
        return saves

    def _read_record(self, path: Path) -> tuple[dict[str, Any], dict[str, Any], str, int] | None:
        """Load one on-disk record; warn and skip on parse failure.

        Returns ``(state_json, metadata_json, node_path, iteration)``. The
        latter two live inside the file so :meth:`load_checkpoint` can
        filter without a separate index.
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return (
                payload["state"],
                payload["metadata"],
                payload["node_path"],
                payload["iteration"],
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            self._lg.warning(
                "checkpoint file unreadable; treating as absent",
                extra={"exception": e, "path": str(path)},
            )
            return None

    def _prune_superseded(
        self, traj_dir: Path, current_seq: int, node_path: str, iteration: int
    ) -> None:
        """Delete older saves for the same ``(node_path, iteration)``.

        Called after a successful save to remove superseded records. Errors
        are logged and swallowed — pruning is best-effort housekeeping.
        """
        for seq, path in self._saves_desc(traj_dir):
            if seq >= current_seq:
                continue
            record = self._read_record(path)
            if record is None:
                continue
            _, _, rec_node_path, rec_iteration = record
            if rec_node_path == node_path and rec_iteration == iteration:
                try:
                    path.unlink()
                except OSError as e:
                    self._lg.warning(
                        "failed to prune superseded checkpoint",
                        extra={"exception": e, "path": str(path)},
                    )

    @staticmethod
    def _decode_id(dirname: str) -> str:
        """Inverse of ``quote``; exposed for tooling that walks the root."""
        return unquote(dirname)
