# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Filesystem-backed :class:`CheckpointStore` — one JSON file per iteration.

Layout::

    <root>/<encoded-client-flow-id>/iter-<N>.json

``client_flow_id`` is URL-quoted for directory-safety (arbitrary caller
strings survive round-trip). Each per-iteration file contains a single
JSON object ``{"state": state_json, "metadata": metadata_json}``. Writes
are atomic within a filesystem (write to ``.tmp`` sibling, then
``os.replace``); a partial write cannot leave a truncated file the
next load would misread. Load-latest scans the directory for
``iter-*.json`` and returns the highest ``N``; concurrent saves at the
same iteration collapse to whichever ``os.replace`` runs last.

Intended for local dev / small-scale ops. For a shared-fleet setup use
:class:`llm_gent.flow.stores.PgCheckpointStore`.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from appinfra.log import Logger


_ITER_RE = re.compile(r"^iter-(\d+)\.json$")


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
        iteration: int,
        state_json: dict[str, Any],
        metadata_json: dict[str, Any],
    ) -> None:
        """Persist one iteration record. Atomic via write-then-replace."""
        traj_dir = self._trajectory_dir(client_flow_id)
        traj_dir.mkdir(parents=True, exist_ok=True)
        target = traj_dir / f"iter-{iteration}.json"
        tmp = target.with_suffix(".json.tmp")
        payload = {"state": state_json, "metadata": metadata_json}
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, target)

    def load_checkpoint(
        self,
        client_flow_id: str,
        iteration: int | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Read one iteration record (or the latest under this trajectory)."""
        traj_dir = self._trajectory_dir(client_flow_id)
        if not traj_dir.is_dir():
            return None
        if iteration is None:
            target = self._latest_file(traj_dir)
            if target is None:
                return None
        else:
            target = traj_dir / f"iter-{iteration}.json"
            if not target.is_file():
                return None
        return self._read_record(target)

    def delete_checkpoint(self, client_flow_id: str) -> None:
        """Remove every iteration file under this trajectory. Idempotent."""
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

    def _latest_file(self, traj_dir: Path) -> Path | None:
        """Scan for ``iter-N.json`` files; return the highest-N path."""
        best: tuple[int, Path] | None = None
        for entry in traj_dir.iterdir():
            m = _ITER_RE.match(entry.name)
            if m is None:
                continue
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, entry)
        return None if best is None else best[1]

    def _read_record(self, path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Load one on-disk record; warn and skip on parse failure."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload["state"], payload["metadata"]
        except (OSError, json.JSONDecodeError, KeyError) as e:
            self._lg.warning(
                "checkpoint file unreadable; treating as absent",
                extra={"exception": e, "path": str(path)},
            )
            return None

    @staticmethod
    def _decode_id(dirname: str) -> str:
        """Inverse of ``quote``; exposed for tooling that walks the root."""
        return unquote(dirname)
