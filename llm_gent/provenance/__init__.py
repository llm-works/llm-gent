# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Provenance for agent runs: version control on :mod:`appinfra.cas`, built on Flow hooks.

Provenance is version control, attestation and authorization. This package
holds version control (:class:`VersionControl`): every step of a run is a
commit, recording who acted, what went in and what came out, by content id.
It uses only the public hook API (:mod:`llm_gent.flow.hooks`), so other
provenance layers and bridges to other frameworks can be built the same way.
"""

from .version_control import REF, Messages, Record, VersionControl


__all__ = ["REF", "Messages", "Record", "VersionControl"]
