# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Shared test primitives for consumers of :mod:`llm_gent.flow`.

Follows the ``<pkg>.<subsystem>.testing`` shape established by
:mod:`appinfra.db.pg.testing` — a shipped module downstream test suites
import from, not a gent-internal ``tests/`` fixture.

:mod:`llm_gent.flow.testing.checkpoint` — canonical multi-stage Flow +
subprocess resume helper for pinning checkpoint/resume determinism
invariants (same-process and cross-process).
"""

from .checkpoint import (
    CanonicalCounter,
    assert_resume_determinism,
    build_canonical_flow,
    resume_in_subprocess,
)


__all__ = [
    "CanonicalCounter",
    "assert_resume_determinism",
    "build_canonical_flow",
    "resume_in_subprocess",
]
