# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Provenance for agent runs: where everything came from.

Provenance is version control and attestation: a record of what each step
used and generated, and proof that the record is authentic. Authorization is
not provenance; it is governance, enforced through Flow's ``before_*`` hooks
(:mod:`llm_gent.flow.hooks`), and an approval it grants is recorded here like
any other activity.

This package holds version control (:class:`VersionControl`) on
:mod:`appinfra.cas`: every step is a commit, recording who acted, what went
in and what came out, by content id. Commits are made explicitly, with no
Flow, SAIA or llm-infer required (:meth:`VersionControl.commit`,
:meth:`VersionControl.activity`), or by attaching it to a Flow as a hook.
"""

from ..flow.hooks import Actor, TraceIds
from .version_control import REF, Activity, Messages, Record, VersionControl


__all__ = ["REF", "Activity", "Actor", "Messages", "Record", "TraceIds", "VersionControl"]
