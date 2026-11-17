# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Provenance examples: version control of what an agent did, on :mod:`appinfra.cas`.

Modules:

- :mod:`explicit_calls` — no Flow: a plain llm-infer call and a SAIA task
  committed by hand inside one activity. Any message list and response value
  works the same way, including a raw HTTP call's JSON.
- :mod:`flow_run` — a Flow run with :class:`~llm_gent.provenance.VersionControl`
  attached as a hook: every step and model call committed, an approval gate
  that pauses the run, and the approval recorded when it is granted.
"""
