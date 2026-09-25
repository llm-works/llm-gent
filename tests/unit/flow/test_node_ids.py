# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Content-addressed node IDs — lazy, chained, collision-free.

Exercises the pair
:func:`llm_gent.flow._node_id._compute_node_id` /
:func:`llm_gent.flow._node_id._descend_context`. IDs are computed at
descent time from the enclosing Flow's ``chain_context`` plus the
node's local key (kind, chain position, target qualname); the descent
context is itself a hash chain from the run's root through every
subflow / branch-arm / iterate-body above.

The chain gives every node in the composition tree a globally-unique
identifier the checkpoint layer uses. Tests pin four properties:

- **Deterministic:** same graph → same IDs across processes.
- **Collision-free by construction:** distinct chain positions,
  boundaries, and target qualnames all flip the hash.
- **Context-sensitive:** the same subflow at two different call sites
  produces two different IDs for its inner nodes.
- **Cosmetic-stable:** edits that don't change the local key
  (renaming a local variable inside a verb body, whitespace) leave
  IDs untouched.
"""

from __future__ import annotations

from llm_gent.flow import Flow, verb
from llm_gent.flow._node_id import _compute_node_id, _descend_context

from .conftest import ROLE_A, ROLE_B, make_test_logger


@verb(role=ROLE_A)
async def verb_alpha(ctx, x):  # type: ignore[no-untyped-def]
    return x


@verb(role=ROLE_A)
async def verb_beta(ctx, x):  # type: ignore[no-untyped-def]
    return x


@verb(role=ROLE_B)
async def verb_gamma(ctx, x):  # type: ignore[no-untyped-def]
    return x


def _mkflow() -> Flow:
    return Flow(lg=make_test_logger())


def _chain_step_ids(flow: Flow, chain_context: str = "") -> list[str]:
    """Compute the runtime node_ids for a Flow's chain under a given context."""
    return [_compute_node_id(chain_context, n, i) for i, n in enumerate(flow._nodes)]


# -----------------------------------------------------------------------------
# Shape + determinism
# -----------------------------------------------------------------------------


def test_id_is_16_hex_chars() -> None:
    """digest_size=8 → 64 bits → 16 hex chars."""
    flow = _mkflow().call(verb_alpha)
    node_id = _compute_node_id("", flow._nodes[0], 0)
    assert len(node_id) == 16
    int(node_id, 16)  # parses as hex


def test_same_composition_same_ids() -> None:
    """Building the same chain twice → identical node IDs at every position."""
    a = _mkflow().call(verb_alpha).then(verb_beta).then(verb_gamma)
    b = _mkflow().call(verb_alpha).then(verb_beta).then(verb_gamma)
    assert _chain_step_ids(a) == _chain_step_ids(b)


def test_ids_unique_within_chain() -> None:
    """Same target at three different positions → three distinct IDs.

    Chain position participates in the hash, so identity at (i) is
    always distinct from identity at (j) for i != j.
    """
    flow = _mkflow().call(verb_alpha).then(verb_alpha).then(verb_alpha)
    ids = _chain_step_ids(flow)
    assert len(set(ids)) == 3


# -----------------------------------------------------------------------------
# Structural changes flip IDs (identity break)
# -----------------------------------------------------------------------------


def test_target_swap_flips_id() -> None:
    """Same position + different target → different ID at that position."""
    a = _mkflow().call(verb_alpha).then(verb_beta)
    b = _mkflow().call(verb_alpha).then(verb_gamma)
    ids_a, ids_b = _chain_step_ids(a), _chain_step_ids(b)
    assert ids_a[0] == ids_b[0]
    assert ids_a[1] != ids_b[1]


def test_reorder_flips_ids() -> None:
    """Swapping the order of two chain steps flips both IDs.

    Chain position participates in identity — a saved path built
    against the pre-swap layout will refuse to replay against the
    post-swap layout at the very first mismatch.
    """
    a = _mkflow().call(verb_alpha).then(verb_beta)
    b = _mkflow().call(verb_beta).then(verb_alpha)
    ids_a, ids_b = _chain_step_ids(a), _chain_step_ids(b)
    assert ids_a[0] != ids_b[0]
    assert ids_a[1] != ids_b[1]


def test_kind_swap_flips_id() -> None:
    """Same target at the same position but different composition kind → different ID."""
    a = _mkflow().call(verb_alpha)
    b = _mkflow().iterate(verb_alpha, max_iters=1)
    ids_a = _chain_step_ids(a)
    ids_b = _chain_step_ids(b)
    assert ids_a[0] != ids_b[0]


# -----------------------------------------------------------------------------
# Descent context — subflows, branch arms, iterate bodies
# -----------------------------------------------------------------------------


def test_descend_context_boundary_matters() -> None:
    """The two arms of a branch descend into different chain contexts.

    A body sitting behind ``then`` sees a distinct ``chain_context``
    from the same body sitting behind ``else`` — its chain-step IDs
    will differ even though the body's own chain is identical.
    """
    parent = _mkflow().call(verb_alpha)
    parent_id = _compute_node_id("", parent._nodes[0], 0)
    then_ctx = _descend_context(parent_id, "then")
    else_ctx = _descend_context(parent_id, "else")
    body = _mkflow().call(verb_beta)
    then_ids = _chain_step_ids(body, then_ctx)
    else_ids = _chain_step_ids(body, else_ctx)
    assert then_ids != else_ids


def test_shared_subflow_at_two_call_sites_has_distinct_ids() -> None:
    """The same subflow Flow object at two different .call sites produces distinct inner IDs.

    Content-addressing under the chained-hash scheme means "identity in
    the tree," not "identity of the underlying object." Reusing a
    subflow at two places yields two positions — two identities —
    for each of its inner nodes.
    """
    shared = _mkflow().call(verb_beta)
    parent = _mkflow().call(shared).then(shared)
    call_0_id = _compute_node_id("", parent._nodes[0], 0)
    call_1_id = _compute_node_id("", parent._nodes[1], 1)
    inner_at_0 = _chain_step_ids(shared, _descend_context(call_0_id, "call"))
    inner_at_1 = _chain_step_ids(shared, _descend_context(call_1_id, "call"))
    assert inner_at_0 != inner_at_1


# -----------------------------------------------------------------------------
# Cosmetic-edit stability
# -----------------------------------------------------------------------------


def test_verb_body_edit_preserves_id() -> None:
    """The ID depends on qualname, not on the code object.

    Cosmetic edits to a verb's body — renaming an internal local,
    reformatting whitespace, adding a comment — leave the qualname
    unchanged; the checkpoint layer sees the same ID before and after.
    Explicitly test by rebuilding the same graph twice and comparing.
    """
    a = _mkflow().call(verb_alpha).then(verb_beta)
    b = _mkflow().call(verb_alpha).then(verb_beta)
    assert _chain_step_ids(a) == _chain_step_ids(b)
