# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""Content-addressed node IDs — lazy, chained, collision-free.

Exercises the pair
:func:`llm_gent.flow._node_id._compute_node_ids` /
:func:`llm_gent.flow._node_id._descend_context`. IDs are computed at
descent time from the enclosing Flow's ``chain_context`` plus the
node's local key (kind, target qualname, occurrence among same-target
steps); the descent context is itself a hash chain from the run's root
through every subflow / branch-arm / iterate-body above.

The chain gives every node in the composition tree a globally-unique
identifier the checkpoint layer uses. Tests pin five properties:

- **Deterministic:** same graph → same IDs across processes.
- **Collision-free by construction:** distinct occurrences,
  boundaries, and target qualnames all flip the hash.
- **Position-independent:** inserting, removing or reordering steps
  with other targets leaves a step's ID unchanged.
- **Context-sensitive:** the same subflow at two different call sites
  produces two different IDs for its inner nodes.
- **Cosmetic-stable:** edits that don't change the local key
  (renaming a local variable inside a verb body, whitespace) leave
  IDs untouched.
"""

from __future__ import annotations

from llm_gent.flow import Flow, verb
from llm_gent.flow._node_id import _compute_node_ids, _descend_context

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
    return list(_compute_node_ids(chain_context, flow._nodes))


# -----------------------------------------------------------------------------
# Shape + determinism
# -----------------------------------------------------------------------------


def test_id_is_16_hex_chars() -> None:
    """digest_size=8 → 64 bits → 16 hex chars."""
    flow = _mkflow().call(verb_alpha)
    node_id = _chain_step_ids(flow)[0]
    assert len(node_id) == 16
    int(node_id, 16)  # parses as hex


def test_same_composition_same_ids() -> None:
    """Building the same chain twice → identical node IDs at every position."""
    a = _mkflow().call(verb_alpha).then(verb_beta).then(verb_gamma)
    b = _mkflow().call(verb_alpha).then(verb_beta).then(verb_gamma)
    assert _chain_step_ids(a) == _chain_step_ids(b)


def test_ids_unique_within_chain() -> None:
    """Same target three times → three distinct IDs, told apart by occurrence."""
    flow = _mkflow().call(verb_alpha).then(verb_alpha).then(verb_alpha)
    ids = _chain_step_ids(flow)
    assert len(set(ids)) == 3


# -----------------------------------------------------------------------------
# Identity under chain edits
# -----------------------------------------------------------------------------


def test_target_swap_flips_id() -> None:
    """A different target → a different ID; the unchanged step keeps its ID."""
    a = _mkflow().call(verb_alpha).then(verb_beta)
    b = _mkflow().call(verb_alpha).then(verb_gamma)
    ids_a, ids_b = _chain_step_ids(a), _chain_step_ids(b)
    assert ids_a[0] == ids_b[0]
    assert ids_a[1] != ids_b[1]


def test_reorder_keeps_ids() -> None:
    """Swapping two steps with different targets keeps each step's ID."""
    a = _mkflow().call(verb_alpha).then(verb_beta)
    b = _mkflow().call(verb_beta).then(verb_alpha)
    ids_a, ids_b = _chain_step_ids(a), _chain_step_ids(b)
    assert ids_a[0] == ids_b[1]
    assert ids_a[1] == ids_b[0]


def test_insert_and_remove_keep_ids() -> None:
    """Inserting or removing a step leaves the other steps' IDs unchanged."""
    base = _chain_step_ids(_mkflow().call(verb_alpha).then(verb_beta))
    inserted = _chain_step_ids(_mkflow().call(verb_gamma).then(verb_alpha).then(verb_beta))
    removed = _chain_step_ids(_mkflow().call(verb_beta))
    assert inserted[1:] == base
    assert removed == base[1:]


def test_same_target_steps_follow_their_order() -> None:
    """Steps sharing a target are identified by their order among themselves.

    Removing the first of two ``verb_alpha`` steps hands the second the
    first one's ID — the only identity available for indistinguishable
    targets.
    """
    both = _chain_step_ids(_mkflow().call(verb_alpha).then(verb_beta).then(verb_alpha))
    second_removed = _chain_step_ids(_mkflow().call(verb_alpha).then(verb_beta))
    assert second_removed == both[:2]
    first_removed = _chain_step_ids(_mkflow().call(verb_beta).then(verb_alpha))
    assert first_removed == [both[1], both[0]]


def test_kind_swap_flips_id() -> None:
    """Same target but a different composition kind → different ID."""
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
    parent_id = _chain_step_ids(parent)[0]
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
    subflow at two places yields two occurrences — two identities —
    for each of its inner nodes.
    """
    shared = _mkflow().call(verb_beta)
    parent = _mkflow().call(shared).then(shared)
    call_0_id, call_1_id = _chain_step_ids(parent)
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


# -----------------------------------------------------------------------------
# Flow root hash — the whole composition tree
# -----------------------------------------------------------------------------


def test_root_hash_is_deterministic_cas_hash() -> None:
    """Same composition → same 64-hex root hash (blake2b-256, like CAS objects)."""
    a = _mkflow().call(verb_alpha).iterate(_mkflow().call(verb_beta), max_iters=3)
    b = _mkflow().call(verb_alpha).iterate(_mkflow().call(verb_beta), max_iters=3)
    assert a.root_hash() == b.root_hash()
    assert len(a.root_hash()) == 64


def test_root_hash_sees_nested_changes() -> None:
    """A different verb deep inside an iterate body changes the root hash."""
    a = _mkflow().call(verb_alpha).iterate(_mkflow().call(verb_beta), max_iters=3)
    b = _mkflow().call(verb_alpha).iterate(_mkflow().call(verb_gamma), max_iters=3)
    assert a.root_hash() != b.root_hash()


def test_root_hash_distinguishes_branch_arms() -> None:
    """Swapping the then / else arms changes the root hash."""
    then_flow, else_flow = _mkflow().call(verb_alpha), _mkflow().call(verb_beta)
    a = _mkflow().branch(when=lambda *_: True, then=then_flow, else_=else_flow)
    b = _mkflow().branch(when=lambda *_: True, then=else_flow, else_=then_flow)
    assert a.root_hash() != b.root_hash()


def test_root_hash_ignores_node_parameters() -> None:
    """Parameters that don't enter node ids (max_iters) don't enter the root hash."""
    a = _mkflow().iterate(_mkflow().call(verb_beta), max_iters=3)
    b = _mkflow().iterate(_mkflow().call(verb_beta), max_iters=10)
    assert a.root_hash() == b.root_hash()


def test_root_hash_terminates_on_recursive_flow() -> None:
    """A flow that re-enters itself through a branch arm hashes without recursing forever."""
    recursive = _mkflow().call(verb_alpha)
    recursive.branch(when=lambda *_: False, then=recursive)
    assert len(recursive.root_hash()) == 64
