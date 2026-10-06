# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""The resources each run of a flow runs with, and their place in the run's checkpoints.

A flow's resources (:meth:`Flow.with_resource`):

- declared — ``with_resource(key, value)``: the run and everything below
  it run with ``value`` under ``key``;
- per-run child — ``with_resource(key, **args)``: each run of the flow
  (a map item, an iterate pass, a ``.call``) runs with
  ``resource.child(**args)`` of the flow's own ``key`` resource, else the
  enclosing one;
- neither — the run shares the enclosing resources.

A resource's accounting is a position like a cursor: its ``snapshot()``
is in every checkpoint taken while it is in use, and handed back to its
``restore()`` before the run's first step on resume.

- A declared resource is saved at its flow's path
  (:data:`~llm_gent.flow.state.snapshot.RESOURCES`) while its run is in
  progress; the top-level flow's stay through the run's end, so the
  completion commit holds them too and a later run continues from
  there. A resource a flow inherits — the same object as its parent's —
  is saved once, where it is first declared.
- A per-run child is saved at the run's path
  (:data:`~llm_gent.flow.state.snapshot.RUN_RESOURCES`) while the run is
  in progress.

Both entries map each resource's key name to its snapshot.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from ._node_id import _child_flows
from .resource import Resource, ResourceKey, has_child
from .state.snapshot import RESOURCES, RUN_RESOURCES, ScopePath, ScopeRegistry, path_str


if TYPE_CHECKING:
    from .flow import Flow


Resources = Mapping[ResourceKey[Any], Any]
"""The resources a run runs with, by key."""


@contextlib.contextmanager
def run_resources(
    flow: Flow, path: ScopePath, scopes: ScopeRegistry, parent: Resources
) -> Iterator[Resources]:
    """The resources of one run of ``flow`` at ``path``, for the duration of the run.

    ``parent`` are the enclosing run's. The resources ``flow`` declares
    (those its parent does not already run with) and the run's per-run
    children are kept registered at ``path`` — after restoring what a
    checked-out snapshot saved there — while the run is in progress; a
    run that stops keeps them, for the run's halt checkpoint. The
    top-level flow's declared resources stay registered after the run
    completes, for the completion commit.

    Raises:
        RuntimeError: ``flow`` asks for a per-run child of a resource no
            flow above it declares (a flow reached outside its parent's
            composition tree).
    """
    effective = {**parent, **flow._resources}
    declared = {k: v for k, v in flow._resources.items() if parent.get(k) is not v}
    with _kept(flow._lg, scopes, path, RESOURCES, declared, past_the_run=path == ()):
        children = {
            key: _child(flow, effective.get(key), key, args)
            for key, args in flow._resource_children.items()
        }
        effective.update(children)
        with _kept(flow._lg, scopes, path, RUN_RESOURCES, children, past_the_run=False):
            yield MappingProxyType(effective)


def _child(flow: Flow, base: Any, key: ResourceKey[Any], args: dict[str, Any]) -> Any:
    """``base.child(**args)``: the per-run child ``flow`` asks for under ``key``.

    Raises:
        RuntimeError: No ``key`` resource is in scope.
        TypeError: The resource has no ``child()``, or the child does not
            implement :class:`Resource`.
    """
    if base is None:
        raise RuntimeError(_no_resource_message(flow, key))
    if not has_child(base):
        raise TypeError(
            f"resource {key.name!r} ({type(base).__name__}) has no child(); "
            f"a per-run {key.name!r} resource needs one"
        )
    child = base.child(**args)
    if not isinstance(child, Resource):
        raise TypeError(
            f"{type(base).__name__}.child() for resource {key.name!r} must return a "
            f"Resource; got {type(child).__name__}"
        )
    return child


@contextlib.contextmanager
def _kept(
    lg: Logger,
    scopes: ScopeRegistry,
    path: ScopePath,
    entry: str,
    resources: dict[ResourceKey[Any], Any],
    *,
    past_the_run: bool,
) -> Iterator[None]:
    """Keep ``resources``' accounting at ``path`` as ``entry`` while the block runs.

    Restores them first from a checked-out snapshot. Dropped when the
    block completes, unless ``past_the_run``; kept when it stops early
    (the halt checkpoint needs them). Nothing when ``resources`` is empty.
    """
    if not resources:
        yield
        return
    _restore(lg, scopes, path, entry, resources)
    cursor = _ResourcesCursor(entry, resources)
    scopes.open_cursor(path, cursor)
    yield
    if not past_the_run:
        scopes.close_cursor(path, cursor)


def _restore(
    lg: Logger,
    scopes: ScopeRegistry,
    path: ScopePath,
    entry: str,
    resources: dict[ResourceKey[Any], Any],
) -> None:
    """Restore each of ``resources`` from what a checked-out snapshot saved at ``path``, if any.

    A saved resource none of ``resources`` is named after belongs to no
    resource of this flow any more (its declaration was removed or
    renamed): it is dropped with a warning naming it.

    Raises:
        TypeError: The saved entry, or a resource's ``restore()``, rejects it.
    """
    found, saved = scopes.take_cursor(path, entry)
    if not found:
        return
    where = path_str((*path, entry))
    if not isinstance(saved, dict):
        raise TypeError(f"cursor at {where!r} cannot be restored: not a mapping")
    for key, resource in resources.items():
        if key.name not in saved:
            continue
        try:
            resource.restore(saved[key.name])
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise TypeError(f"resource {key.name!r} at {where!r} cannot be restored: {e}") from e
    unknown = sorted(set(saved) - {key.name for key in resources})
    if unknown:
        lg.warning(
            "saved resources with no resource declared here; dropped them",
            extra={"path": where, "names": unknown},
        )


class _ResourcesCursor:
    """Cursor of resources kept in the run's snapshots: name to ``snapshot()``, as ``entry``."""

    def __init__(self, entry: str, resources: dict[ResourceKey[Any], Any]) -> None:
        self.entry = entry
        self.resources = resources

    def cursor(self) -> dict[str, Any]:
        """``{entry: {name: snapshot}}``."""
        return {self.entry: {key.name: r.snapshot() for key, r in self.resources.items()}}


def check_resources(root: Flow) -> None:
    """Raise when ``root``'s composition tree uses resources it cannot run with.

    Raises:
        ValueError: Two different keys with the same name — they would
            share one place in a checkpoint.
        RuntimeError: A flow asks for a per-run child of a resource that
            neither it nor any flow enclosing it declares.
    """
    names: dict[str, ResourceKey[Any]] = {}
    seen: set[tuple[int, frozenset[ResourceKey[Any]]]] = set()
    stack: list[tuple[Flow, frozenset[ResourceKey[Any]]]] = [(root, frozenset())]
    while stack:
        flow, covered = stack.pop()
        if (id(flow), covered) in seen:
            continue
        seen.add((id(flow), covered))
        _check_names(names, [*flow._resources, *flow._resource_children])
        covered = covered | frozenset(flow._resources)
        for key in flow._resource_children:
            if key not in covered:
                raise RuntimeError(_no_resource_message(flow, key))
        covered = covered | frozenset(flow._resource_children)
        for node in flow._nodes:
            stack.extend((child, covered) for _, child in _child_flows(node))


def _check_names(names: dict[str, ResourceKey[Any]], keys: list[ResourceKey[Any]]) -> None:
    """Record ``keys`` by name in ``names``; raise on a name another key already has."""
    for key in keys:
        other = names.setdefault(key.name, key)
        if other is not key:
            raise ValueError(
                f"two resource keys are named {key.name!r}: a flow's resources are stored "
                f"by name, so each name needs one key"
            )


def _no_resource_message(flow: Flow, key: ResourceKey[Any]) -> str:
    """The error for a per-run child of a resource no flow above declares."""
    label = flow._name or "<anonymous>"
    return (
        f"Flow {label!r} asks for a per-run {key.name!r} resource but none is in "
        f"scope: declare one with with_resource(key, resource) on it or an enclosing flow"
    )
