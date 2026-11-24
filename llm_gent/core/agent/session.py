# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-gent Authors

"""An agent's sessions: one conversation each, addressed by an id the agent issues.

A caller never makes a session id. Asking without one opens a session and
returns its id; asking with one continues that session, which must exist.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from appinfra.log import Logger
from llm_kelt.conversation import Config as KeltConfig
from llm_kelt.conversation import Conversation, SessionStorage, SlidingWindowCompactor
from llm_kelt.core.errors import NotFoundError

from ..errors import ConfigError, SessionNotFoundError
from .identity import Identity


_KEY_MAX = 255
"""The longest key kelt's session storage takes."""

_ID_LEN = 36
"""The length of a session id: a uuid in its canonical form."""


@dataclass(frozen=True)
class SessionConfig:
    """How an agent's sessions keep their conversations: the config's ``conversation`` section.

    ``max_tokens``, ``compact_threshold``, ``preserve_system`` and
    ``min_recent_messages`` tune a conversation's compaction; ``compactor``
    names the strategy (``sliding_window``). ``max_live`` bounds the sessions
    held in memory when sessions persist: beyond it the least recently used
    idle one is dropped, and its next ask reloads it from storage.
    """

    max_tokens: int = 32000
    compact_threshold: float = 0.8
    preserve_system: bool = True
    min_recent_messages: int = 4
    compactor: str = "sliding_window"
    max_live: int = 1000


class Session:
    """One conversation with an agent. Callers know it only by ``id``.

    ``lock`` serializes the asks in the session: a conversation is a sequence.
    ``users`` counts the asks holding the session, waiting for its lock
    included; it changes only under the table's lock.
    """

    def __init__(self, session_id: str, conversation: Conversation) -> None:
        self.id = session_id
        self.conversation = conversation
        self.lock = threading.Lock()
        self.users = 0


class Sessions:
    """The sessions of one agent: opened or found again for an ask (:meth:`use`), saved.

    With ``storage`` every session is saved under a key of the agent's
    ``context_key``, name and the session id, so an id is valid only at the
    agent that issued it; a session dropped from memory, or held by an
    earlier process, reloads from storage. Without storage sessions live
    in memory only and are never dropped.
    """

    def __init__(
        self,
        lg: Logger,
        identity: Identity,
        config: SessionConfig,
        storage: SessionStorage | None,
    ) -> None:
        if config.compactor != "sliding_window":
            raise ConfigError(f"unsupported conversation compactor: {config.compactor!r}")
        self._lg = lg
        self._prefix = f"{identity.context_key}/{identity.name}/"
        if len(self._prefix) + _ID_LEN > _KEY_MAX:
            raise ConfigError(
                f"agent {identity.name!r}: context_key and name too long for session keys "
                f"(at most {_KEY_MAX - _ID_LEN - 2} characters together)"
            )
        self._config = config
        self._storage = storage
        self._kelt_config = KeltConfig(
            max_tokens=config.max_tokens,
            compact_threshold=config.compact_threshold,
            preserve_system=config.preserve_system,
            min_recent_messages=config.min_recent_messages,
        )
        self._live: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    @contextmanager
    def use(self, session_id: str | None) -> Iterator[Session]:
        """The session to ask in, held for the ask: its lock, and kept in memory.

        ``None`` opens a new session under a fresh id; an id continues that
        session, from memory or storage. The session counts as in use from
        the lookup on, before its lock is taken, so it cannot leave memory
        while an ask waits for it or runs in it.

        Raises:
            SessionNotFoundError: The id is not one this agent issued, or the
                session is gone (memory only, and the process restarted).
        """
        session = self._acquire(session_id)
        try:
            with session.lock:
                yield session
        finally:
            with self._lock:
                session.users -= 1
                self._evict()

    def save(self, session: Session) -> None:
        """Write the session's conversation to storage; nothing without storage."""
        if self._storage is not None:
            self._storage.save(self._prefix + session.id, session.conversation)

    def _acquire(self, session_id: str | None) -> Session:
        """The session ``session_id`` (``None``: a new one), counted as in use, most recent."""
        if session_id is not None:
            _check_id(session_id)
        with self._lock:
            session = self._find(session_id)
            self._live[session.id] = session
            self._live.move_to_end(session.id)
            session.users += 1
            self._evict()
        return session

    def _find(self, session_id: str | None) -> Session:
        """A new session for ``None``; else the session in memory, else the stored one."""
        if session_id is None:
            return Session(str(uuid.uuid4()), self._conversation())
        return self._live.get(session_id) or self._load(session_id)

    def _load(self, session_id: str) -> Session:
        """The stored session ``session_id``."""
        if self._storage is None:
            raise SessionNotFoundError(f"no session {session_id}")
        try:
            stored = self._storage.load(self._prefix + session_id)
        except NotFoundError:
            raise SessionNotFoundError(f"no session {session_id}") from None
        conversation = Conversation.from_dict(
            {"messages": stored.messages},
            self._lg,
            config=self._kelt_config,
            compactor=SlidingWindowCompactor(),
        )
        return Session(session_id, conversation)

    def _conversation(self) -> Conversation:
        """An empty conversation on the configured limits."""
        return Conversation(self._lg, config=self._kelt_config, compactor=SlidingWindowCompactor())

    def _evict(self) -> None:
        """Drop the least recently used idle sessions beyond ``max_live``; only when they persist.

        A session in use stays: dropping it would let another ask reload a
        second copy of it, beside the one in use. Called under the table's lock.
        """
        excess = len(self._live) - self._config.max_live
        if self._storage is None or excess <= 0:
            return
        for session_id in [sid for sid, s in self._live.items() if s.users == 0][:excess]:
            del self._live[session_id]


def _check_id(session_id: str) -> None:
    """Reject anything but a uuid in canonical form, the only ids sessions are issued under."""
    try:
        canonical = str(uuid.UUID(session_id))
    except (ValueError, TypeError, AttributeError):
        canonical = None
    if canonical != session_id:
        raise SessionNotFoundError(f"no session {session_id!r}")
