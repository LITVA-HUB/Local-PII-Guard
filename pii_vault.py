from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

from pii_models import (
    PIIGatewayPolicy,
    TOKEN_PATTERN,
    format_for_storage,
    normalize_for_key,
)


@dataclass
class SessionVault:
    token_to_value: Dict[str, str] = field(default_factory=dict)     # token -> value
    token_to_type: Dict[str, str] = field(default_factory=dict)      # token -> type
    value_to_token: Dict[str, Dict[str, str]] = field(default_factory=dict)  # type -> (norm_value -> token)
    counters: Dict[str, int] = field(default_factory=dict)           # type -> counter


class InMemoryPIIVault:
    """
    Local vault (in-memory).
    Stores token/value mapping per session_id, with separate counters per entity type.
    """

    def __init__(self, policy: PIIGatewayPolicy):
        self.policy = policy
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionVault] = {}

    def _get_session(self, session_id: str) -> SessionVault:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionVault()
        return self._sessions[session_id]

    def get_or_create_token(self, session_id: str, entity_type: str, raw_value: str) -> Tuple[str, str, bool]:
        """
        Returns (token, stored_value, is_new).
        stored_value is what will be restored into text later (may be canonical formatted).
        """
        key = normalize_for_key(entity_type, raw_value)
        stored_value = format_for_storage(entity_type, raw_value)

        with self._lock:
            s = self._get_session(session_id)
            per_type = s.value_to_token.setdefault(entity_type, {})
            if key in per_type:
                tok = per_type[key]
                return tok, s.token_to_value[tok], False

            next_i = s.counters.get(entity_type, 0) + 1
            s.counters[entity_type] = next_i
            tok = f"{self.policy.token_prefix}{entity_type}{self.policy.token_separator}{next_i}{self.policy.token_suffix}"

            per_type[key] = tok
            s.token_to_value[tok] = stored_value
            s.token_to_type[tok] = entity_type
            return tok, stored_value, True

    def get_value(self, session_id: str, token: str) -> Optional[str]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return None
            return s.token_to_value.get(token)

    def get_type(self, session_id: str, token: str) -> Optional[str]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return None
            return s.token_to_type.get(token)

    def list_tokens(self, session_id: str) -> Dict[str, str]:
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return {}
            return dict(s.token_to_value)

    def export_values_by_type(self, session_id: str) -> Dict[str, List[str]]:
        """
        Values grouped by type, ordered by token index.
        """
        with self._lock:
            s = self._sessions.get(session_id)
            if not s:
                return {}
            by_type: Dict[str, List[Tuple[int, str]]] = {}
            for tok, val in s.token_to_value.items():
                typ = s.token_to_type.get(tok, "UNKNOWN")
                m = TOKEN_PATTERN.fullmatch(tok)
                idx = int(m.group(2)) if m else 0
                by_type.setdefault(typ, []).append((idx, val))

            out: Dict[str, List[str]] = {}
            for typ, items in by_type.items():
                items.sort(key=lambda x: x[0])
                out[typ] = [v for _, v in items]
            return out

    def reset_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)