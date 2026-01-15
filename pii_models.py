from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

# Supported entity types (internal IDs used in tokens)
ENTITY_EMAIL = "EMAIL"
ENTITY_PHONE = "PHONE"
ENTITY_CARD = "CARD"
ENTITY_SNILS = "SNILS"
ENTITY_PASSPORT = "PASSPORT"
ENTITY_DATE = "DATE"
ENTITY_NAME = "NAME"

DEFAULT_ENABLED_TYPES = {
    ENTITY_EMAIL,
    ENTITY_PHONE,
    ENTITY_CARD,
    ENTITY_SNILS,
    ENTITY_PASSPORT,
    ENTITY_DATE,
    ENTITY_NAME,
}

TOKEN_PATTERN = re.compile(r"<<([A-Z]+)_([0-9]+)>>")

Span = Tuple[int, int]


@dataclass
class Entity:
    entity_type: str
    start: int
    end: int
    value: str

    method: str = "regex"          # regex | llm | heuristic
    detector: str = ""
    confidence: float = 1.0
    priority: int = 50

    def span(self) -> Span:
        return (self.start, self.end)


def overlaps(a: Span, b: Span) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])


def resolve_overlaps(entities: Sequence[Entity]) -> List[Entity]:
    """
    Resolve overlaps by selecting the best entity in each overlapping group.
    "Best" = higher priority, then longer span, then higher confidence.

    This group-based greedy resolver is good enough for typical PII patterns.
    """
    if not entities:
        return []

    ents = sorted(entities, key=lambda e: (e.start, e.end))
    out: List[Entity] = []

    group: List[Entity] = []
    group_end = -1

    def pick_best(g: List[Entity]) -> Entity:
        return max(g, key=lambda e: (e.priority, e.end - e.start, e.confidence))

    for e in ents:
        if not group:
            group = [e]
            group_end = e.end
            continue

        if e.start >= group_end:
            out.append(pick_best(group))
            group = [e]
            group_end = e.end
        else:
            group.append(e)
            group_end = max(group_end, e.end)

    if group:
        out.append(pick_best(group))

    out.sort(key=lambda e: (e.start, e.end))
    return out


@dataclass
class PIIGatewayPolicy:
    """
    High-level policy controlling what we detect, how we tokenize, and how we fail.

    Core:
    - Tokenize enabled PII types into <<TYPE_N>>
    - Store token->value mapping locally (vault)
    - Restore tokens back for UI/messages/CRM export

    Security knobs:
    - fail_closed: if True, you prefer raising if name extraction is unavailable and NAME is enabled.
    """
    enabled_types: set = field(default_factory=lambda: set(DEFAULT_ENABLED_TYPES))

    # Token formatting
    token_prefix: str = "<<"
    token_suffix: str = ">>"
    token_separator: str = "_"

    # Fail-closed behavior
    fail_closed: bool = True

    # Local LLM names (NER)
    enable_llm_names: bool = True
    llm_n_ctx: int = 4096
    llm_max_tokens: int = 256
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0

    # Chunking (characters) for long texts
    llm_max_chars_per_chunk: int = 8000
    llm_chunk_overlap: int = 200

    # Blindfolding for local LLM (reduce hallucinations / avoid digits)
    llm_mask_char: str = "█"
    llm_digit_char: str = "D"

    # If LLM fails or returns garbage, what to do for names:
    # "heuristic" => conservative fallback regex-based
    # "skip"      => do nothing
    # "raise"     => raise (if fail_closed=True)
    name_fallback: str = "heuristic"

    # Passport context heuristics to reduce false positives in corporate text
    passport_context_window: int = 30
    passport_blocklist: Tuple[str, ...] = (
        "договор", "счет", "счёт", "акт", "наклад", "заказ", "invoice", "order", "id", "№", "номер"
    )
    passport_allowlist: Tuple[str, ...] = ("паспорт", "серия", "выдан")
    passport_use_context_filter: bool = True

    # Card context filter (optional)
    card_require_context: bool = False
    card_context_window: int = 30
    card_context_words: Tuple[str, ...] = (
        "карта", "card", "visa", "mastercard", "мир", "оплата", "payment"
    )

    # Integration convenience
    include_raw_values_in_result: bool = True  # keep local only


def _digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def normalize_for_key(entity_type: str, raw: str) -> str:
    """
    Normalization used ONLY for stable token assignment within a session.
    """
    s = raw.strip()
    if entity_type == ENTITY_EMAIL:
        return s.lower()
    if entity_type == ENTITY_PHONE:
        d = _digits_only(s)
        if len(d) == 11 and d[0] == "8":
            d = "7" + d[1:]
        return d
    if entity_type in (ENTITY_CARD, ENTITY_PASSPORT, ENTITY_SNILS):
        return _digits_only(s)
    if entity_type == ENTITY_NAME:
        return " ".join(s.split()).lower()
    if entity_type == ENTITY_DATE:
        return s
    return s.lower()


def format_for_storage(entity_type: str, raw: str) -> str:
    """
    Canonical formatting for what we store and restore back into outgoing text.
    """
    s = raw.strip()
    if entity_type == ENTITY_EMAIL:
        return s.lower()
    if entity_type == ENTITY_PHONE:
        d = _digits_only(s)
        if len(d) == 11 and d[0] == "8":
            d = "7" + d[1:]
        if len(d) == 11 and d[0] == "7":
            return "+7" + d[1:]
        return s
    if entity_type == ENTITY_CARD:
        d = _digits_only(s)
        if len(d) >= 12:
            groups = [d[i:i+4] for i in range(0, len(d), 4)]
            return " ".join(groups)
        return s
    if entity_type == ENTITY_PASSPORT:
        d = _digits_only(s)
        if len(d) == 10:
            return d[:4] + " " + d[4:]
        return s
    if entity_type == ENTITY_SNILS:
        d = _digits_only(s)
        if len(d) == 11:
            return f"{d[:3]}-{d[3:6]}-{d[6:9]} {d[9:]}"
        return s
    if entity_type == ENTITY_NAME:
        return " ".join(s.split())
    return s