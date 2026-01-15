from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Pattern

from pii_models import (
    Entity,
    PIIGatewayPolicy,
    ENTITY_EMAIL,
    ENTITY_PHONE,
    ENTITY_CARD,
    ENTITY_SNILS,
    ENTITY_PASSPORT,
    ENTITY_DATE,
)

Validator = Callable[[re.Match[str], str], bool]
ContextFilter = Callable[[str, int, int], bool]


def is_luhn_valid(number: str) -> bool:
    digits = [int(d) for d in number]
    checksum = 0
    rev = digits[::-1]
    for i, d in enumerate(rev):
        if i % 2 == 1:
            d2 = d * 2
            checksum += d2 if d2 < 10 else d2 - 9
        else:
            checksum += d
    return checksum % 10 == 0


@dataclass
class RegexDetector:
    name: str
    entity_type: str
    pattern: Pattern[str]
    priority: int
    validator: Optional[Validator] = None
    context_filter: Optional[ContextFilter] = None

    def find(self, text: str) -> List[Entity]:
        out: List[Entity] = []
        for m in self.pattern.finditer(text):
            s, e = m.start(), m.end()
            if s >= e:
                continue
            if self.validator is not None:
                try:
                    if not self.validator(m, text):
                        continue
                except Exception:
                    continue
            if self.context_filter is not None:
                try:
                    if not self.context_filter(text, s, e):
                        continue
                except Exception:
                    continue

            out.append(
                Entity(
                    entity_type=self.entity_type,
                    start=s,
                    end=e,
                    value=m.group(0),
                    method="regex",
                    detector=self.name,
                    confidence=1.0,
                    priority=self.priority,
                )
            )
        return out


def build_default_regex_detectors(policy: PIIGatewayPolicy) -> List[RegexDetector]:
    dets: List[RegexDetector] = []

    # EMAIL
    dets.append(
        RegexDetector(
            name="email",
            entity_type=ENTITY_EMAIL,
            pattern=re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            priority=100,
        )
    )

    # SNILS
    dets.append(
        RegexDetector(
            name="snils",
            entity_type=ENTITY_SNILS,
            pattern=re.compile(r"\b\d{3}-\d{3}-\d{3}[ -]\d{2}\b"),
            priority=90,
        )
    )

    # CARD (13-19 digits w/ separators) + Luhn + optional context
    # Важно: границы (?!\d)/(?<!\d) запрещают матчить кусок внутри длинной цифровой строки.
    card_pattern = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")

    def card_validator(m: re.Match[str], text: str) -> bool:
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 13 or len(digits) > 19:
            return False
        if not is_luhn_valid(digits):
            return False
        if policy.card_require_context:
            w = policy.card_context_window
            lo = max(0, m.start() - w)
            hi = min(len(text), m.end() + w)
            window = text[lo:hi].lower()
            return any(k in window for k in policy.card_context_words)
        return True

    dets.append(
        RegexDetector(
            name="card_luhn",
            entity_type=ENTITY_CARD,
            pattern=card_pattern,
            priority=80,
            validator=card_validator,
        )
    )

    # PHONE RU
    # FIX: запрет на матч "внутри числа" (это именно твои FP по заказам).
    phone_pattern = re.compile(
        r"(?<!\d)(?:\+7|8)[\s\(.-]?\d{3}[\s\).-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}(?!\d)"
    )

    def phone_validator(m: re.Match[str], _: str) -> bool:
        digits = re.sub(r"\D", "", m.group(0))
        return len(digits) == 11 and digits[0] in ("7", "8")

    dets.append(
        RegexDetector(
            name="phone_ru",
            entity_type=ENTITY_PHONE,
            pattern=phone_pattern,
            priority=70,
            validator=phone_validator,
        )
    )

    # PASSPORT RU 4+6, with optional context filter
    passport_pattern = re.compile(r"\b\d{4}[\s-]?\d{6}\b")

    def passport_context_ok(text: str, start: int, end: int) -> bool:
        if not policy.passport_use_context_filter:
            return True
        w = policy.passport_context_window
        lo = max(0, start - w)
        hi = min(len(text), end + w)
        window = text[lo:hi].lower()

        if any(k in window for k in policy.passport_allowlist):
            return True
        if any(k in window for k in policy.passport_blocklist):
            return False
        return True

    dets.append(
        RegexDetector(
            name="passport_ru",
            entity_type=ENTITY_PASSPORT,
            pattern=passport_pattern,
            priority=60,
            context_filter=passport_context_ok,
        )
    )

    # DATE dd.mm.yyyy (1900-2099)
    date_pattern = re.compile(
        r"\b(?:0[1-9]|[12][0-9]|3[01])[\.\/-](?:0[1-9]|1[012])[\.\/-](?:19|20)\d{2}\b"
    )
    dets.append(
        RegexDetector(
            name="date_dmy",
            entity_type=ENTITY_DATE,
            pattern=date_pattern,
            priority=50,
        )
    )

    return dets