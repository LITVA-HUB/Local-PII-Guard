from __future__ import annotations

import time
import re
from typing import Any, Dict, List, Optional

from pii_models import (
    Entity,
    PIIGatewayPolicy,
    TOKEN_PATTERN,
    resolve_overlaps,
    overlaps,
    ENTITY_NAME,
)
from pii_detectors import build_default_regex_detectors
from pii_vault import InMemoryPIIVault
from pii_llm_names import LocalNameExtractor, heuristic_name_spans


def _perf() -> float:
    return time.perf_counter()


class PIIGateway:
    def __init__(
        self,
        *,
        model_path: str = "Qwen2.5-3B-Instruct-IQ3_M.gguf",
        policy: Optional[PIIGatewayPolicy] = None,
        vault: Optional[InMemoryPIIVault] = None,
        llama_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.policy = policy or PIIGatewayPolicy()
        self.vault = vault or InMemoryPIIVault(self.policy)
        self._regex_detectors = build_default_regex_detectors(self.policy)

        self._name_extractor: Optional[LocalNameExtractor] = None
        if self.policy.enable_llm_names and ENTITY_NAME in self.policy.enabled_types:
            try:
                self._name_extractor = LocalNameExtractor(
                    model_path=model_path,
                    policy=self.policy,
                    llama_kwargs=llama_kwargs or {},
                )
            except Exception:
                if self.policy.fail_closed:
                    raise
                self._name_extractor = None
                self.policy.enable_llm_names = False

    def tokenize(self, session_id: str, text: str) -> Dict[str, Any]:
        t0 = _perf()
        if not isinstance(text, str):
            text = str(text)

        warnings: List[str] = []

        # 1) Regex (структурные ПДн)
        t1 = _perf()
        regex_entities: List[Entity] = []
        for det in self._regex_detectors:
            if det.entity_type not in self.policy.enabled_types:
                continue
            regex_entities.extend(det.find(text))
        t2 = _perf()

        regex_entities = resolve_overlaps(regex_entities)
        forbidden_spans = [e.span() for e in regex_entities]

        # 2) Names: LLM + ALWAYS heuristic backup
        t3 = _perf()
        name_warnings: List[str] = []
        llm_name_entities: List[Entity] = []
        heuristic_entities: List[Entity] = []

        if ENTITY_NAME in self.policy.enabled_types:
            # LLM part
            if self._name_extractor is None:
                name_warnings.append("name_llm_unavailable")
            else:
                try:
                    llm_name_entities, name_warnings = self._name_extractor.extract(text, forbidden_spans=forbidden_spans)
                except Exception:
                    name_warnings = ["name_llm_exception"]
                    llm_name_entities = []

            failed_hard = any(w in ("name_llm_parse_failed", "name_llm_exception") for w in name_warnings)
            if failed_hard and self.policy.name_fallback == "raise" and self.policy.fail_closed:
                raise RuntimeError(f"Name extraction failed (fail-closed): {name_warnings}")

            # Heuristic ALWAYS (потому что у тебя именно "missing token" и "кривые спаны")
            for s, e in heuristic_name_spans(text):
                if any(overlaps((s, e), fs) for fs in forbidden_spans):
                    continue
                heuristic_entities.append(
                    Entity(
                        ENTITY_NAME,
                        s,
                        e,
                        text[s:e],
                        method="heuristic",
                        detector="heuristic:name",
                        confidence=0.6,
                        priority=30,
                    )
                )

            # Merge: LLM выше приоритетом, heuristic закрывает дыры
            llm_name_entities = [
                ne for ne in llm_name_entities
                if not any(overlaps(ne.span(), fs) for fs in forbidden_spans)
            ]
            name_entities = resolve_overlaps(list(llm_name_entities) + list(heuristic_entities))
        else:
            name_entities = []

        warnings.extend(name_warnings)
        t4 = _perf()

        # 3) Merge all + resolve overlaps
        entities = resolve_overlaps(list(regex_entities) + list(name_entities))

        # 4) Replace from end to start
        t5 = _perf()
        entities_sorted = sorted(entities, key=lambda e: e.start, reverse=True)

        tokenized = text
        result_entities: List[Dict[str, Any]] = []
        vault_delta: Dict[str, str] = {}

        for e in entities_sorted:
            if e.entity_type not in self.policy.enabled_types:
                continue

            token, stored_value, is_new = self.vault.get_or_create_token(session_id, e.entity_type, e.value)
            tokenized = tokenized[:e.start] + token + tokenized[e.end:]

            if is_new:
                vault_delta[token] = stored_value

            rec: Dict[str, Any] = {
                "type": e.entity_type,
                "start": e.start,
                "end": e.end,
                "token": token,
                "method": e.method,
                "detector": e.detector,
                "confidence": round(float(e.confidence), 4),
            }
            if self.policy.include_raw_values_in_result:
                rec["value"] = stored_value
            result_entities.append(rec)

        result_entities.sort(key=lambda x: x["start"])
        t6 = _perf()

        return {
            "session_id": session_id,
            "tokenized_text": tokenized,
            "entities": result_entities,
            "vault_delta": vault_delta,
            "meta": {
                "warnings": warnings,
                "timings": {
                    "regex_s": round(t2 - t1, 6),
                    "names_s": round(t4 - t3, 6),
                    "replace_s": round(t6 - t5, 6),
                    "total_s": round(_perf() - t0, 6),
                },
            },
        }

    def restore(self, session_id: str, text_with_tokens: str, *, allowed_types: Optional[List[str]] = None) -> str:
        if not isinstance(text_with_tokens, str):
            text_with_tokens = str(text_with_tokens)

        allowed = None if allowed_types is None else set(allowed_types)

        def repl(m: re.Match[str]) -> str:
            token = m.group(0)
            typ = m.group(1)
            if allowed is not None and typ not in allowed:
                return token
            val = self.vault.get_value(session_id, token)
            return val if val is not None else token

        return TOKEN_PATTERN.sub(repl, text_with_tokens)

    def export_fields(self, session_id: str) -> Dict[str, List[str]]:
        return self.vault.export_values_by_type(session_id)

    def tokenize_payload(self, session_id: str, payload: Any) -> Any:
        if payload is None:
            return None
        if isinstance(payload, str):
            return self.tokenize(session_id, payload)["tokenized_text"]
        if isinstance(payload, list):
            return [self.tokenize_payload(session_id, x) for x in payload]
        if isinstance(payload, dict):
            return {k: self.tokenize_payload(session_id, v) for k, v in payload.items()}
        return payload

    def restore_payload(self, session_id: str, payload: Any, *, allowed_types: Optional[List[str]] = None) -> Any:
        if payload is None:
            return None
        if isinstance(payload, str):
            return self.restore(session_id, payload, allowed_types=allowed_types)
        if isinstance(payload, list):
            return [self.restore_payload(session_id, x, allowed_types=allowed_types) for x in payload]
        if isinstance(payload, dict):
            return {k: self.restore_payload(session_id, v, allowed_types=allowed_types) for k, v in payload.items()}
        return payload

    @property
    def cloud_system_prompt(self) -> str:
        return (
            "В тексте могут встречаться приватные токены вида <<TYPE_N>> (например <<NAME_1>>, <<PHONE_1>>).\n"
            "Эти токены заменяют персональные данные и НЕ должны быть изменены.\n"
            "Правила:\n"
            "1) Никогда не раскрывай и не пытайся угадать значения токенов.\n"
            "2) Не изменяй токены (не добавляй пробелы, не меняй регистр, не удаляй символы < > _).\n"
            "3) Используй токены как есть, в том числе для обращения (например: 'Здравствуйте, <<NAME_1>>').\n"
            "4) Не проси повторно телефон/email/паспорт/карту, если уже есть соответствующий токен.\n"
        )