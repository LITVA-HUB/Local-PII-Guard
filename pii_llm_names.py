from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from pii_models import (
    Entity,
    PIIGatewayPolicy,
    ENTITY_NAME,
    Span,
    overlaps,
)

try:
    from llama_cpp import Llama  # type: ignore
except Exception:
    Llama = None  # type: ignore


LETTER_CLASS = r"A-Za-zА-Яа-яЁё"

# Частые слова, которые "похожи на имя", но именами почти никогда не являются.
BAN_WORDS = {
    "меня", "я", "мы", "вы",
    "клиент", "покупатель", "заказ", "договор", "акт", "счет", "счёт", "накладная",
    "invoice", "order", "id", "номер",
    "телефон", "тел", "почта", "email", "e-mail",
    "зовут", "здравствуйте", "привет", "уважаемый", "уважаемая",
}

_WORD_RE = re.compile(rf"^[A-ZА-ЯЁ][{LETTER_CLASS}]+(?:-[A-ZА-ЯЁ][{LETTER_CLASS}]+)?$")


def _mask_ranges_same_length(text: str, ranges: List[Span], mask_char: str) -> str:
    if not ranges:
        return text
    ranges = sorted(ranges)
    out_parts: List[str] = []
    pos = 0
    for s, e in ranges:
        if s < pos:
            continue
        out_parts.append(text[pos:s])
        out_parts.append(mask_char * (e - s))
        pos = e
    out_parts.append(text[pos:])
    return "".join(out_parts)


def _chunk(text: str, max_chars: int, overlap_chars: int) -> Iterable[Tuple[str, int]]:
    n = len(text)
    if n <= max_chars:
        yield text, 0
        return

    overlap_chars = max(0, min(overlap_chars, max_chars // 2))
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        cut = end
        if end < n:
            window = text[start:end]
            for sep in ["\n", ". ", "! ", "? ", " ", "\t"]:
                idx = window.rfind(sep)
                if idx > 200:
                    cut = start + idx + len(sep)
                    break
        chunk = text[start:cut]
        yield chunk, start
        if cut >= n:
            break
        start = max(0, cut - overlap_chars)


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    s = s.replace("```json", "```").replace("```JSON", "```")
    if s.startswith("```") and s.endswith("```"):
        s = s.strip("`").strip()

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = s[start:end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def _is_initials_token(tok: str) -> bool:
    ts = tok.replace(" ", "")
    return bool(re.fullmatch(r"(?:[A-ZА-ЯЁ]\.){2,3}", ts))


def _clean_tokens(text: str) -> List[str]:
    t = " ".join(text.strip().split())
    toks = [tok.strip(" ,;:!?\"'()[]{}") for tok in t.split()]
    return [tok for tok in toks if tok]


def _looks_like_person_name(text: str) -> bool:
    """
    Строгий валидатор имени для PII-шлюза.

    Принимаем:
      - "Имя Фамилия" (2-4 слова, каждое с заглавной)
      - "Фамилия И.И." (фамилия + инициалы)
    Отбрасываем:
      - одиночные слова
      - служебные слова (клиент/меня/зовут/...)
      - токены/маски/цифры
    """
    t = " ".join(text.strip().split())
    if len(t) < 3:
        return False
    if any(ch.isdigit() for ch in t):
        return False
    if "<<" in t or ">>" in t or "[" in t or "]" in t:
        return False
    if "█" in t or "D" in t:
        return False
    if not re.search(rf"[{LETTER_CLASS}]", t):
        return False

    toks = _clean_tokens(t)
    if len(toks) < 2:
        return False

    if toks[0].lower() in BAN_WORDS:
        return False

    # 2-4 words (allow hyphens)
    if all(_WORD_RE.fullmatch(w) for w in toks):
        return 2 <= len(toks) <= 4

    # surname + initials
    if len(toks) == 2 and _WORD_RE.fullmatch(toks[0]) and _is_initials_token(toks[1]):
        return True

    return False


def _token_to_pattern(tok: str) -> str:
    esc = re.escape(tok)
    esc = esc.replace(r"\.", r"\.\s*")  # initials: allow spaces after dots
    return esc


def _build_name_search_regex(name_text: str) -> Optional[re.Pattern[str]]:
    """
    Делает regex, который ищет `name_text` в исходном куске:
      - слова соединяем через \s+ (терпим разные пробелы)
      - инициалы терпят пробелы после точек
      - границы по буквам, чтобы не матчить внутри слов
    """
    name_text = " ".join(name_text.strip().split())
    if not name_text or len(name_text) > 120:
        return None

    toks = _clean_tokens(name_text)
    if len(toks) < 2:
        return None

    inner = r"\s+".join(_token_to_pattern(t) for t in toks)
    pat = rf"(?<![{LETTER_CLASS}]){inner}(?![{LETTER_CLASS}])"
    try:
        return re.compile(pat, flags=re.IGNORECASE)
    except re.error:
        return None


def heuristic_name_spans(text: str) -> List[Tuple[int, int]]:
    """
    Консервативный fallback для имен (важно: OVERLAP-AWARE).
    Ищем:
      - "Имя Фамилия" / "Фамилия Имя" (две заглавные подряд)
      - "Иванов И.И." (фамилия + инициалы)
    """
    spans: set[Tuple[int, int]] = set()

    # OVERLAP-AWARE: используем lookahead, чтобы ловить перекрывающиеся пары.
    rx_full = re.compile(
        rf"(?=(?<![{LETTER_CLASS}])([A-ZА-ЯЁ][{LETTER_CLASS}]+)\s+([A-ZА-ЯЁ][{LETTER_CLASS}]+)(?![{LETTER_CLASS}]))"
    )
    for m in rx_full.finditer(text):
        w1 = m.group(1)
        w2 = m.group(2)
        if w1.lower() in BAN_WORDS or w2.lower() in BAN_WORDS:
            continue
        spans.add((m.start(1), m.end(2)))

    rx_init = re.compile(
        rf"(?=(?<![{LETTER_CLASS}])([A-ZА-ЯЁ][{LETTER_CLASS}]+)\s+([A-ZА-ЯЁ]\.\s*[A-ZА-ЯЁ]\.)(?![{LETTER_CLASS}]))"
    )
    for m in rx_init.finditer(text):
        w1 = m.group(1)
        if w1.lower() in BAN_WORDS:
            continue
        spans.add((m.start(1), m.end(2)))

    return sorted(spans)


@dataclass
class LocalNameExtractor:
    model_path: str
    policy: PIIGatewayPolicy
    llama_kwargs: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if Llama is None:
            raise RuntimeError("llama_cpp is not installed. Install llama-cpp-python.")
        self._lock = threading.Lock()
        kwargs = self.llama_kwargs or {}
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=kwargs.pop("n_gpu_layers", -1),
            n_ctx=kwargs.pop("n_ctx", self.policy.llm_n_ctx),
            verbose=kwargs.pop("verbose", False),
            **kwargs,
        )

    def _make_llm_input(self, text: str, forbidden_spans: List[Span]) -> str:
        t = _mask_ranges_same_length(text, forbidden_spans, self.policy.llm_mask_char)
        t = re.sub(r"\d", self.policy.llm_digit_char, t)
        return t

    def _call_llm(self, prompt: str) -> str:
        with self._lock:
            out = self.llm(
                prompt,
                max_tokens=self.policy.llm_max_tokens,
                stop=["<|im_end|>"],
                temperature=self.policy.llm_temperature,
                top_p=self.policy.llm_top_p,
            )
        return out["choices"][0]["text"]

    def _extract_from_chunk(self, chunk_text: str) -> Tuple[List[Union[str, Dict[str, Any]]], bool]:
        system_prompt = (
            "Ты — строгий NER-экстрактор ИМЕН ЛЮДЕЙ.\n"
            "Найди в тексте все ФИО/имя+фамилия/фамилия+инициалы.\n"
            "В тексте могут быть замены: цифры заменены на 'D', приватные данные могут быть замаскированы '█'.\n"
            "Не извлекай токены вида <<TYPE_N>> и не извлекай то, что содержит █ или D.\n\n"
            "Верни СТРОГО JSON без пояснений:\n"
            "{\"names\": [{\"start\": 0, \"end\": 10, \"text\": \"Иванов Иван\"}]}\n"
            "start/end — индексы символов в этом тексте, end не включается.\n"
            "text должен совпадать с подстрокой text[start:end] ИЛИ быть точным именем без служебных слов.\n"
            "Если имен нет — {\"names\": []}.\n"
        )

        prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{chunk_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        try:
            raw = self._call_llm(prompt).strip()
        except Exception:
            return [], False

        obj = _extract_json(raw)
        if not obj:
            return [], False

        names = obj.get("names", [])
        if not isinstance(names, list):
            return [], False

        return names, True

    def extract(self, text: str, *, forbidden_spans: List[Span]) -> Tuple[List[Entity], List[str]]:
        warnings: List[str] = []
        if not self.policy.enable_llm_names:
            return [], ["name_llm_disabled"]

        masked = self._make_llm_input(text, forbidden_spans)

        out: List[Entity] = []
        parse_fail_chunks = 0
        total_chunks = 0

        def add_entity(start: int, end: int, value: str, detector: str, conf: float) -> None:
            if start < 0 or end <= start or end > len(text):
                return
            if any(overlaps((start, end), fs) for fs in forbidden_spans):
                return
            v = value
            if not _looks_like_person_name(v):
                return
            out.append(
                Entity(
                    ENTITY_NAME,
                    start,
                    end,
                    v,
                    method="llm",
                    detector=detector,
                    confidence=conf,
                    priority=40,
                )
            )

        def norm_spaces(s: str) -> str:
            return " ".join(s.strip().split()).lower()

        for chunk, offset in _chunk(masked, self.policy.llm_max_chars_per_chunk, self.policy.llm_chunk_overlap):
            total_chunks += 1
            original_chunk = text[offset: offset + len(chunk)]

            candidates, ok = self._extract_from_chunk(chunk)
            if not ok:
                parse_fail_chunks += 1
                continue

            for cand in candidates:
                # Иногда модели возвращают просто строки, а не dict
                if isinstance(cand, str):
                    cand_text = cand.strip()
                    if not _looks_like_person_name(cand_text):
                        continue
                    rx = _build_name_search_regex(cand_text)
                    if rx is None:
                        continue
                    for m in rx.finditer(original_chunk):
                        s = m.start() + offset
                        e = m.end() + offset
                        add_entity(s, e, text[s:e], "llm:name_str_search", 0.75)
                    continue

                if not isinstance(cand, dict):
                    continue

                cand_text = str(cand.get("text", "")).strip()
                try:
                    s0 = int(cand.get("start", -1))
                    e0 = int(cand.get("end", -1))
                except Exception:
                    s0, e0 = -1, -1

                # Вариант 1: доверяем span ТОЛЬКО если подстрока реально выглядит как имя
                # и (если есть text) совпадает с text по нормализованным пробелам.
                if 0 <= s0 < e0 <= len(original_chunk):
                    sub = original_chunk[s0:e0]
                    if _looks_like_person_name(sub):
                        if (not cand_text) or (norm_spaces(sub) == norm_spaces(cand_text)):
                            s = s0 + offset
                            e = e0 + offset
                            add_entity(s, e, text[s:e], "llm:span_exact", 0.9)
                            continue

                # Вариант 2: snap-to-substring — индексы кривые, ищем cand_text в исходнике.
                if cand_text and _looks_like_person_name(cand_text):
                    rx = _build_name_search_regex(cand_text)
                    if rx is None:
                        continue
                    found_any = False
                    for m in rx.finditer(original_chunk):
                        found_any = True
                        s = m.start() + offset
                        e = m.end() + offset
                        add_entity(s, e, text[s:e], "llm:text_search", 0.85)
                    if found_any:
                        continue

                # Вариант 3 (span-only, но не проходит валидатор) — игнорируем, чтобы не было "Меня з<<NAME>>..."

        if total_chunks > 0 and parse_fail_chunks == total_chunks:
            warnings.append("name_llm_parse_failed")

        # дедуп по span
        uniq: Dict[Tuple[int, int], Entity] = {}
        for e in out:
            key = (e.start, e.end)
            if key not in uniq or uniq[key].confidence < e.confidence:
                uniq[key] = e

        entities = sorted(list(uniq.values()), key=lambda x: (x.start, x.end))
        return entities, warnings