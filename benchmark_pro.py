# benchmark_pro.py
# Улучшенный бенчмарк:
# - разные количества кейсов по категориям (дешёвые regex — много, LLM-имена — меньше)
# - несколько seed прогонов (проверка устойчивости)
# - отчёт по категориям + общий отчёт
# - отдельные счётчики: leaks / missing_token / fp_token / fp_passport / restore_missing
# - latency p50/p95 по прогону и в сумме

from __future__ import annotations

import random
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from faker import Faker

from pii_agent_pro import PIIGateway
from pii_models import (
    ENTITY_EMAIL,
    ENTITY_PHONE,
    ENTITY_CARD,
    ENTITY_SNILS,
    ENTITY_PASSPORT,
    ENTITY_DATE,
    ENTITY_NAME,
    TOKEN_PATTERN,
    PIIGatewayPolicy,
)
from pii_detectors import is_luhn_valid


# -------------------------
# Utils
# -------------------------
def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    k = int(round((p / 100.0) * (len(v) - 1)))
    k = max(0, min(k, len(v) - 1))
    return v[k]


def re_search(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def gen_non_luhn_digits(length: int = 16) -> str:
    """Гарантированно НЕ проходит Луна (для теста FP по картам)."""
    while True:
        s = "".join(str(random.randint(0, 9)) for _ in range(length))
        if not is_luhn_valid(s):
            return s


# -------------------------
# Case model
# -------------------------
@dataclass
class Case:
    category: str
    text: str
    expected: List[Tuple[str, str]] = field(default_factory=list)
    # expected = list of (entity_type, raw_value_substring_expected_in_text)


# -------------------------
# Case builders
# -------------------------
def build_cases(
    fake: Faker,
    counts: Dict[str, int],
) -> List[Case]:
    """
    counts keys:
      pos_email, pos_phone, pos_card, pos_snils, pos_passport, pos_date, pos_name
      neg_fake_card_order, neg_passport_like_contract, hard_neg_numbers, mixed
    """
    cases: List[Case] = []

    n = counts

    for _ in range(n.get("pos_email", 0)):
        email = fake.email()
        cases.append(Case("pos_email", f"Пиши на {email} пожалуйста.", expected=[(ENTITY_EMAIL, email)]))

    for _ in range(n.get("pos_phone", 0)):
        phone = "+7" + "".join(str(random.randint(0, 9)) for _ in range(10))
        cases.append(Case("pos_phone", f"Мой телефон {phone}, запиши.", expected=[(ENTITY_PHONE, phone)]))

    for _ in range(n.get("pos_card", 0)):
        card = fake.credit_card_number()
        cases.append(Case("pos_card", f"Оплата картой {card} прошла.", expected=[(ENTITY_CARD, card)]))

    for _ in range(n.get("neg_fake_card_order", 0)):
        order = gen_non_luhn_digits(16)
        cases.append(Case("neg_fake_card_order", f"Номер заказа {order}.", expected=[]))

    for _ in range(n.get("pos_snils", 0)):
        snils = f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(100, 999)} {random.randint(10, 99)}"
        cases.append(Case("pos_snils", f"СНИЛС: {snils}", expected=[(ENTITY_SNILS, snils)]))

    for _ in range(n.get("pos_passport", 0)):
        passport = f"{random.randint(1000, 9999)} {random.randint(100000, 999999)}"
        cases.append(Case("pos_passport", f"Паспорт: {passport}, выдан.", expected=[(ENTITY_PASSPORT, passport)]))

    for _ in range(n.get("neg_passport_like_contract", 0)):
        pseudo = f"{random.randint(1000, 9999)} {random.randint(100000, 999999)}"
        cases.append(
            Case(
                "neg_passport_like_contract",
                f"Договор № {pseudo} от 12.12.2020",
                expected=[(ENTITY_DATE, "12.12.2020")],
            )
        )

    for _ in range(n.get("pos_date", 0)):
        date = f"{random.randint(1, 28):02d}.{random.randint(1, 12):02d}.{random.randint(1950, 2023)}"
        cases.append(Case("pos_date", f"Дата рождения {date}", expected=[(ENTITY_DATE, date)]))

    for _ in range(n.get("pos_name", 0)):
        name = f"{fake.first_name_male()} {fake.last_name_male()}"
        cases.append(Case("pos_name", f"Меня зовут {name}, приятно познакомиться.", expected=[(ENTITY_NAME, name)]))

    for _ in range(n.get("hard_neg_numbers", 0)):
        ver = f"{random.randint(0, 9)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        price = str(random.randint(100, 5000))
        year = str(random.randint(1995, 2025))
        cases.append(Case("hard_neg_numbers", f"Версия {ver}, цена {price} руб, в {year} году.", expected=[]))

    for _ in range(n.get("mixed", 0)):
        email = fake.email()
        phone = "+7" + "".join(str(random.randint(0, 9)) for _ in range(10))
        name = f"{fake.first_name_male()} {fake.last_name_male()}"
        passport = f"{random.randint(1000, 9999)} {random.randint(100000, 999999)}"
        card = fake.credit_card_number()
        txt = f"Клиент {name}, email {email}, тел {phone}, паспорт {passport}, карта {card}."
        cases.append(
            Case(
                "mixed",
                txt,
                expected=[
                    (ENTITY_NAME, name),
                    (ENTITY_EMAIL, email),
                    (ENTITY_PHONE, phone),
                    (ENTITY_PASSPORT, passport),
                    (ENTITY_CARD, card),
                ],
            )
        )

    return cases


# -------------------------
# Evaluation
# -------------------------
@dataclass
class CategoryStats:
    total: int = 0
    passed: int = 0
    leaks: int = 0
    missing_token: int = 0
    fp_token: int = 0
    fp_passport: int = 0
    restore_missing: int = 0


@dataclass
class RunStats:
    seed: int
    total: int = 0
    passed: int = 0
    leaks: int = 0
    missing_token: int = 0
    fp_token: int = 0
    fp_passport: int = 0
    restore_missing: int = 0
    latencies: List[float] = field(default_factory=list)
    by_cat: Dict[str, CategoryStats] = field(default_factory=dict)


def ensure_cat(stats: RunStats, cat: str) -> CategoryStats:
    if cat not in stats.by_cat:
        stats.by_cat[cat] = CategoryStats()
    return stats.by_cat[cat]


def eval_case(
    gateway: PIIGateway,
    policy: PIIGatewayPolicy,
    session_id: str,
    case: Case,
    *,
    verbose_fail: bool = False,
) -> Tuple[bool, Dict[str, int], float]:
    """
    Returns:
      ok: bool
      counters: dict increments
      latency_s: float
    """
    t0 = time.perf_counter()
    res = gateway.tokenize(session_id, case.text)
    tokenized = res["tokenized_text"]
    restored = gateway.restore(session_id, tokenized)
    latency = float(res["meta"]["timings"]["total_s"]) if "meta" in res else (time.perf_counter() - t0)

    ok = True
    inc = {
        "leaks": 0,
        "missing_token": 0,
        "fp_token": 0,
        "fp_passport": 0,
        "restore_missing": 0,
    }

    # Expected: must remove raw + must have token for that type
    for typ, raw in case.expected:
        if raw in tokenized:
            ok = False
            inc["leaks"] += 1
            if verbose_fail:
                print(f"\n❌ LEAK ({case.category})")
                print("Input:    ", case.text)
                print("Tokenized: ", tokenized)
                print("Expected removed:", raw)

        if not re_search(fr"<<{typ}_[0-9]+>>", tokenized):
            # If we disabled names, don't fail for missing NAME
            if typ == ENTITY_NAME and not policy.enable_llm_names:
                pass
            else:
                ok = False
                inc["missing_token"] += 1
                if verbose_fail:
                    print(f"\n❌ MISSING TOKEN ({case.category})")
                    print("Input:    ", case.text)
                    print("Tokenized: ", tokenized)
                    print("Expected token type:", typ)

    # Negatives: must not contain any tokens at all
    if case.category in ("neg_fake_card_order", "hard_neg_numbers"):
        if TOKEN_PATTERN.search(tokenized):
            ok = False
            inc["fp_token"] += 1
            if verbose_fail:
                print(f"\n❌ FP TOKEN ({case.category})")
                print("Input:    ", case.text)
                print("Tokenized: ", tokenized)

    # Contract case: should NOT produce PASSPORT token, but should produce DATE (already in expected)
    if case.category == "neg_passport_like_contract":
        if re_search(fr"<<{ENTITY_PASSPORT}_[0-9]+>>", tokenized):
            ok = False
            inc["fp_passport"] += 1
            if verbose_fail:
                print(f"\n❌ FP PASSPORT TOKEN ({case.category})")
                print("Input:    ", case.text)
                print("Tokenized: ", tokenized)

    # Restore: must contain raw (CARD: digits-only match ok)
    for typ, raw in case.expected:
        if typ == ENTITY_NAME and not policy.enable_llm_names:
            continue

        if raw not in restored:
            if typ == ENTITY_CARD:
                if digits_only(raw) not in digits_only(restored):
                    ok = False
                    inc["restore_missing"] += 1
                    if verbose_fail:
                        print(f"\n❌ RESTORE MISSING ({case.category})")
                        print("Input:    ", case.text)
                        print("Tokenized: ", tokenized)
                        print("Restored:  ", restored)
                        print("Missing:", raw)
            else:
                ok = False
                inc["restore_missing"] += 1
                if verbose_fail:
                    print(f"\n❌ RESTORE MISSING ({case.category})")
                    print("Input:    ", case.text)
                    print("Tokenized: ", tokenized)
                    print("Restored:  ", restored)
                    print("Missing:", raw)

    return ok, inc, latency


def print_run_summary(stats: RunStats) -> None:
    mean = statistics.mean(stats.latencies) if stats.latencies else 0.0
    p50 = percentile(stats.latencies, 50)
    p95 = percentile(stats.latencies, 95)

    print("\n" + "=" * 74)
    print(f"RUN SUMMARY (seed={stats.seed})")
    print("=" * 74)
    print(f"Total cases: {stats.total}")
    print(f"Passed:      {stats.passed} ({(stats.passed / stats.total * 100.0) if stats.total else 0.0:.2f}%)")
    print(f"Leaks:       {stats.leaks}")
    print(f"MissingTok:  {stats.missing_token}")
    print(f"FP Token:    {stats.fp_token}")
    print(f"FP Passport: {stats.fp_passport}")
    print(f"RestoreMiss: {stats.restore_missing}")
    print(f"Latency: mean={mean:.4f}s p50={p50:.4f}s p95={p95:.4f}s")
    print("=" * 74)

    # Per-category table (sorted by worst pass rate)
    rows = []
    for cat, cs in stats.by_cat.items():
        pass_rate = (cs.passed / cs.total * 100.0) if cs.total else 0.0
        rows.append((pass_rate, cat, cs))
    rows.sort(key=lambda x: x[0])

    print("\nPer-category:")
    print(f"{'Category':<28} {'Total':>6} {'Pass%':>7} {'Leaks':>6} {'MissTok':>7} {'FPtok':>6} {'FPpass':>6} {'Rmiss':>6}")
    for pass_rate, cat, cs in rows:
        print(
            f"{cat:<28} {cs.total:>6} {pass_rate:>6.2f}% {cs.leaks:>6} {cs.missing_token:>7} "
            f"{cs.fp_token:>6} {cs.fp_passport:>6} {cs.restore_missing:>6}"
        )


def print_global_summary(all_runs: List[RunStats]) -> None:
    total = sum(r.total for r in all_runs)
    passed = sum(r.passed for r in all_runs)
    leaks = sum(r.leaks for r in all_runs)
    missing = sum(r.missing_token for r in all_runs)
    fp_token = sum(r.fp_token for r in all_runs)
    fp_passport = sum(r.fp_passport for r in all_runs)
    restore_missing = sum(r.restore_missing for r in all_runs)
    latencies = [x for r in all_runs for x in r.latencies]

    mean = statistics.mean(latencies) if latencies else 0.0
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)

    print("\n" + "=" * 74)
    print("GLOBAL SUMMARY (all seeds)")
    print("=" * 74)
    print(f"Runs:        {len(all_runs)}")
    print(f"Total cases: {total}")
    print(f"Passed:      {passed} ({(passed / total * 100.0) if total else 0.0:.2f}%)")
    print(f"Leaks:       {leaks}")
    print(f"MissingTok:  {missing}")
    print(f"FP Token:    {fp_token}")
    print(f"FP Passport: {fp_passport}")
    print(f"RestoreMiss: {restore_missing}")
    print(f"Latency: mean={mean:.4f}s p50={p50:.4f}s p95={p95:.4f}s")
    print("=" * 74)

    # Aggregate per-category across runs
    agg: Dict[str, CategoryStats] = {}
    for r in all_runs:
        for cat, cs in r.by_cat.items():
            if cat not in agg:
                agg[cat] = CategoryStats()
            a = agg[cat]
            a.total += cs.total
            a.passed += cs.passed
            a.leaks += cs.leaks
            a.missing_token += cs.missing_token
            a.fp_token += cs.fp_token
            a.fp_passport += cs.fp_passport
            a.restore_missing += cs.restore_missing

    rows = []
    for cat, cs in agg.items():
        pass_rate = (cs.passed / cs.total * 100.0) if cs.total else 0.0
        rows.append((pass_rate, cat, cs))
    rows.sort(key=lambda x: x[0])

    print("\nPer-category (aggregated):")
    print(f"{'Category':<28} {'Total':>6} {'Pass%':>7} {'Leaks':>6} {'MissTok':>7} {'FPtok':>6} {'FPpass':>6} {'Rmiss':>6}")
    for pass_rate, cat, cs in rows:
        print(
            f"{cat:<28} {cs.total:>6} {pass_rate:>6.2f}% {cs.leaks:>6} {cs.missing_token:>7} "
            f"{cs.fp_token:>6} {cs.fp_passport:>6} {cs.restore_missing:>6}"
        )


# -------------------------
# Session stability
# -------------------------
def session_stability_test(gateway: PIIGateway) -> None:
    print("\nSession stability test...")
    s2 = "session_stability"
    msg1 = "Меня зовут Петр. Мой телефон +79991234567."
    r1 = gateway.tokenize(s2, msg1)
    t1 = r1["tokenized_text"]
    msg2 = "Петр, перезвоните пожалуйста на +79991234567."
    r2 = gateway.tokenize(s2, msg2)
    t2 = r2["tokenized_text"]

    print("Msg1 tokenized:", t1)
    print("Msg2 tokenized:", t2)
    print("Vault fields:", gateway.export_fields(s2))


# -------------------------
# Main benchmark runner
# -------------------------
def run_benchmark(
    *,
    seeds: List[int],
    counts: Dict[str, int],
    verbose_fail: bool = False,
) -> None:
    fake = Faker("ru_RU")

    policy = PIIGatewayPolicy(
        passport_use_context_filter=True,
        include_raw_values_in_result=True,
        # fail_closed оставь True в проде.
    )

    # Init gateway once (так быстрее). Seed влияет на генерацию кейсов, не на gateway.
    try:
        gateway = PIIGateway(
            model_path="Qwen2.5-3B-Instruct-IQ3_M.gguf",
            policy=policy,
        )
    except Exception as e:
        print(f"⚠️ Could not init local LLM for names: {e}")
        policy.enable_llm_names = False
        gateway = PIIGateway(policy=policy)

    all_runs: List[RunStats] = []

    for seed in seeds:
        random.seed(seed)

        cases = build_cases(fake, counts=counts)
        # перемешаем, чтобы не было "партиями" (иногда удобно для выявления зависимостей)
        random.shuffle(cases)

        stats = RunStats(seed=seed)
        session_id = f"bench_session_{seed}"

        for case in cases:
            stats.total += 1
            cs = ensure_cat(stats, case.category)
            cs.total += 1

            ok, inc, latency = eval_case(
                gateway,
                policy,
                session_id,
                case,
                verbose_fail=verbose_fail,
            )
            stats.latencies.append(latency)

            if ok:
                stats.passed += 1
                cs.passed += 1
            else:
                # counts
                stats.leaks += inc["leaks"]
                stats.missing_token += inc["missing_token"]
                stats.fp_token += inc["fp_token"]
                stats.fp_passport += inc["fp_passport"]
                stats.restore_missing += inc["restore_missing"]

                cs.leaks += inc["leaks"]
                cs.missing_token += inc["missing_token"]
                cs.fp_token += inc["fp_token"]
                cs.fp_passport += inc["fp_passport"]
                cs.restore_missing += inc["restore_missing"]

        print_run_summary(stats)
        all_runs.append(stats)

    print_global_summary(all_runs)

    # Один стабильностный тест в конце
    session_stability_test(gateway)
    print("\n✅ Benchmark finished.")


if __name__ == "__main__":
    # Конфиг "по-взрослому":
    # - много дешёвых regex-кейсов
    # - меньше имён (LLM дороже)
    # - mixed — средне, потому что проверяет всё вместе
    COUNTS = {
        "pos_email": 200,
        "pos_phone": 200,
        "pos_card": 200,
        "neg_fake_card_order": 200,
        "pos_snils": 120,
        "pos_passport": 150,
        "neg_passport_like_contract": 150,
        "pos_date": 150,
        "pos_name": 80,
        "hard_neg_numbers": 200,
        "mixed": 120,
    }
    # Это даст примерно: 1770 кейсов на один прогон (в зависимости от mix/настроек).
    # Если хочешь ближе к 1000 — уменьши значения примерно на 40%.

    SEEDS = [1, 2, 3]  # 3 прогона для устойчивости
    run_benchmark(seeds=SEEDS, counts=COUNTS, verbose_fail=False)