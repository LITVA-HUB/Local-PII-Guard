# Local PII Guard

[![CI](https://github.com/LITVA-HUB/Local-PII-Guard/actions/workflows/ci.yml/badge.svg)](https://github.com/LITVA-HUB/Local-PII-Guard/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](./LICENSE)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)
![Privacy](https://img.shields.io/badge/privacy-local--first-brightgreen)
![LLM](https://img.shields.io/badge/LLM-local%20NER-informational)

**Local PII Guard is a local-first privacy gateway for LLM apps and AI agents.**

It detects personal data, replaces it with stable reversible tokens, keeps original values in a local Vault, sends only sanitized text to cloud LLMs, and restores the final response locally.

- 🇷🇺 Russian README: [README.ru.md](./README.ru.md)
- Benchmark methodology: [docs/BENCHMARK.md](./docs/BENCHMARK.md)
- Roadmap: [ROADMAP.md](./ROADMAP.md)
- Changelog: [CHANGELOG.md](./CHANGELOG.md)

---

## Why this matters for OSS maintainers

LLM apps and AI agents often send user messages, CRM data, logs, payloads, and support conversations to cloud models. Local PII Guard gives developers a local privacy layer that can tokenize personal data before it leaves the trusted environment and restore it after the model response returns.

This helps maintainers build privacy-preserving AI workflows without forcing every project to implement its own PII detection, reversible tokenization, Vault mapping, restore logic, and leakage checks.

---

## What it does

- detects structured PII such as email, phone numbers, bank cards, Russian passport-like numbers, SNILS-like identifiers, and dates;
- detects names with local LLM-assisted NER plus validation and fallback logic;
- replaces sensitive values with stable reversible tokens such as `<<NAME_1>>` and `<<PHONE_1>>`;
- stores originals in a local Vault scoped to a session or user;
- sends only tokenized text to cloud LLMs;
- restores cloud responses back to the original values locally;
- supports benchmark-driven validation for leakage, false positives, restore correctness, mixed cases, negative cases, and token stability.

---

## Security model

Local PII Guard assumes that cloud LLM providers, logs, traces, and downstream agent tools should not receive raw personal data. The gateway replaces sensitive values with stable reversible tokens before the request leaves the local environment. Original values stay in the local Vault and are restored only after the model response returns.

Security-sensitive areas:

- detector false negatives;
- detector false positives that damage useful content;
- token collisions;
- local Vault leakage;
- accidental logging of raw PII;
- restore mismatches;
- prompt injection that tries to reveal or manipulate tokens;
- unsafe integrations with CRM, Telegram, databases, or agent tools.

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Download the local NER model (~1.5 GB, required for name detection):

```bash
python download_model.py
```

### Usage example

```python
from pii_agent_pro import PIIGateway
from pii_models import PIIGatewayPolicy

policy = PIIGatewayPolicy(fail_closed=True)

gateway = PIIGateway(
    model_path="Qwen2.5-3B-Instruct-IQ3_M.gguf",
    policy=policy,
)

session_id = "chat_001"
text = "My name is Peter Ivanov. Phone +79991234567."

result = gateway.tokenize(session_id, text)
print(result["tokenized_text"])
# My name is <<NAME_1>>. Phone <<PHONE_1>>.

cloud_reply = "Hello, <<NAME_1>>!"
print(gateway.restore(session_id, cloud_reply))
# Hello, Peter Ivanov!
```

### CRM / database export

```python
gateway.export_fields(session_id)
# {"NAME": ["Peter Ivanov"], "PHONE": ["+79991234567"]}
```

---

## Supported PII types

### Structured (regex + validation)

| Type | Token | Notes |
|------|-------|-------|
| Email | `<<EMAIL_N>>` | |
| Phone (RU) | `<<PHONE_N>>` | +7/8 prefix, false-positive filtering |
| Bank card | `<<CARD_N>>` | 13–19 digits, Luhn check |
| Passport (RU) | `<<PASSPORT_N>>` | 4+6 format, context filter |
| SNILS | `<<SNILS_N>>` | |
| Date | `<<DATE_N>>` | dd.mm.yyyy |

### Unstructured (local LLM NER)

| Type | Token | Notes |
|------|-------|-------|
| Full name / initials | `<<NAME_N>>` | Qwen 2.5 3B via llama.cpp + heuristic fallback |

---

## Benchmark and quality

The repository includes a golden benchmark for synthetic and mixed test cases. Current README-reported results should be interpreted as results on the included benchmark set, not as a universal guarantee for all real-world text.

Quality checks cover:

- no raw PII leakage in tokenized output;
- correct restore behavior;
- no token collisions inside a session;
- false positive control;
- false negative regression tests;
- mixed structured/unstructured examples;
- negative cases where non-PII should remain unchanged.

Run the full benchmark (requires local model):

```bash
python benchmark_pro.py
```

See [docs/BENCHMARK.md](./docs/BENCHMARK.md) for methodology and result interpretation.

---

## Project structure

```
pii_agent_pro.py      # Main gateway
pii_detectors.py      # Regex detectors + validators
pii_llm_names.py      # Local LLM NER for names
pii_vault.py          # Token ↔ value Vault
pii_models.py         # Policies and constants
benchmark_pro.py      # Golden benchmark
download_model.py     # Model download script
```

---

## Known limits

- address detection is not implemented yet;
- INN/KPP/OGRN detection is not implemented yet;
- multilingual NER is not complete;
- Vault encryption is planned but not implemented yet;
- real-world accuracy depends on language, domain, formatting, and benchmark coverage;
- local LLM-based NER requires a model download (~1.5 GB) and enough local compute.

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for the full roadmap.

---

## Contributing

Contributions are welcome. Good first areas include detectors, benchmark cases, docs, Vault backends, CI, and integration examples. See [CONTRIBUTING.md](./CONTRIBUTING.md).

---

## Security

Please do not report security vulnerabilities in public issues. See [SECURITY.md](./SECURITY.md).

---

## License

Apache License 2.0. See [LICENSE](./LICENSE).
