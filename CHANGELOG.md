# Changelog

All notable changes to Local PII Guard are documented here.

## [0.1.0] — initial OSS baseline

### Added

- Local-first PII tokenization gateway (`pii_agent_pro.py`).
- Structured PII detectors: email, phone (RU), bank card (Luhn), passport (RU), SNILS, date (`pii_detectors.py`).
- Local LLM-assisted name detection via Qwen 2.5 3B + heuristic fallback (`pii_llm_names.py`).
- Stable reversible tokens (`<<TYPE_N>>` format).
- In-memory local Vault mapping with session/user scoping (`pii_vault.py`).
- Gateway policy model with `fail_closed` mode (`pii_models.py`).
- Restore flow for cloud model responses.
- Golden benchmark script for leakage, false positives, restore, mixed, and negative cases (`benchmark_pro.py`).
- Model download script for Qwen2.5-3B-Instruct-IQ3_M.gguf (`download_model.py`).
- Apache-2.0 license.
