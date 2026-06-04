# Local PII Guard Roadmap

This roadmap tracks work that makes Local PII Guard more useful as a local-first privacy gateway for LLM apps and AI agents.

## Near term

- Add CI with Python 3.9+.
- Add a CI-friendly benchmark subset (no model download required).
- Add more false positive and false negative regression cases.
- Improve README examples and integration docs.
- Add issue templates and PR template.
- Clarify benchmark methodology.

## Detector coverage

- Add Russian address detection.
- Add INN/KPP/OGRN detectors.
- Improve multilingual name detection.
- Add organization/legal-entity detection.
- Add domain-specific negative examples.

## Vault and storage

- Add encrypted local Vault backend.
- Add SQLite Vault backend.
- Add Postgres Vault backend.
- Document retention and deletion patterns.
- Add session cleanup examples.

## Security hardening

- Add tests for raw PII leakage in tokenized output.
- Add token collision regression tests.
- Add restore mismatch tests.
- Add logging safety tests (no raw PII in logs).
- Add prompt-injection leakage scenarios.

## Integrations

- Add minimal FastAPI example.
- Add Telegram bot example.
- Add CRM export example.
- Add batch processing example.
- Add streaming tokenization mode.

## Maintenance

- Add release checklist.
- Add versioned changelog.
