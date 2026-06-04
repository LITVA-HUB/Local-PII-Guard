# Contributing to Local PII Guard

Thanks for helping improve Local PII Guard.

## Good first contributions

- Add detector test cases.
- Add benchmark cases for false positives and false negatives.
- Improve documentation and examples.
- Add Vault backends.
- Improve local NER validation.
- Add CI-friendly quick benchmark mode.
- Add integration examples for agents, CRM, Telegram, or databases.

## Development setup

```bash
git clone https://github.com/LITVA-HUB/Local-PII-Guard.git
cd Local-PII-Guard

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Running checks

```bash
python -m compileall .
python benchmark_pro.py
```

If the full benchmark is slow or requires a local model, prefer adding a quick benchmark mode that can run in CI without downloading large files.

## Detector contributions

When adding or changing a detector, include:

- positive examples (synthetic data only);
- negative examples;
- mixed examples;
- restore tests;
- token stability tests;
- notes about likely false positives and false negatives.

Do not include real personal data in examples or tests.

## Pull requests

Please keep PRs focused and describe:

- what changed;
- why it matters;
- how it was tested;
- privacy / security impact;
- benchmark impact (false positives, false negatives, leakage count).

Security-sensitive changes include detector logic, Vault storage, restore behavior, logging, model prompts, and integrations.
