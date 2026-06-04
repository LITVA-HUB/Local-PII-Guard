# Security Policy

Local PII Guard is a privacy gateway for LLM apps and AI agents. Security-sensitive areas include PII detection, reversible tokenization, local Vault storage, restore logic, logging, prompt handling, and downstream integrations.

## Supported scope

Security reports are welcome for:

- raw PII leakage in tokenized output;
- token collisions;
- restore mismatches that expose or corrupt sensitive values;
- accidental logging of raw PII;
- Vault leakage or unsafe storage defaults;
- prompt-injection paths that reveal or manipulate protected values;
- detector bypasses for supported PII types;
- unsafe integration examples;
- dependency or model-loading issues that create security risk.

## Reporting

Please do not open a public GitHub issue for security vulnerabilities.

Use GitHub private vulnerability reporting if available. If it is not available, open a minimal public issue that says only:

> Security report available. Please enable private vulnerability reporting or provide a private contact.

Do not include raw personal data, exploit payloads, secrets, private logs, or real user records in public issues.

## Maintainer response

The maintainer will triage reports, confirm impact where possible, prepare a fix, and publish notes once the issue is resolved.

## Privacy note

Please use synthetic data in reports whenever possible. Do not submit real passports, bank cards, phone numbers, names, CRM exports, or customer records.
