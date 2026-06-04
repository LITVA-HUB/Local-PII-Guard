# Architecture

Local PII Guard is a local-first privacy gateway. All sensitive data stays
on the local machine. Cloud LLMs only ever see tokenized text.

## Data flow

```
User text
    │
    ▼
┌───────────────────────────────────────────┐
│              PIIGateway.tokenize()        │
│                                           │
│  1. Regex detectors (email, phone, card,  │
│     SNILS, passport, date)                │
│  2. Local LLM NER (names via Qwen2.5 3B) │
│  3. Heuristic name fallback               │
│  4. Overlap resolution                    │
│  5. Token assignment + Vault storage      │
└───────────────────────────────────────────┘
    │
    ▼
Tokenized text (<<NAME_1>>, <<PHONE_1>>, ...)
    │
    ▼  [sent to cloud LLM — no raw PII]
Cloud model response (with tokens)
    │
    ▼
┌───────────────────────────────────────────┐
│              PIIGateway.restore()         │
│                                           │
│  TOKEN_PATTERN.sub() → Vault lookup       │
└───────────────────────────────────────────┘
    │
    ▼
Restored response (original values)
```

## Components

### `pii_agent_pro.py` — PIIGateway

The main entry point. Orchestrates detection, tokenization, and restore.
Holds references to regex detectors, the name extractor, and the Vault.

Key methods:
- `tokenize(session_id, text)` — detect PII and return tokenized text + metadata
- `restore(session_id, text)` — replace tokens with original values from Vault
- `export_fields(session_id)` — return Vault contents grouped by PII type
- `tokenize_payload(session_id, payload)` — recursive tokenization for dicts/lists
- `restore_payload(session_id, payload)` — recursive restore for dicts/lists

### `pii_detectors.py` — RegexDetector

Stateless regex-based detectors. Each detector has:
- a compiled `Pattern`
- an optional `validator` (e.g. Luhn check for cards)
- an optional `context_filter` (e.g. passport context words)

Detectors never modify state. They return `List[Entity]` for each input text.

### `pii_llm_names.py` — LocalNameExtractor

Wraps a local Qwen2.5 3B GGUF model (via `llama_cpp`) for name detection.

Before calling the LLM, already-detected PII spans are masked with a
placeholder character (`█` by default) to reduce hallucinations.
Long texts are split into overlapping chunks.

Falls back to a conservative heuristic (`heuristic_name_spans`) when
the LLM is unavailable or returns unparseable output.

### `pii_vault.py` — InMemoryPIIVault

Thread-safe in-memory mapping of `token → original_value` per session.

Tokens are stable within a session: the same input value always maps
to the same token. Normalization (`normalize_for_key`) handles formatting
variants (e.g. `+79991234567` and `89991234567` map to the same token).

### `pii_models.py` — PIIGatewayPolicy, Entity

Shared data classes and constants:
- `PIIGatewayPolicy` — controls detection, tokenization, and failure behavior
- `Entity` — a detected PII span with type, position, value, and confidence
- `resolve_overlaps()` — greedy span deduplication by priority + length

## Token format

```
<<TYPE_N>>
```

- `TYPE` is the entity type in uppercase: `NAME`, `PHONE`, `EMAIL`, etc.
- `N` is a 1-based counter scoped to the session and type.
- Example: `<<NAME_1>>`, `<<PHONE_1>>`, `<<CARD_2>>`

Tokens are designed to be stable across re-tokenization of the same value
within a session. The same phone number always gets the same token.

## Security boundaries

```
┌──────────────────────────────────────────────────────┐
│  TRUSTED LOCAL ENVIRONMENT                           │
│                                                      │
│  PIIGateway  →  InMemoryPIIVault  (raw PII)         │
│      │                                               │
│      └── pii_llm_names (local Qwen2.5 model)        │
│                                                      │
└──────────────────────────────────────────────────────┘
         │
         │  [only <<TOKEN>> strings cross this boundary]
         ▼
┌──────────────────────────────────────────────────────┐
│  UNTRUSTED: cloud LLM provider, logs, traces,        │
│             downstream agent tools                   │
└──────────────────────────────────────────────────────┘
```

The Vault never leaves the local environment. Logs, traces, and API calls
should only ever contain tokenized text. See [SECURITY.md](../SECURITY.md)
for the full threat model.
