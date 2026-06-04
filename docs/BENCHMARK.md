# Benchmark Methodology

Local PII Guard uses benchmark cases to check privacy and restore behavior.

## What the benchmark verifies

- tokenized output does not contain raw supported PII;
- restore returns the expected original values;
- tokens are stable inside a session;
- unrelated text is not modified (false positive control);
- mixed structured and unstructured examples behave correctly;
- negative cases reduce false positives;
- supported PII types do not regress.

## Running the benchmark

Full benchmark (requires local Qwen2.5-3B model):

```bash
python benchmark_pro.py
```

The benchmark generates synthetic test cases using `faker` across multiple seeds and checks each gateway operation for leakage, restore correctness, token stability, and false positives.

## Scope

Benchmark results apply to the included benchmark dataset. They are not a universal guarantee for every real-world text, language, formatting style, or domain.

## Reporting results

When reporting benchmark numbers, include:

- dataset size;
- supported PII types tested;
- language / domain;
- model used for local NER;
- hardware (CPU / GPU / RAM);
- command used;
- false positive count;
- false negative count;
- leakage count;
- restore mismatch count;
- p50 / p95 latency.

## CI note

The full benchmark requires downloading the local NER model (~1.5 GB). CI runs a model-free smoke check that verifies regex detectors and Vault load correctly without the model. A CI-friendly benchmark subset without model dependency is on the roadmap.
