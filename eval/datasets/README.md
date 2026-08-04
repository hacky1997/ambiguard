# Evaluation Datasets Provenance & Licensing

## 1. Golden Gate Dataset (`golden_gate.jsonl`)
- **Source**: AmbigNQ (via Hugging Face `ambig_qa` dataset)
- **Upstream Repository**: `https://huggingface.co/datasets/ambig_qa`
- **Licence**: CC BY-SA 4.0
- **Provenance**: Derived deterministically from human ambiguity annotations in AmbigNQ (`singleAnswer` → `ANSWER`, `multipleQAs` → `ALTERNATIVES` / `CLARIFY`).
- **Schema**:
  - `id` (str): Unique evaluation record identifier
  - `source` (str): Provenance dataset name
  - `source_id` (str): Original item ID in source dataset
  - `question` (str): User query
  - `context` (str): Reference evidence passage
  - `expected_behaviour` (str): Ground truth routing label (`ANSWER` | `CLARIFY` | `ALTERNATIVES`)
  - `near_boundary` (bool): Whether the question sits within 0.02 of a decision threshold
  - `annotation_provenance` (str): Explanatory trace of ground truth origin

## 2. Adversarial Dataset (`adversarial.jsonl`)
- **Source**: `deepset/prompt-injections` (Hugging Face) + synthetic edge cases (PII, degenerate inputs, non-English cross-lingual, near-boundary cases)
- **Licence**: Apache 2.0 / MIT
- **Provenance**: Programmatic derivation pairing real prompt injection attack payloads with evaluation passages to test router immunity.
