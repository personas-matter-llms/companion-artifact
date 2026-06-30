# Dataset and Result Tables

Provide the sampled Hydra-Reviewer JSONL file with `--data-path` when running experiments.

The paper uses 377 review instances sampled from the Hydra-Reviewer dataset with seed `42`.

Result CSVs (inputs to the analysis scripts):

- `shared54_bleu_instances.csv` — per-review-instance BLEU, revision indicators, and token usage for each shared-profile configuration (main analysis input).
- `shared54_bleu.csv` — BLEU summary for each shared-profile configuration.
- `mixed24_bleu.csv` — BLEU for the 24 mixed-profile assignments.
- `self_report_bleu.csv` — self-report baseline per model.
- `sentiment_top_bottom_messages.csv` — per-message SentiCR negativity (0/1) for one fixed best and worst shared-profile configuration per model. If performance ties occur, this file includes one representative tied configuration.
