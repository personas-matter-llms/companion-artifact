# Dataset and Result Tables

Provide the sampled LiveCodeBench JSONL file with `--data-path` when running experiments.

The paper uses 282 tasks sampled from LiveCodeBench-lite with seed `0`.

Result CSVs (inputs to the analysis scripts):

- `shared54_pass1_instances.csv` — per-task pass@1 outcome, revision, over-revision, and token usage for each shared-profile configuration (main analysis input).
- `shared54_pass1.csv` — pass@1 summary for each shared-profile configuration.
- `mixed24_pass1.csv` — pass@1 for the 24 mixed-profile assignments.
- `self_report_pass1.csv` — self-report baseline per model.
- `sentiment_top_bottom_messages.csv` — per-message SentiCR negativity (0/1) for one fixed best and worst shared-profile configuration per model. If performance ties occur, this file includes one representative tied configuration.
