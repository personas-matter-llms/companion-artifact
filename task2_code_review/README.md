# Task 2: Code Review

## Setup

Use `--data-path` to point the runner to the dataset (e.g. sampled Hydra-Reviewer JSONL file).
Hydra-Reviewer BLEU evaluation is not bundled in this artifact. Please use the official Hydra-Reviewer evaluator if needed.

## Run experiments

Shared-profile:

```bash
python scripts/run_main.py --prompt-dir <prompt_dir> \
  --data-path <dataset_jsonl> \
  --models <model_name> --llm-url <openai_compatible_url> --api-key <api_key>
```

Mixed-profile:

```bash
python scripts/run_mixed_profile.py --prompt-dir <prompt_dir> --selected-persona <selected_persona_set> \
  --data-path <dataset_jsonl> \
  --models <model_name> --llm-url <openai_compatible_url> --api-key <api_key>
```

Self-report baseline:

```bash
python scripts/self_report.py <openai_compatible_url> <model_name>
```

## Generate paper result tables

Analysis reads `data/*.csv` and writes `results/csv/`. The R script additionally requires the `lme4` package.

```bash
python analysis/rq1_bleu.py
python analysis/rq1_spearman.py
python analysis/rq2_mixed_profile.py
python analysis/rq3_revision_behavior.py
python analysis/rq3_token_usage.py
python analysis/rq3_sentiment.py
Rscript analysis/rq3_glmm_revision_behavior.R
```
