"""Small file and dataset helpers."""
import json
from datetime import datetime
from pathlib import Path


def timestamp_slug():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_dir_name(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "value"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(text, encoding="utf-8")


def configure_runtime(args, model):
    from src import config, prompts

    config.MODEL = model
    config.BASE_URL = args.llm_url
    config.API_KEY = args.api_key
    config.TEMPERATURE = args.temperature
    config.MAX_REVISE_ROUNDS = args.max_revise_rounds
    config.MAX_TOKENS_REVIEWER = args.max_tokens_reviewer
    config.MAX_TOKENS_JUDGE = args.max_tokens_judge

    prompts.PROMPT_ROOT = Path(args.prompt_dir)
    prompts.TASK_PROMPT_DIR = prompts.PROMPT_ROOT / "task_prompt"
    prompts.TASK_PROMPTS = prompts.load_task_prompts(prompts.PROMPT_ROOT)


def build_input(sample: dict) -> str:
    pre = sample.get("patch_with_additional_information")
    if pre:
        return pre
    raise KeyError("sample is missing patch_with_additional_information")


def read_jsonl(path: str) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def summarize_reviews(records):
    ok = [record for record in records if "_error" not in record]
    errors = [record for record in records if "_error" in record]
    rounds = [record.get("_rounds_executed", 0) for record in ok]
    elapsed = [record.get("_elapsed_s", 0) for record in records]
    tokens = [record.get("token_usage", {}) for record in ok if record.get("token_usage", {}).get("total_tokens", 0) > 0]
    avg = lambda key: (sum(item.get(key, 0) for item in tokens) / len(tokens)) if tokens else 0.0
    return {
        "total_samples": len(records),
        "generated_samples": len(ok),
        "error_samples": len(errors),
        "average_rounds_executed": (sum(rounds) / len(rounds)) if rounds else 0.0,
        "average_elapsed_s": (sum(elapsed) / len(elapsed)) if elapsed else 0.0,
        "average_prompt_tokens": avg("prompt_tokens"),
        "average_completion_tokens": avg("completion_tokens"),
        "average_total_tokens": avg("total_tokens"),
    }


def sample_key(sample, idx):
    for key in ("id", "comment_id", "pr_number"):
        value = sample.get(key)
        if value is not None:
            return str(value)
    return str(idx)
