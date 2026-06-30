import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "shared54_pass1_instances.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "csv"

MODELS = [
    "qwen2_5_1_5b",
    "llama_3_1_8b",
    "mistral_small_3_24b_awq",
    "qwen2.5-32b-awq",
]
EMOTIONS = ["fear", "anger", "sadness", "disgust", "neutral", "happiness"]
TOKEN_FIELDS = ["calls", "prompt_tokens", "completion_tokens", "total_tokens"]


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_int(value):
    return int(float(value)) if str(value).strip() else 0


def require_columns(rows, columns):
    missing = [field for field in columns if field not in rows[0]]
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")


def load_instances(path):
    rows = read_rows(path)
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    required = [
        "model", "assignment", "question_id", "emotion", "personality",
        *TOKEN_FIELDS,
    ]
    require_columns(rows, required)

    items = []
    for row in rows:
        item = {
            "model": row["model"],
            "assignment": row["assignment"],
            "question_id": row["question_id"],
            "emotion": row["emotion"],
            "personality": row["personality"],
        }
        for field in TOKEN_FIELDS:
            item[field] = as_int(row[field])
        items.append(item)
    return items


def summarize(items, prefix=""):
    instances = len(items)
    calls = sum(row[f"{prefix}calls"] for row in items)
    prompt = sum(row[f"{prefix}prompt_tokens"] for row in items)
    completion = sum(row[f"{prefix}completion_tokens"] for row in items)
    total = sum(row[f"{prefix}total_tokens"] for row in items)
    return {
        "instances": instances,
        "calls": calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "calls_per_instance": calls / instances if instances else 0.0,
        "prompt_tokens_per_instance": prompt / instances if instances else 0.0,
        "completion_tokens_per_instance": completion / instances if instances else 0.0,
        "total_tokens_per_instance": total / instances if instances else 0.0,
        "prompt_tokens_per_call": prompt / calls if calls else 0.0,
        "completion_tokens_per_call": completion / calls if calls else 0.0,
        "total_tokens_per_call": total / calls if calls else 0.0,
    }


def add_overall_groups(rows, key_fields):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], *[row[field] for field in key_fields])].append(row)
        grouped[("overall", *[row[field] for field in key_fields])].append(row)
    return grouped


def overall_rows(rows):
    out = []
    for model in MODELS:
        items = [row for row in rows if row["model"] == model]
        if items:
            out.append({
                "model": model,
                "assignment": "shared54",
                **summarize(items),
            })
    out.append({
        "model": "overall",
        "assignment": "shared54",
        **summarize(rows),
    })
    return out


def emotion_rows(rows):
    grouped = add_overall_groups(rows, ["emotion"])
    out = []
    for model in MODELS + ["overall"]:
        for emotion in EMOTIONS:
            items = grouped.get((model, emotion), [])
            if items:
                out.append({
                    "model": model,
                    "assignment": "shared54",
                    "emotion": emotion,
                    **summarize(items),
                })
    return out


def personality_rows(rows):
    group_defs = [
        ("C-High", lambda row: row["personality"][0] == "H"),
        ("C-Low", lambda row: row["personality"][0] == "L"),
        ("O-High", lambda row: row["personality"][1] == "H"),
        ("O-Low", lambda row: row["personality"][1] == "L"),
        ("E-High", lambda row: row["personality"][2] == "H"),
        ("E-Low", lambda row: row["personality"][2] == "L"),
        ("Neutral", lambda row: row["personality"] == "NNN"),
    ]

    out = []
    for model in MODELS + ["overall"]:
        model_rows = rows if model == "overall" else [row for row in rows if row["model"] == model]
        for group_name, predicate in group_defs:
            items = [row for row in model_rows if predicate(row)]
            if items:
                out.append({
                    "model": model,
                    "assignment": "shared54",
                    "personality_group": group_name,
                    **summarize(items),
                })
    return out


def metric_fields(prefix_fields):
    return [
        *prefix_fields,
        "instances", "calls", "prompt_tokens", "completion_tokens", "total_tokens",
        "calls_per_instance", "prompt_tokens_per_instance",
        "completion_tokens_per_instance", "total_tokens_per_instance",
        "prompt_tokens_per_call", "completion_tokens_per_call", "total_tokens_per_call",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    rows = load_instances(args.data)
    out_dir = Path(args.output_dir)

    write_rows(out_dir / "rq3_token_usage_overall.csv", overall_rows(rows), metric_fields(["model", "assignment"]))
    write_rows(out_dir / "rq3_token_usage_by_emotion.csv", emotion_rows(rows), metric_fields(["model", "assignment", "emotion"]))
    write_rows(
        out_dir / "rq3_token_usage_by_personality.csv",
        personality_rows(rows),
        metric_fields(["model", "assignment", "personality_group"]),
    )
    print(f"instances={len(rows)}")
    print(f"calls={sum(row['calls'] for row in rows)}")
    print(f"total_tokens={sum(row['total_tokens'] for row in rows)}")


if __name__ == "__main__":
    main()
