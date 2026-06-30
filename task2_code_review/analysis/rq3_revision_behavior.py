import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "shared54_bleu_instances.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "csv"

MODELS = [
    "qwen2_5_1_5b",
    "llama_3_1_8b",
    "mistral_small_3_24b_awq",
    "qwen2.5-32b-awq",
]
EMOTIONS = ["fear", "anger", "sadness", "disgust", "neutral", "happiness"]


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


def load_instances(path):
    rows = []
    for row in read_rows(path):
        rows.append({
            "model": row["model"],
            "assignment": row["assignment"],
            "question_id": row["question_id"],
            "emotion": row["emotion"],
            "personality": row["personality"],
            "revised": as_int(row["revised"]),
        })
    return rows


def summarize(items):
    instances = len(items)
    revised = sum(row["revised"] for row in items)
    return {
        "instances": instances,
        "revised": revised,
        "revision_rate": revised / instances if instances else 0.0,
    }


def overall_rows(rows):
    out = []
    for model in MODELS:
        items = [row for row in rows if row["model"] == model]
        if not items:
            continue
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
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["emotion"])].append(row)
        grouped[("overall", row["emotion"])].append(row)

    out = []
    for model in MODELS + ["overall"]:
        for emotion in EMOTIONS:
            items = grouped.get((model, emotion), [])
            if not items:
                continue
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
            if not items:
                continue
            out.append({
                "model": model,
                "assignment": "shared54",
                "personality_group": group_name,
                **summarize(items),
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    rows = load_instances(args.data)
    out_dir = Path(args.output_dir)

    fields_base = [
        "model", "assignment", "instances", "revised", "revision_rate",
    ]
    write_rows(out_dir / "rq3_revision_behavior_overall.csv", overall_rows(rows), fields_base)
    write_rows(out_dir / "rq3_revision_behavior_by_emotion.csv", emotion_rows(rows), [
        "model", "assignment", "emotion", "instances", "revised", "revision_rate",
    ])
    write_rows(out_dir / "rq3_revision_behavior_by_personality.csv", personality_rows(rows), [
        "model", "assignment", "personality_group",
        "instances", "revised", "revision_rate",
    ])

    print(f"instances={len(rows)}")
    print(f"revised={sum(row['revised'] for row in rows)}")


if __name__ == "__main__":
    main()
