import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "results" / "csv"

MODELS = [
    "qwen2_5_1_5b",
    "llama_3_1_8b",
    "mistral_small_3_24b_awq",
    "qwen2.5-32b-awq",
]
EMOTIONS = ["anger", "fear", "disgust", "sadness", "happiness", "neutral"]
PERSONALITIES = ["HLL", "HLH", "HHL", "HHH", "LLL", "LLH", "LHL", "LHH", "NNN"]


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value):
    return float(value) if str(value).strip() else 0.0


def profile_name(row):
    return f"{row['emotion']}-{row['personality']}"


def profile_sort_key(row):
    emotion_index = EMOTIONS.index(row["emotion"]) if row["emotion"] in EMOTIONS else len(EMOTIONS)
    personality_index = PERSONALITIES.index(row["personality"]) if row["personality"] in PERSONALITIES else len(PERSONALITIES)
    return emotion_index, personality_index, row["emotion"], row["personality"]


def joined_profiles(rows):
    return "; ".join(profile_name(row) for row in sorted(rows, key=profile_sort_key))


def load_shared_instances(path):
    grouped = defaultdict(list)
    for row in read_rows(path):
        grouped[(row["model"], row["assignment"], row["emotion"], row["personality"])].append(as_float(row["bleu"]))

    rows = []
    for (model, assignment, emotion, personality), values in grouped.items():
        rows.append({
            "model": model,
            "assignment": assignment,
            "emotion": emotion,
            "personality": personality,
            "bleu": sum(values) / len(values) if values else 0.0,
        })
    return rows


def load_baselines(path):
    return {row["model"]: {"bleu": as_float(row["bleu"])} for row in read_rows(path)}


def range_rows(rows, baselines):
    out = []
    for model in MODELS:
        items = [row for row in rows if row["model"] == model]
        if not items:
            continue
        worst_bleu = min(row["bleu"] for row in items)
        best_bleu = max(row["bleu"] for row in items)
        worst_ties = [row for row in items if row["bleu"] == worst_bleu]
        best_ties = [row for row in items if row["bleu"] == best_bleu]
        out.append({
            "model": model,
            "assignment": "shared54",
            "configurations": len(items),
            "worst_profile": joined_profiles(worst_ties),
            "worst_bleu": worst_bleu,
            "best_profile": joined_profiles(best_ties),
            "best_bleu": best_bleu,
            "range_bleu": best_bleu - worst_bleu,
            "range_relative_to_worst": (best_bleu - worst_bleu) / worst_bleu if worst_bleu else 0.0,
            "baseline_bleu": baselines[model]["bleu"],
        })
    return out


def baseline_detail_rows(rows, baselines):
    out = []
    for row in rows:
        baseline = baselines[row["model"]]
        delta = row["bleu"] - baseline["bleu"]
        out.append({
            "model": row["model"],
            "assignment": "shared54",
            "emotion": row["emotion"],
            "personality": row["personality"],
            "bleu": row["bleu"],
            "baseline_bleu": baseline["bleu"],
            "delta_bleu": delta,
            "beat_self_report": int(delta > 0),
        })
    return sorted(out, key=lambda row: (row["model"], -row["bleu"], row["emotion"], row["personality"]))


def baseline_summary_rows(details, baselines):
    out = []
    for model in MODELS:
        items = [row for row in details if row["model"] == model]
        if not items:
            continue
        baseline = baselines[model]
        best = max(items, key=lambda row: (row["bleu"], row["emotion"], row["personality"]))
        beats = sum(row["beat_self_report"] for row in items)
        max_gain = best["bleu"] - baseline["bleu"]
        out.append({
            "model": model,
            "assignment": "shared54",
            "baseline_bleu": baseline["bleu"],
            "configurations": len(items),
            "beat_self_report": beats,
            "beat_rate": beats / len(items) if items else 0.0,
            "max_gain_bleu": max_gain,
            "max_gain_relative": max_gain / baseline["bleu"] if baseline["bleu"] else 0.0,
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    shared = load_shared_instances(data_dir / "shared54_bleu_instances.csv")
    baselines = load_baselines(data_dir / "self_report_bleu.csv")
    details = baseline_detail_rows(shared, baselines)

    write_rows(output_dir / "rq1_bleu_shared54_range_by_model.csv", range_rows(shared, baselines), [
        "model", "assignment", "configurations", "worst_profile", "worst_bleu",
        "best_profile", "best_bleu", "range_bleu", "range_relative_to_worst", "baseline_bleu",
    ])
    write_rows(output_dir / "rq1_bleu_shared54_vs_self_report_by_configuration.csv", details, [
        "model", "assignment", "emotion", "personality", "bleu",
        "baseline_bleu", "delta_bleu", "beat_self_report",
    ])
    write_rows(output_dir / "rq1_bleu_shared54_vs_self_report_by_model.csv", baseline_summary_rows(details, baselines), [
        "model", "assignment", "baseline_bleu", "configurations", "beat_self_report",
        "beat_rate", "max_gain_bleu", "max_gain_relative",
    ])
    print(f"shared54_rows={len(shared)}")


if __name__ == "__main__":
    main()
