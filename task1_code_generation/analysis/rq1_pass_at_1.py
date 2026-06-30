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


def as_int(value):
    return int(float(value)) if str(value).strip() else 0


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
        key = (row["model"], row["assignment"], row["emotion"], row["personality"])
        grouped[key].append(as_int(row["passed"]))

    rows = []
    for (model, assignment, emotion, personality), values in grouped.items():
        evaluated = len(values)
        passed = sum(values)
        rows.append({
            "model": model,
            "assignment": assignment,
            "emotion": emotion,
            "personality": personality,
            "passed": passed,
            "evaluated": evaluated,
            "total": evaluated,
            "pass_at_1": passed / evaluated if evaluated else 0.0,
        })
    return rows


def load_baselines(path):
    return {
        row["model"]: {
            "passed": as_int(row["passed"]),
            "evaluated": as_int(row["evaluated"]),
            "pass_at_1": as_float(row["pass_at_1"]),
        }
        for row in read_rows(path)
    }


def range_rows(rows, baselines):
    out = []
    for model in MODELS:
        items = [row for row in rows if row["model"] == model]
        if not items:
            continue
        worst_passed = min(row["passed"] for row in items)
        best_passed = max(row["passed"] for row in items)
        worst_ties = [row for row in items if row["passed"] == worst_passed]
        best_ties = [row for row in items if row["passed"] == best_passed]
        worst = sorted(worst_ties, key=profile_sort_key)[0]
        best = sorted(best_ties, key=profile_sort_key)[0]
        baseline = baselines[model]
        out.append({
            "model": model,
            "assignment": "shared54",
            "configurations": len(items),
            "worst_profile": joined_profiles(worst_ties),
            "worst_passed": worst["passed"],
            "worst_pass_at_1": worst["pass_at_1"],
            "best_profile": joined_profiles(best_ties),
            "best_passed": best["passed"],
            "best_pass_at_1": best["pass_at_1"],
            "range_passed": best["passed"] - worst["passed"],
            "range_pass_at_1": best["pass_at_1"] - worst["pass_at_1"],
            "baseline_passed": baseline["passed"],
            "baseline_pass_at_1": baseline["pass_at_1"],
        })
    return out


def baseline_summary_rows(rows, baselines):
    out = []
    for model in MODELS:
        items = [row for row in rows if row["model"] == model]
        if not items:
            continue
        baseline = baselines[model]
        best = max(items, key=lambda row: (row["passed"], row["emotion"], row["personality"]))
        beats = sum(1 for row in items if row["passed"] > baseline["passed"])
        max_gain_passed = best["passed"] - baseline["passed"]
        out.append({
            "model": model,
            "assignment": "shared54",
            "baseline_passed": baseline["passed"],
            "baseline_evaluated": baseline["evaluated"],
            "baseline_pass_at_1": baseline["pass_at_1"],
            "configurations": len(items),
            "beat_self_report": beats,
            "beat_rate": beats / len(items) if items else 0.0,
            "max_gain_passed": max_gain_passed,
            "max_gain_relative": max_gain_passed / baseline["passed"] if baseline["passed"] else 0.0,
            "max_gain_pass_at_1": best["pass_at_1"] - baseline["pass_at_1"],
        })
    return out


def summarize_group(items):
    passed = sum(row["passed"] for row in items)
    evaluated = sum(row["evaluated"] for row in items)
    return {
        "configurations": len(items),
        "passed_sum": passed,
        "evaluated_sum": evaluated,
        "overall_pass_at_1": passed / evaluated if evaluated else 0.0,
        "mean_passed": sum(row["passed"] for row in items) / len(items),
        "mean_pass_at_1": sum(row["pass_at_1"] for row in items) / len(items),
    }


def emotion_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model"], row["emotion"]), []).append(row)
    out = []
    for (model, emotion), items in sorted(grouped.items()):
        out.append({
            "model": model,
            "assignment": "shared54",
            "emotion": emotion,
            **summarize_group(items),
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
    for model in sorted({row["model"] for row in rows}):
        model_rows = [row for row in rows if row["model"] == model]
        for group_name, predicate in group_defs:
            items = [row for row in model_rows if predicate(row)]
            if not items:
                continue
            out.append({
                "model": model,
                "assignment": "shared54",
                "personality_group": group_name,
                **summarize_group(items),
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    shared = load_shared_instances(data_dir / "shared54_pass1_instances.csv")
    baselines = load_baselines(data_dir / "self_report_pass1.csv")

    write_rows(output_dir / "rq1_pass1_shared54_range_by_model.csv", range_rows(shared, baselines), [
        "model", "assignment", "configurations", "worst_profile", "worst_passed", "worst_pass_at_1",
        "best_profile", "best_passed", "best_pass_at_1", "range_passed", "range_pass_at_1",
        "baseline_passed", "baseline_pass_at_1",
    ])
    write_rows(output_dir / "rq1_pass1_shared54_vs_self_report_by_model.csv", baseline_summary_rows(shared, baselines), [
        "model", "assignment", "baseline_passed", "baseline_evaluated", "baseline_pass_at_1",
        "configurations", "beat_self_report", "beat_rate", "max_gain_passed",
        "max_gain_relative", "max_gain_pass_at_1",
    ])
    write_rows(output_dir / "rq1_pass1_shared54_by_emotion.csv", emotion_rows(shared), [
        "model", "assignment", "emotion", "configurations", "passed_sum", "evaluated_sum",
        "overall_pass_at_1", "mean_passed", "mean_pass_at_1",
    ])
    write_rows(output_dir / "rq1_pass1_shared54_by_personality.csv", personality_rows(shared), [
        "model", "assignment", "personality_group", "configurations", "passed_sum", "evaluated_sum",
        "overall_pass_at_1", "mean_passed", "mean_pass_at_1",
    ])
    print(f"shared54_rows={len(shared)}")


if __name__ == "__main__":
    main()
