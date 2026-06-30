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


def load_mixed(path):
    rows = []
    for row in read_rows(path):
        rows.append({
            "model": row["model"],
            "assignment": row["assignment"],
            "planner_profile": row["planner_profile"],
            "implementer_profile": row["implementer_profile"],
            "reviewer_profile": row["reviewer_profile"],
            "passed": as_int(row["passed"]),
            "evaluated": as_int(row["evaluated"]),
            "pass_at_1": as_float(row["pass_at_1"]),
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


def load_shared_frontiers(path):
    grouped = defaultdict(list)
    for row in read_rows(path):
        key = (row["model"], row["emotion"], row["personality"])
        grouped[key].append(as_int(row["passed"]))

    frontiers = {model: {"passed": 0, "pass_at_1": 0.0} for model in MODELS}
    for (model, _, _), values in grouped.items():
        if model not in frontiers:
            continue
        evaluated = len(values)
        passed = sum(values)
        pass_at_1 = passed / evaluated if evaluated else 0.0
        if passed > frontiers[model]["passed"]:
            frontiers[model] = {"passed": passed, "pass_at_1": pass_at_1}
    return frontiers


def cell_name(row):
    return ";".join([row["planner_profile"], row["implementer_profile"], row["reviewer_profile"]])


def joined_assignments(rows):
    return " | ".join(cell_name(row) for row in sorted(rows, key=cell_name))


def range_rows(rows):
    out = []
    for model in MODELS:
        items = [row for row in rows if row["model"] == model]
        if not items:
            continue
        worst_passed = min(row["passed"] for row in items)
        best_passed = max(row["passed"] for row in items)
        worst_ties = [row for row in items if row["passed"] == worst_passed]
        best_ties = [row for row in items if row["passed"] == best_passed]
        worst = sorted(worst_ties, key=cell_name)[0]
        best = sorted(best_ties, key=cell_name)[0]
        out.append({
            "model": model,
            "assignment": "mixed24",
            "mixed_assignments": len(items),
            "worst_mixed_assignment": joined_assignments(worst_ties),
            "worst_passed": worst["passed"],
            "worst_pass_at_1": worst["pass_at_1"],
            "best_mixed_assignment": joined_assignments(best_ties),
            "best_passed": best["passed"],
            "best_pass_at_1": best["pass_at_1"],
            "range_passed": best["passed"] - worst["passed"],
            "range_pass_at_1": best["pass_at_1"] - worst["pass_at_1"],
        })
    return out


def detail_rows(rows, baselines):
    out = []
    for row in rows:
        baseline = baselines[row["model"]]
        delta_passed = row["passed"] - baseline["passed"]
        out.append({
            "model": row["model"],
            "assignment": "mixed24",
            "mixed_profile_assignment": cell_name(row),
            "planner_profile": row["planner_profile"],
            "implementer_profile": row["implementer_profile"],
            "reviewer_profile": row["reviewer_profile"],
            "passed": row["passed"],
            "evaluated": row["evaluated"],
            "pass_at_1": row["pass_at_1"],
            "baseline_passed": baseline["passed"],
            "baseline_pass_at_1": baseline["pass_at_1"],
            "delta_passed": delta_passed,
            "delta_pass_at_1": row["pass_at_1"] - baseline["pass_at_1"],
            "beat_self_report": int(delta_passed > 0),
        })
    return sorted(out, key=lambda row: (row["model"], -row["passed"], row["mixed_profile_assignment"]))


def summary_rows(details, baselines):
    out = []
    for model in MODELS:
        items = [row for row in details if row["model"] == model]
        if not items:
            continue
        baseline = baselines[model]
        best = max(items, key=lambda row: (row["passed"], row["mixed_profile_assignment"]))
        beats = sum(row["beat_self_report"] for row in items)
        max_gain_passed = best["passed"] - baseline["passed"]
        out.append({
            "model": model,
            "assignment": "mixed24",
            "baseline_passed": baseline["passed"],
            "baseline_evaluated": baseline["evaluated"],
            "baseline_pass_at_1": baseline["pass_at_1"],
            "mixed_assignments": len(items),
            "beat_self_report": beats,
            "beat_rate": beats / len(items) if items else 0.0,
            "max_gain_passed": max_gain_passed,
            "max_gain_relative": max_gain_passed / baseline["passed"] if baseline["passed"] else 0.0,
            "max_gain_pass_at_1": best["pass_at_1"] - baseline["pass_at_1"],
        })
    return out


def shared_frontier_rows(details, frontiers):
    out = []
    for model in MODELS:
        items = [row for row in details if row["model"] == model]
        if not items:
            continue
        best = max(items, key=lambda row: (row["passed"], row["mixed_profile_assignment"]))
        frontier = frontiers[model]
        delta_passed = best["passed"] - frontier["passed"]
        out.append({
            "model": model,
            "assignment": "mixed24",
            "best_mixed_passed": best["passed"],
            "best_mixed_pass_at_1": best["pass_at_1"],
            "best_shared_passed": frontier["passed"],
            "best_shared_pass_at_1": frontier["pass_at_1"],
            "delta_passed": delta_passed,
            "delta_pass_at_1": best["pass_at_1"] - frontier["pass_at_1"],
            "exceeds_best_shared": int(delta_passed > 0),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    mixed = load_mixed(data_dir / "mixed24_pass1.csv")
    baselines = load_baselines(data_dir / "self_report_pass1.csv")
    frontiers = load_shared_frontiers(data_dir / "shared54_pass1_instances.csv")
    details = detail_rows(mixed, baselines)
    summary = summary_rows(details, baselines)

    write_rows(output_dir / "rq2_pass1_mixed24_range_by_model.csv", range_rows(mixed), [
        "model", "assignment", "mixed_assignments", "worst_mixed_assignment", "worst_passed",
        "worst_pass_at_1", "best_mixed_assignment", "best_passed", "best_pass_at_1",
        "range_passed", "range_pass_at_1",
    ])
    write_rows(output_dir / "rq2_pass1_mixed24_vs_self_report_by_assignment.csv", details, [
        "model", "assignment", "mixed_profile_assignment", "planner_profile", "implementer_profile", "reviewer_profile",
        "passed", "evaluated", "pass_at_1", "baseline_passed", "baseline_pass_at_1",
        "delta_passed", "delta_pass_at_1", "beat_self_report",
    ])
    write_rows(output_dir / "rq2_pass1_mixed24_vs_self_report_by_model.csv", summary, [
        "model", "assignment", "baseline_passed", "baseline_evaluated", "baseline_pass_at_1",
        "mixed_assignments", "beat_self_report", "beat_rate", "max_gain_passed",
        "max_gain_relative", "max_gain_pass_at_1",
    ])
    write_rows(output_dir / "rq2_pass1_mixed24_vs_best_shared_by_model.csv", shared_frontier_rows(details, frontiers), [
        "model", "assignment", "best_mixed_passed", "best_mixed_pass_at_1", "best_shared_passed",
        "best_shared_pass_at_1", "delta_passed", "delta_pass_at_1", "exceeds_best_shared",
    ])

    print(f"mixed24_rows={len(mixed)}")


if __name__ == "__main__":
    main()
