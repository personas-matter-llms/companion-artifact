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


def as_float(value):
    return float(value) if str(value).strip() else 0.0


def load_mixed(path):
    rows = []
    for row in read_rows(path):
        rows.append({
            "model": row["model"],
            "assignment": row["assignment"],
            "writer1_profile": row["writer1_profile"],
            "writer2_profile": row["writer2_profile"],
            "supervisor_profile": row["supervisor_profile"],
            "bleu": as_float(row["bleu"]),
        })
    return rows


def load_baselines(path):
    return {row["model"]: {"bleu": as_float(row["bleu"])} for row in read_rows(path)}


def load_shared_frontiers(path):
    grouped = defaultdict(list)
    for row in read_rows(path):
        grouped[(row["model"], row["emotion"], row["personality"])].append(as_float(row["bleu"]))

    frontiers = {model: 0.0 for model in MODELS}
    for (model, _, _), values in grouped.items():
        if model in frontiers:
            frontiers[model] = max(frontiers[model], sum(values) / len(values) if values else 0.0)
    return frontiers


def cell_name(row):
    return ";".join([row["writer1_profile"], row["writer2_profile"], row["supervisor_profile"]])


def joined_assignments(rows):
    return " | ".join(cell_name(row) for row in sorted(rows, key=cell_name))


def range_rows(rows):
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
            "assignment": "mixed24",
            "mixed_assignments": len(items),
            "worst_mixed_assignment": joined_assignments(worst_ties),
            "worst_bleu": worst_bleu,
            "best_mixed_assignment": joined_assignments(best_ties),
            "best_bleu": best_bleu,
            "range_bleu": best_bleu - worst_bleu,
        })
    return out


def baseline_detail_rows(rows, baselines):
    out = []
    for row in rows:
        baseline = baselines[row["model"]]
        delta = row["bleu"] - baseline["bleu"]
        out.append({
            "model": row["model"],
            "assignment": "mixed24",
            "mixed_profile_assignment": cell_name(row),
            "writer1_profile": row["writer1_profile"],
            "writer2_profile": row["writer2_profile"],
            "supervisor_profile": row["supervisor_profile"],
            "bleu": row["bleu"],
            "baseline_bleu": baseline["bleu"],
            "delta_bleu": delta,
            "beat_self_report": int(delta > 0),
        })
    return sorted(out, key=lambda row: (row["model"], -row["bleu"], row["mixed_profile_assignment"]))


def baseline_summary_rows(details, baselines):
    out = []
    for model in MODELS:
        items = [row for row in details if row["model"] == model]
        if not items:
            continue
        baseline = baselines[model]
        best = max(items, key=lambda row: (row["bleu"], row["mixed_profile_assignment"]))
        beats = sum(row["beat_self_report"] for row in items)
        max_gain = best["bleu"] - baseline["bleu"]
        out.append({
            "model": model,
            "assignment": "mixed24",
            "baseline_bleu": baseline["bleu"],
            "mixed_assignments": len(items),
            "beat_self_report": beats,
            "beat_rate": beats / len(items) if items else 0.0,
            "max_gain_bleu": max_gain,
            "max_gain_relative": max_gain / baseline["bleu"] if baseline["bleu"] else 0.0,
        })
    return out


def shared_frontier_rows(details, frontiers):
    out = []
    for model in MODELS:
        items = [row for row in details if row["model"] == model]
        if not items:
            continue
        best = max(items, key=lambda row: (row["bleu"], row["mixed_profile_assignment"]))
        delta = best["bleu"] - frontiers[model]
        out.append({
            "model": model,
            "assignment": "mixed24",
            "best_mixed_bleu": best["bleu"],
            "best_shared_bleu": frontiers[model],
            "delta_bleu": delta,
            "exceeds_best_shared": int(delta > 0),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    mixed = load_mixed(data_dir / "mixed24_bleu.csv")
    baselines = load_baselines(data_dir / "self_report_bleu.csv")
    frontiers = load_shared_frontiers(data_dir / "shared54_bleu_instances.csv")
    details = baseline_detail_rows(mixed, baselines)

    write_rows(output_dir / "rq2_bleu_mixed24_range_by_model.csv", range_rows(mixed), [
        "model", "assignment", "mixed_assignments", "worst_mixed_assignment", "worst_bleu",
        "best_mixed_assignment", "best_bleu", "range_bleu",
    ])
    write_rows(output_dir / "rq2_bleu_mixed24_vs_self_report_by_assignment.csv", details, [
        "model", "assignment", "mixed_profile_assignment", "writer1_profile", "writer2_profile", "supervisor_profile",
        "bleu", "baseline_bleu", "delta_bleu", "beat_self_report",
    ])
    write_rows(output_dir / "rq2_bleu_mixed24_vs_self_report_by_model.csv", baseline_summary_rows(details, baselines), [
        "model", "assignment", "baseline_bleu", "mixed_assignments", "beat_self_report",
        "beat_rate", "max_gain_bleu", "max_gain_relative",
    ])
    write_rows(output_dir / "rq2_bleu_mixed24_vs_best_shared_by_model.csv", shared_frontier_rows(details, frontiers), [
        "model", "assignment", "best_mixed_bleu", "best_shared_bleu", "delta_bleu", "exceeds_best_shared",
    ])

    print(f"mixed24_rows={len(mixed)}")


if __name__ == "__main__":
    main()
