import argparse
import csv
import itertools
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data"

MODELS = [
    ("qwen2_5_1_5b", "Qwen 1.5B"),
    ("llama_3_1_8b", "Llama 8B"),
    ("mistral_small_3_24b_awq", "Mistral 24B"),
    ("qwen2.5-32b-awq", "Qwen 32B"),
]


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        average = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = average
        i = j + 1
    return ranks


def pearson(xs, ys):
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    return numerator / (denominator_x * denominator_y) if denominator_x and denominator_y else 0.0


def load_shared(path):
    grouped = defaultdict(list)
    for row in read_rows(path):
        profile = (row["emotion"], row["personality"])
        grouped[(row["model"], profile)].append(int(float(row["passed"])))

    by_model = {model: {} for model, _ in MODELS}
    for (model, profile), values in grouped.items():
        by_model[model][profile] = sum(values) / len(values) if values else 0.0
    return by_model


def spearman_rows(by_model):
    rows = []
    for (model_a, label_a), (model_b, label_b) in itertools.combinations(MODELS, 2):
        profiles = sorted(set(by_model[model_a]) & set(by_model[model_b]))
        rho = pearson(
            average_ranks([by_model[model_a][profile] for profile in profiles]),
            average_ranks([by_model[model_b][profile] for profile in profiles]),
        )
        rows.append({
            "model_a": label_a,
            "model_b": label_b,
            "configurations": len(profiles),
            "spearman_rho": rho,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    args = parser.parse_args()

    by_model = load_shared(Path(args.data_dir) / "shared54_pass1_instances.csv")
    rows = spearman_rows(by_model)
    mean_rho = sum(row["spearman_rho"] for row in rows) / len(rows)

    print("RQ1 Spearman rank transfer: code generation, shared54")
    for row in rows:
        print(
            f"  {row['model_a']} vs {row['model_b']}: "
            f"rho={row['spearman_rho']:.6f} (n={row['configurations']})"
        )
    print(f"mean_spearman_rho={mean_rho:.6f}")


if __name__ == "__main__":
    main()
