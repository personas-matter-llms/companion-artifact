import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRIPT = Path(__file__).resolve()
ARTIFACT_ROOT = SCRIPT.parents[2]
TASK1_DATA = ARTIFACT_ROOT / "task1_code_generation" / "data"
TASK2_DATA = ARTIFACT_ROOT / "task2_code_review" / "data"
OUT_FIG = SCRIPT.parents[1] / "figures"

MODELS = [
    "qwen2_5_1_5b",
    "llama_3_1_8b",
    "mistral_small_3_24b_awq",
    "qwen2.5-32b-awq",
]
MODEL_LABELS = {
    "qwen2_5_1_5b": "Qwen 1.5B",
    "llama_3_1_8b": "Llama 8B",
    "mistral_small_3_24b_awq": "Mistral 24B",
    "qwen2.5-32b-awq": "Qwen 32B",
}


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def display_path(path):
    return path.relative_to(ARTIFACT_ROOT)


def fnum(value):
    return float(value) if value not in {"", None} else 0.0


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.8,
    })


def deterministic_offsets(n):
    pattern = [0, -0.03, 0.03, -0.06, 0.06, -0.09, 0.09, -0.12, 0.12]
    return [pattern[i % len(pattern)] for i in range(n)]


def code_generation_data():
    values = {model: [] for model in MODELS}
    for row in read_csv(TASK1_DATA / "mixed24_pass1.csv"):
        if row["model"] in MODELS and row.get("assignment") == "mixed24":
            values[row["model"]].append(100 * fnum(row["pass_at_1"]))

    frontiers = {model: 0.0 for model in MODELS}
    for row in read_csv(TASK1_DATA / "shared54_pass1.csv"):
        if row["model"] in MODELS and row.get("assignment") == "shared54":
            frontiers[row["model"]] = max(frontiers[row["model"]], 100 * fnum(row["pass_at_1"]))

    baselines = {
        row["model"]: 100 * fnum(row["pass_at_1"])
        for row in read_csv(TASK1_DATA / "self_report_pass1.csv")
    }
    return values, baselines, frontiers


def code_review_data():
    values = {model: [] for model in MODELS}
    for row in read_csv(TASK2_DATA / "mixed24_bleu.csv"):
        if row["model"] in MODELS and row.get("assignment") == "mixed24":
            values[row["model"]].append(fnum(row["bleu"]))

    frontiers = {model: 0.0 for model in MODELS}
    for row in read_csv(TASK2_DATA / "shared54_bleu.csv"):
        if row["model"] in MODELS and row.get("assignment") == "shared54":
            frontiers[row["model"]] = max(frontiers[row["model"]], fnum(row["bleu"]))

    baselines = {
        row["model"]: fnum(row["bleu"])
        for row in read_csv(TASK2_DATA / "self_report_bleu.csv")
    }
    return values, baselines, frontiers


def plot_distribution(values, baselines, frontiers, xlabel, filename, xlim, color, light, box_color):
    setup_style()
    model_gap = 0.5
    positions = [1 + (len(MODELS) - 1 - i) * model_gap for i in range(len(MODELS))]

    fig, ax = plt.subplots(figsize=(5.05, 1.85), constrained_layout=True)
    data = [values[model] for model in MODELS]
    boxplot = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.0},
        boxprops={"facecolor": box_color, "edgecolor": color, "linewidth": 1.0},
        whiskerprops={"color": color, "linewidth": 1.0},
        capprops={"color": color, "linewidth": 1.0},
        zorder=1,
    )
    for patch in boxplot["boxes"]:
        patch.set_alpha(0.72)

    for pos, model in zip(positions, MODELS):
        vals = sorted(values[model])
        offsets = deterministic_offsets(len(vals))
        ax.scatter(vals, [pos + off for off in offsets], s=2.52, color=light, alpha=0.58, edgecolor="none", zorder=2)
        ax.scatter([min(vals), max(vals)], [pos, pos], s=28, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax.scatter(baselines[model], pos, marker="D", s=24, color="#111111", edgecolor="white", linewidth=0.7, zorder=5)
        ax.scatter(frontiers[model], pos, marker="^", s=34, color="#111111", edgecolor="white", linewidth=0.7, zorder=6)

        gap = max(vals) - min(vals)
        label = f"+{gap:.1f}" if "Pass@1" in xlabel else f"+{gap:.2f}"
        offset = 0.65 if "Pass@1" in xlabel else 0.035
        ax.text(max(max(vals), baselines[model], frontiers[model]) + offset, pos, label, ha="left", va="center", fontsize=8.0, color=color)

    ax.set_yticks(positions)
    ax.set_yticklabels([MODEL_LABELS[model] for model in MODELS], fontsize=9.5)
    ax.set_ylim(min(positions) - 0.35, max(positions) + 0.35)
    ax.set_xlabel(xlabel, fontsize=10.5, labelpad=1)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color="#e1e1e1", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    legend = [
        Line2D([0], [0], color=light, marker="o", linewidth=0, markersize=3.4, label="Mixed-Profile Config"),
        Line2D([0], [0], color=box_color, marker="s", linewidth=0, markersize=6, label="IQR"),
        Line2D([0], [0], color=color, marker="o", linewidth=1.2, markersize=5, label="Min-Max"),
        Line2D([0], [0], color="#111111", marker="D", linewidth=0, markersize=4.2, label="Self-Report Baseline"),
        Line2D([0], [0], color="#111111", marker="^", linewidth=0, markersize=4.8, label="Best Shared-Profile Config"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.66), ncol=3, frameon=False, fontsize=7.7)

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(OUT_FIG / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    cg_values, cg_baselines, cg_frontiers = code_generation_data()
    cr_values, cr_baselines, cr_frontiers = code_review_data()

    plot_distribution(
        cg_values,
        cg_baselines,
        cg_frontiers,
        "Pass@1 (%)",
        "rq2_mixed_profile_distribution_code_generation",
        (0, 50),
        "#1f5f8b",
        "#4f7f9d",
        "#d9e6ee",
    )
    plot_distribution(
        cr_values,
        cr_baselines,
        cr_frontiers,
        "BLEU",
        "rq2_mixed_profile_distribution_code_review",
        (5.35, 8.18),
        "#c87912",
        "#a76712",
        "#f1dfc4",
    )

    print(display_path(OUT_FIG / "rq2_mixed_profile_distribution_code_generation.pdf"))
    print(display_path(OUT_FIG / "rq2_mixed_profile_distribution_code_review.pdf"))


if __name__ == "__main__":
    main()
