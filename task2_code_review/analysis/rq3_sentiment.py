import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "sentiment_top_bottom_messages.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "csv"


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


def as_float(value):
    return float(value) if str(value).strip() else 0.0


def summarize(items):
    message_count = len(items)
    negative = sum(as_int(row["sentiment_negative"]) for row in items)
    return {
        "messages": message_count,
        "negative_fraction": negative / message_count if message_count else 0.0,
        "negative_messages": negative,
        "nonnegative_messages": message_count - negative,
    }


def summary_rows(rows):
    grouped = defaultdict(list)
    meta = {}
    for row in rows:
        key = (
            row["model"], row["assignment"], row["cell_group"], row["emotion"],
            row["personality"], row["bleu"], row["role"],
        )
        grouped[key].append(row)
        meta[key] = {
            "model": row["model"],
            "assignment": row["assignment"],
            "cell_group": row["cell_group"],
            "emotion": row["emotion"],
            "personality": row["personality"],
            "bleu": as_float(row["bleu"]),
            "role": row["role"],
        }

    out = []
    for key in sorted(grouped):
        out.append({
            **meta[key],
            **summarize(grouped[key]),
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    rows = read_rows(args.data)
    out = summary_rows(rows)
    write_rows(
        Path(args.output_dir) / "rq3_sentiment_top_bottom_summary_by_role.csv",
        out,
        [
            "model", "assignment", "cell_group", "emotion", "personality",
            "bleu", "role", "messages", "negative_fraction",
            "negative_messages", "nonnegative_messages",
        ],
    )

    print(f"messages={len(rows)}")
    print(f"summary_rows={len(out)}")
    print(f"negative_messages={sum(as_int(row['sentiment_negative']) for row in rows)}")


if __name__ == "__main__":
    main()
