import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generate_responses import generate_responses
from load_dataset import load_samples
from processing.assets import discover_persona_variants

LOGS_ROOT = ROOT / "logs"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-dir", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--max-revise-rounds", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--llm-url", required=True)
    p.add_argument("--api-key", default=None)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def print_summary(variants):
    by_emo = defaultdict(list)
    for v in variants:
        by_emo[v["emotion"]].append(v["team_dir_name"])
    print(f"[persona] found {len(by_emo)} emotions and {len(variants)} complete variants", flush=True)
    for emo in sorted(by_emo):
        teams = sorted(by_emo[emo])
        print(f"[persona] {emo}: {len(teams)} teams ({', '.join(teams)})", flush=True)


def main():
    args = parse_args()
    variants = discover_persona_variants(args.prompt_dir)
    if not variants:
        raise SystemExit(f"No persona variants found under prompt dir: {args.prompt_dir}")
    print_summary(variants)

    samples = list(load_samples(args.data_path))
    generate_responses(args=args, samples=samples, persona_variants=variants, logs_root=LOGS_ROOT)


if __name__ == "__main__":
    main()
