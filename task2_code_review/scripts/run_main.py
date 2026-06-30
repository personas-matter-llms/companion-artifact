import argparse
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from src.utils import (
    configure_runtime,
    read_jsonl,
    safe_dir_name,
    sample_key,
    summarize_reviews,
    timestamp_slug,
    write_json,
    write_jsonl,
)


LOGS_ROOT = ROOT / "logs"
ROLES = ("R1", "R2", "R3")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-dir", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--llm-url", required=True)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--max-revise-rounds", type=int, default=config.MAX_REVISE_ROUNDS)
    p.add_argument("--max-tokens-reviewer", type=int, default=config.MAX_TOKENS_REVIEWER)
    p.add_argument("--max-tokens-judge", type=int, default=config.MAX_TOKENS_JUDGE)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--persona-cell", default=None)
    return p.parse_args()


def complete_cell(cell_dir):
    return cell_dir.is_dir() and all((cell_dir / f"{role}.md").is_file() for role in ROLES)


def discover_persona_variants(prompt_dir):
    root = Path(prompt_dir) / "actual_output_persona_description"
    variants = []
    for emotion_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for team_dir in sorted(path for path in emotion_dir.iterdir() if complete_cell(path)):
            variants.append({
                "emotion": emotion_dir.name,
                "team_dir_name": team_dir.name,
                "persona_cell": f"{emotion_dir.name}/{team_dir.name}",
            })
    return variants


def print_summary(variants):
    by_emotion = defaultdict(list)
    for variant in variants:
        by_emotion[variant["emotion"]].append(variant["team_dir_name"])
    print(f"[persona] found {len(by_emotion)} emotions and {len(variants)} complete variants", flush=True)
    for emotion in sorted(by_emotion):
        teams = sorted(by_emotion[emotion])
        print(f"[persona] {emotion}: {len(teams)} teams ({', '.join(teams)})", flush=True)


def build_config(args, model, variant):
    return {
        "data_path": str(Path(args.data_path)),
        "prompt_dir": str(Path(args.prompt_dir)),
        "model": model,
        "llm_url": args.llm_url,
        "temperature": args.temperature,
        "max_revise_rounds": args.max_revise_rounds,
        "max_tokens_reviewer": args.max_tokens_reviewer,
        "max_tokens_judge": args.max_tokens_judge,
        "workers": max(int(args.workers or 1), 1),
        "emotion": variant["emotion"],
        "team_dir_name": variant["team_dir_name"],
        "persona_cell": variant["persona_cell"],
    }


def run_one_sample(args, sample, idx, total, run_dir, model, variant):
    from src.pipeline import run_review_pipeline

    sample_dir = run_dir / "samples" / f"{idx:04d}_{safe_dir_name(sample_key(sample, idx))}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[{variant['persona_cell']} | {model}]"
    started = time.monotonic()
    print(f"{tag} {idx + 1}/{total} starting", flush=True)

    try:
        result = run_review_pipeline(sample, sample_idx=idx, persona_cell=variant["persona_cell"])
        record = {
            **sample,
            "_idx": idx,
            "_elapsed_s": round(time.monotonic() - started, 2),
            "_model": model,
            "_emotion": variant["emotion"],
            "_team_dir_name": variant["team_dir_name"],
            "_persona_cell": result["persona_cell"],
            "_rounds_executed": result["rounds_executed"],
            "_decisions": result["decisions"],
            "token_usage": result["token_usage"],
            "local_r1_history": result["r1_history"],
            "local_r2_history": result["r2_history"],
            "local_r3_history": result["r3_history"],
            "local_judge_decisions": result["judge_decisions"],
            "local_final": result["final"],
        }
        write_jsonl(sample_dir / "trace.jsonl", result["trace"])
        write_json(sample_dir / "review.json", record)
        error_path = sample_dir / "error.json"
        if error_path.exists():
            error_path.unlink()
        status = "ok"
    except KeyboardInterrupt:
        raise
    except Exception as error:
        record = {
            **sample,
            "_idx": idx,
            "_elapsed_s": round(time.monotonic() - started, 2),
            "_model": model,
            "_emotion": variant["emotion"],
            "_team_dir_name": variant["team_dir_name"],
            "_persona_cell": variant["persona_cell"],
            "_error": f"{type(error).__name__}: {error}",
        }
        write_json(sample_dir / "error.json", {
            "idx": idx,
            "error": record["_error"],
            "traceback": traceback.format_exc(),
        })
        trace_path = sample_dir / "trace.jsonl"
        if trace_path.exists():
            trace_path.unlink()
        write_json(sample_dir / "review.json", record)
        status = "error"

    print(f"{tag} {idx + 1}/{total} finished dur={record['_elapsed_s']:.1f}s ({status})", flush=True)
    return record


def run_samples(args, samples, run_dir, model, variant):
    workers = max(int(args.workers or 1), 1)
    return Parallel(n_jobs=workers, prefer="threads")(
        delayed(run_one_sample)(args, sample, idx, len(samples), run_dir, model, variant)
        for idx, sample in enumerate(samples)
    )


def run_variant(args, samples, batch_dir, model, variant):
    configure_runtime(args, model)
    run_dir = batch_dir / f"{safe_dir_name(model)}_{safe_dir_name(variant['persona_cell'])}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", build_config(args, model, variant))
    records = run_samples(args, samples, run_dir, model, variant)
    write_jsonl(run_dir / "reviews.jsonl", records)
    write_json(run_dir / "review_summary.json", summarize_reviews(records))
    return {
        "run_dir": str(run_dir),
        "model": model,
        "emotion": variant["emotion"],
        "team_dir_name": variant["team_dir_name"],
        "persona_cell": variant["persona_cell"],
    }


def main():
    args = parse_args()
    variants = discover_persona_variants(args.prompt_dir)
    if args.persona_cell:
        variants = [variant for variant in variants if variant["persona_cell"] == args.persona_cell]
    if not variants:
        raise SystemExit(f"No persona variants found under prompt dir: {args.prompt_dir}")
    print_summary(variants)

    samples = read_jsonl(args.data_path)
    batch_dir = LOGS_ROOT / timestamp_slug()
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {len(samples)} samples from {args.data_path}", flush=True)
    print(f"Batch dir: {batch_dir}", flush=True)

    runs = []
    for model in args.models:
        for variant in variants:
            runs.append(run_variant(args, samples, batch_dir, model, variant))

    write_json(batch_dir / "batch_summary.json", {
        "data_path": str(Path(args.data_path)),
        "prompt_dir": str(Path(args.prompt_dir)),
        "models": args.models,
        "llm_url": args.llm_url,
        "temperature": args.temperature,
        "workers": max(int(args.workers or 1), 1),
        "runs": runs,
    })


if __name__ == "__main__":
    main()
