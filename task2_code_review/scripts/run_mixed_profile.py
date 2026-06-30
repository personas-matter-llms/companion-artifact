import argparse
import os
import re
import sys
import time
import traceback
from itertools import product
from pathlib import Path

from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, llm, personas, pipeline
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


ROLES = ("R1", "R2", "R3")
EMOTIONS = ("anger", "disgust", "fear", "happiness", "sadness", "neutral")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-dir", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--llm-url", required=True)
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--selected-persona", default=None)
    p.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    p.add_argument("--max-revise-rounds", type=int, default=config.MAX_REVISE_ROUNDS)
    p.add_argument("--max-tokens-reviewer", type=int, default=config.MAX_TOKENS_REVIEWER)
    p.add_argument("--max-tokens-judge", type=int, default=config.MAX_TOKENS_JUDGE)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def compact_persona_name(name):
    match = re.fullmatch(r"team(\d+)_([A-Z]+)_([a-z]+)", name)
    if not match:
        raise ValueError(f"selected persona directory must look like team05_HHH_happiness: {name}")
    number, traits, emotion = match.groups()
    if emotion not in EMOTIONS:
        raise ValueError(f"unknown emotion in selected persona directory: {name}")
    return f"{emotion[:3]}_t{number}{traits}"


def selected_persona_root(prompt_dir, selected_persona=None):
    root = Path(prompt_dir) / "selected_persona_description"
    return root / selected_persona if selected_persona else root


def load_selected_personas(prompt_dir, selected_persona=None):
    root = selected_persona_root(prompt_dir, selected_persona)
    if not root.is_dir():
        raise ValueError(f"selected persona directory not found: {root}")

    candidate_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    role_dirs = [
        path for path in candidate_dirs
        if any((path / f"{role}.md").is_file() for role in ROLES)
    ]
    if selected_persona is None and candidate_dirs and not any(p.name.startswith("team") for p in candidate_dirs):
        choices = ", ".join(path.name for path in candidate_dirs)
        raise ValueError(f"choose --selected-persona from: {choices}")

    selected = []
    for cell_dir in role_dirs:
        missing = [role for role in ROLES if not (cell_dir / f"{role}.md").is_file()]
        if missing:
            raise ValueError(f"{cell_dir} is missing {missing}")
        selected.append({
            "id": compact_persona_name(cell_dir.name),
            "name": cell_dir.name,
            "path": str(cell_dir),
            "persona": {
                role: personas.persona_text((cell_dir / f"{role}.md").read_text(encoding="utf-8"))
                for role in ROLES
            },
        })
    if len(selected) < 2:
        raise ValueError(f"need at least 2 selected persona directories under {root}")
    return selected


def role_configs(selected):
    configs = []
    for r1, r2, r3 in product(selected, repeat=3):
        if r1["name"] == r2["name"] == r3["name"]:
            continue
        name = f"R1.{r1['id']}__R2.{r2['id']}__R3.{r3['id']}"
        configs.append({
            "config_name": name,
            "r1_id": r1["id"],
            "r2_id": r2["id"],
            "r3_id": r3["id"],
            "r1_persona": r1["name"],
            "r2_persona": r2["name"],
            "r3_persona": r3["name"],
            "persona": {
                "R1": r1["persona"]["R1"],
                "R2": r2["persona"]["R2"],
                "R3": r3["persona"]["R3"],
            },
        })
    return configs


def build_config(args, model, role_config):
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
        "baseline_type": "mixed_profile",
        "emotion": "mixed_profile",
        "team": "mixed_profile",
        "team_dir_name": role_config["config_name"],
        "persona_cell": f"mixed_profile/{role_config['config_name']}",
        "config_name": role_config["config_name"],
        "r1_id": role_config["r1_id"],
        "r2_id": role_config["r2_id"],
        "r3_id": role_config["r3_id"],
        "r1_persona": role_config["r1_persona"],
        "r2_persona": role_config["r2_persona"],
        "r3_persona": role_config["r3_persona"],
        "selected_persona": args.selected_persona or "",
        "selected_persona_dir": str(selected_persona_root(args.prompt_dir, args.selected_persona)),
    }


def run_one_sample(args, sample, idx, total, run_dir, model, role_config):
    key = sample_key(sample, idx)
    sample_dir = run_dir / "samples" / f"{idx:04d}_{safe_dir_name(key)}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[mixed_profile | {role_config['config_name']} | {model}]"
    started = time.monotonic()
    print(f"{tag} {idx + 1}/{total} starting {key}", flush=True)

    try:
        result = pipeline.run_review_pipeline_with_personas(
            sample=sample,
            sample_idx=idx,
            persona_cell=f"mixed_profile/{role_config['config_name']}",
            persona=role_config["persona"],
        )
        record = {
            **sample,
            "_idx": idx,
            "_elapsed_s": round(time.monotonic() - started, 2),
            "_model": model,
            "_emotion": "mixed_profile",
            "_team_dir_name": role_config["config_name"],
            "_persona_cell": result["persona_cell"],
            "_mixed_profile_config": {
                key: role_config[key]
                for key in ("config_name", "r1_id", "r2_id", "r3_id", "r1_persona", "r2_persona", "r3_persona")
            },
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
        status = "ok"
    except KeyboardInterrupt:
        raise
    except Exception as error:
        record = {
            **sample,
            "_idx": idx,
            "_elapsed_s": round(time.monotonic() - started, 2),
            "_model": model,
            "_emotion": "mixed_profile",
            "_team_dir_name": role_config["config_name"],
            "_persona_cell": f"mixed_profile/{role_config['config_name']}",
            "_error": f"{type(error).__name__}: {error}",
        }
        write_json(sample_dir / "error.json", {
            "idx": idx,
            "error": record["_error"],
            "traceback": traceback.format_exc(),
        })
        write_json(sample_dir / "review.json", record)
        status = "error"

    print(f"{tag} {idx + 1}/{total} finished {key} dur={record['_elapsed_s']:.1f}s ({status})", flush=True)
    return record


def run_samples(args, samples, run_dir, model, role_config):
    workers = max(int(args.workers or 1), 1)
    return Parallel(n_jobs=workers, prefer="threads")(
        delayed(run_one_sample)(args, sample, idx, len(samples), run_dir, model, role_config)
        for idx, sample in enumerate(samples)
    )


def run_config(args, samples, batch_dir, model, role_config):
    configure_runtime(args, model)
    llm._client = None
    run_dir = batch_dir / f"{safe_dir_name(model)}_mixed_profile_{safe_dir_name(role_config['config_name'])}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", build_config(args, model, role_config))
    records = run_samples(args, samples, run_dir, model, role_config)
    write_jsonl(run_dir / "reviews.jsonl", records)
    write_json(run_dir / "review_summary.json", summarize_reviews(records))
    return {
        "run_dir": str(run_dir),
        "model": model,
        "baseline_type": "mixed_profile",
        "config_name": role_config["config_name"],
        "r1_persona": role_config["r1_persona"],
        "r2_persona": role_config["r2_persona"],
        "r3_persona": role_config["r3_persona"],
    }


def main():
    args = parse_args()
    selected = load_selected_personas(args.prompt_dir, args.selected_persona)
    configs = role_configs(selected)
    samples = read_jsonl(args.data_path)

    batch_dir = PROJECT_ROOT / "logs" / timestamp_slug()
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mixed_profile] teams={len(selected)} assignments={len(configs)} samples={len(samples)}", flush=True)
    print(f"[mixed_profile] teams: {', '.join(item['name'] for item in selected)}", flush=True)

    runs = []
    for model in args.models:
        for index, role_config in enumerate(configs, start=1):
            print(f"[mixed_profile] {model} {index}/{len(configs)} {role_config['config_name']}", flush=True)
            runs.append(run_config(args, samples, batch_dir, model, role_config))

    write_json(batch_dir / "mixed_profile_runs.json", {
        "experiment_type": "mixed_profile_assignment",
        "selected_persona": args.selected_persona or "",
        "selected_persona_dir": str(selected_persona_root(args.prompt_dir, args.selected_persona)),
        "team_count": len(selected),
        "assignment_count": len(configs),
        "all_same_configs_removed": len(selected),
        "data_path": str(Path(args.data_path)),
        "prompt_dir": str(Path(args.prompt_dir)),
        "models": args.models,
        "llm_url": args.llm_url,
        "temperature": args.temperature,
        "workers": max(int(args.workers or 1), 1),
        "baseline_type": "mixed_profile",
        "selected_personas": [
            {"id": item["id"], "name": item["name"], "path": item["path"]}
            for item in selected
        ],
        "runs": runs,
    })
    print(f"[mixed_profile] DONE batch={batch_dir}", flush=True)


if __name__ == "__main__":
    main()
