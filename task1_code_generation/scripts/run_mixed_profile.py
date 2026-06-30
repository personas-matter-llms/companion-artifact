import argparse
import itertools
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from joblib import Parallel, delayed

from load_dataset import load_samples
from processing.assets import PERSONA_MARKERS
from mixed_profile.chain import ROLES, MixedProfileChain
from utils import utc_timestamp_slug, write_json, write_jsonl


LOGS_ROOT = ROOT / "logs"
DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 300


def parse_args():
    parser = argparse.ArgumentParser(description="Run mixed-profile persona assignment experiment.")
    parser.add_argument("--prompt-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--max-revise-rounds", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-url", required=True)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--selected-persona", default=None)
    return parser.parse_args()


def safe_dir_name(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "value"


def team_id(team_dir_name):
    return team_dir_name.split("_", 1)[0].removeprefix("team")


def assignment_name(planner_team, implementer_team, reviewer_team):
    return f"P{team_id(planner_team)}_I{team_id(implementer_team)}_R{team_id(reviewer_team)}"


def read_persona(path):
    text = path.read_text(encoding="utf-8")
    for marker in PERSONA_MARKERS:
        if marker in text:
            return text.split(marker, 1)[1].strip()
    raise ValueError(f"No known persona marker in {path}")


def selected_persona_root(prompt_dir, selected_persona=None):
    root = Path(prompt_dir) / "selected_persona_description"
    return root / selected_persona if selected_persona else root


def load_selected_personas(prompt_dir, selected_persona=None):
    root = selected_persona_root(prompt_dir, selected_persona)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing selected persona directory: {root}")

    candidate_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if selected_persona is None and candidate_dirs and not any(p.name.startswith("team") for p in candidate_dirs):
        choices = ", ".join(p.name for p in candidate_dirs)
        raise ValueError(f"choose --selected-persona from: {choices}")

    selected = {}
    for team_dir in sorted(p for p in candidate_dirs if p.name.startswith("team")):
        selected[team_dir.name] = {}
        for role in ROLES:
            path = team_dir / f"{role}.md"
            if not path.is_file():
                raise FileNotFoundError(f"Missing selected persona file: {path}")
            selected[team_dir.name][role] = read_persona(path)

    if not selected:
        raise ValueError(f"No selected team directories found under {root}")
    return selected


def build_role_configs(selected_personas):
    teams = sorted(selected_personas)
    configs = []
    for plan, impl, rev in itertools.product(teams, repeat=3):
        if plan == impl == rev:
            continue
        configs.append({
            "assignment_name": assignment_name(plan, impl, rev),
            "planner_team": plan,
            "implementer_team": impl,
            "reviewer_team": rev,
            "personas": {
                "planner": selected_personas[plan]["planner"],
                "implementer": selected_personas[impl]["implementer"],
                "reviewer": selected_personas[rev]["reviewer"],
            },
        })
    return configs


def sum_token_usage(traces):
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {k: sum(e.get("metrics", {}).get(k, 0) for e in traces) for k in keys}


def write_implementer_round_files(sample_dir, traces):
    for path in sample_dir.glob("implementer_round*.py"):
        path.unlink()
    for event in traces:
        if event.get("role") != "implementer":
            continue
        round_index = event.get("round_index")
        code = event.get("solution_code")
        if not isinstance(round_index, int) or not code:
            continue
        (sample_dir / f"implementer_round{round_index}.py").write_text(code, encoding="utf-8")


def run_one_sample(*, chain, sample, sample_index, total_samples, samples_dir, model, name):
    sample_dir = samples_dir / sample.question_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tag = f"[mixed_profile | {name} | {model}]"
    print(f"{tag} {sample_index}/{total_samples} starting {sample.question_id}", flush=True)

    try:
        result = chain.run(sample)
        write_jsonl(sample_dir / "trace.jsonl", result.traces)
        write_implementer_round_files(sample_dir, result.traces)
        (sample_dir / "final_solution.py").write_text(result.final_solution_code, encoding="utf-8")
        record = {
            "question_id": sample.question_id,
            "generation_status": "ok",
            "final_decision": result.final_decision,
            "review_rounds": result.reviewer_rounds,
            "implementer_rounds": result.implementer_rounds,
            "token_usage": sum_token_usage(result.traces),
        }
        err_path = sample_dir / "error.json"
        if err_path.exists():
            err_path.unlink()
        status = "ok"
    except KeyboardInterrupt:
        raise
    except Exception as error:
        write_json(
            sample_dir / "error.json",
            {
                "question_id": sample.question_id,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        record = {
            "question_id": sample.question_id,
            "generation_status": "error",
            "final_decision": "ERROR",
            "review_rounds": 0,
            "implementer_rounds": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        status = "error"

    write_json(sample_dir / "generation.json", record)
    duration = time.monotonic() - started
    print(f"{tag} {sample_index}/{total_samples} finished {sample.question_id} dur={duration:.1f}s ({status})", flush=True)
    return record


def _avg(values):
    return (sum(values) / len(values)) if values else 0.0


def summarize_generation(results):
    ok = [r for r in results if r["generation_status"] == "ok"]
    errs = [r for r in results if r["generation_status"] == "error"]
    tokens = [r["token_usage"] for r in results if r["token_usage"]["total_tokens"] > 0]
    return {
        "total_samples": len(results),
        "generated_samples": len(ok),
        "generation_error_samples": len(errs),
        "average_review_rounds": _avg([r["review_rounds"] for r in results]),
        "average_implementer_rounds": _avg([r["implementer_rounds"] for r in results]),
        "average_prompt_tokens": _avg([t["prompt_tokens"] for t in tokens]),
        "average_completion_tokens": _avg([t["completion_tokens"] for t in tokens]),
        "average_total_tokens": _avg([t["total_tokens"] for t in tokens]),
    }


def run_config(*, args, samples, batch_dir, model, config):
    name = config["assignment_name"]
    run_dir = batch_dir / f"{safe_dir_name(model)}_mixed_profile_{name}"
    samples_dir = run_dir / "samples"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        run_dir / "config.json",
        {
            "emotion": "mixed_profile",
            "team": "mixed_profile",
            "team_dir_name": name,
            "baseline_type": "mixed_profile",
            "model": model,
            "max_revise_rounds": args.max_revise_rounds,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": args.temperature,
            "timeout": DEFAULT_TIMEOUT,
            "eval_timeout": 6,
            "prompt_dir": str(args.prompt_dir),
            "workers": max(int(args.workers or 1), 1),
            "assignment_name": name,
            "planner_team": config["planner_team"],
            "implementer_team": config["implementer_team"],
            "reviewer_team": config["reviewer_team"],
            "planner_id": f"P{team_id(config['planner_team'])}",
            "implementer_id": f"I{team_id(config['implementer_team'])}",
            "reviewer_id": f"R{team_id(config['reviewer_team'])}",
            "selected_persona": args.selected_persona or "",
            "selected_persona_dir": str(selected_persona_root(args.prompt_dir, args.selected_persona)),
        },
    )

    chain = MixedProfileChain(
        personas=config["personas"],
        model=model,
        max_revise_rounds=args.max_revise_rounds,
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=args.temperature,
        timeout=DEFAULT_TIMEOUT,
        llm_url=args.llm_url,
        api_key=args.api_key,
        prompt_dir=args.prompt_dir,
    )
    total_samples = len(samples)
    jobs = [
        {
            "chain": chain,
            "sample": sample,
            "sample_index": index,
            "total_samples": total_samples,
            "samples_dir": samples_dir,
            "model": model,
            "name": name,
        }
        for index, sample in enumerate(samples, start=1)
    ]
    results = Parallel(n_jobs=max(int(args.workers or 1), 1), prefer="threads")(
        delayed(run_one_sample)(**job) for job in jobs
    )
    write_json(run_dir / "generation_summary.json", summarize_generation(results))
    return {
        "run_dir": str(run_dir),
        "model": model,
        "assignment_name": name,
        "planner_team": config["planner_team"],
        "implementer_team": config["implementer_team"],
        "reviewer_team": config["reviewer_team"],
    }


def main():
    args = parse_args()
    selected_personas = load_selected_personas(args.prompt_dir, args.selected_persona)
    configs = build_role_configs(selected_personas)
    samples = list(load_samples(args.data_path))

    print(f"[mixed_profile] teams={len(selected_personas)} assignments={len(configs)} samples={len(samples)}", flush=True)
    print(f"[mixed_profile] teams: {', '.join(sorted(selected_personas))}", flush=True)

    batch_dir = LOGS_ROOT / utc_timestamp_slug()
    batch_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for model in args.models:
        for i, cfg in enumerate(configs, start=1):
            print(f"[mixed_profile] {model} {i}/{len(configs)} {cfg['assignment_name']}", flush=True)
            runs.append(run_config(args=args, samples=samples, batch_dir=batch_dir, model=model, config=cfg))

    write_json(
        batch_dir / "mixed_profile_runs.json",
        {
            "experiment_type": "mixed_profile_assignment",
            "selected_persona": args.selected_persona or "",
            "selected_persona_dir": str(selected_persona_root(args.prompt_dir, args.selected_persona)),
            "team_count": len(selected_personas),
            "assignment_count": len(configs),
            "all_same_configs_removed": len(selected_personas),
            "models": args.models,
            "runs": runs,
        },
    )
    print(f"[mixed_profile] DONE batch={batch_dir}", flush=True)


if __name__ == "__main__":
    main()
