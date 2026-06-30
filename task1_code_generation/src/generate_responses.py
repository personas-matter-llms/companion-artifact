import time
import traceback
from pathlib import Path

from joblib import Parallel, delayed

from multi_agent import MultiAgentChain
from utils import utc_timestamp_slug, write_json, write_jsonl

DEFAULT_MAX_TOKENS = 2000
DEFAULT_TIMEOUT = 300


def build_config(args, *, emotion, team, team_dir_name, model, baseline_type="main"):
    return {
        "emotion": emotion,
        "team": team,
        "team_dir_name": team_dir_name,
        "baseline_type": baseline_type,
        "model": model,
        "max_revise_rounds": args.max_revise_rounds,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": args.temperature,
        "timeout": DEFAULT_TIMEOUT,
        "eval_timeout": 6,
        "prompt_dir": str(args.prompt_dir),
        "workers": max(int(getattr(args, "workers", 1) or 1), 1),
    }


def safe_dir_name(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "value"


def sum_token_usage(traces):
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {k: sum(e.get("metrics", {}).get(k, 0) for e in traces) for k in keys}


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


def make_chain(args, *, model, emotion, team_dir_name):
    return MultiAgentChain(
        team_dir_name=team_dir_name, model=model, emotion=emotion,
        max_revise_rounds=args.max_revise_rounds,
        max_tokens=DEFAULT_MAX_TOKENS, temperature=args.temperature, timeout=DEFAULT_TIMEOUT,
        llm_url=args.llm_url, api_key=args.api_key, prompt_dir=args.prompt_dir,
    )


def run_one_sample(*, chain, sample, idx, total, samples_dir, model, emotion, team_dir_name):
    sample_dir = samples_dir / sample.question_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    tag = f"[{emotion} | {team_dir_name} | {model}]"
    print(f"{tag} {idx}/{total} starting {sample.question_id}", flush=True)

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
        write_json(sample_dir / "generation.json", record)
        status = "ok"
    except KeyboardInterrupt:
        raise
    except Exception as error:
        write_json(sample_dir / "error.json", {
            "question_id": sample.question_id,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        record = {
            "question_id": sample.question_id,
            "generation_status": "error",
            "final_decision": "ERROR",
            "review_rounds": 0,
            "implementer_rounds": 0,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        write_json(sample_dir / "generation.json", record)
        status = "error"

    dur = time.monotonic() - started
    print(f"{tag} {idx}/{total} finished {sample.question_id} dur={dur:.1f}s ({status})", flush=True)
    return record


def run_samples(*, args, samples, samples_dir, model, emotion, team_dir_name):
    total = len(samples)
    workers = max(int(getattr(args, "workers", 1) or 1), 1)
    chain = make_chain(args, model=model, emotion=emotion, team_dir_name=team_dir_name)
    jobs = [
        {
            "chain": chain, "sample": sample, "idx": i, "total": total,
            "samples_dir": samples_dir, "model": model,
            "emotion": emotion, "team_dir_name": team_dir_name,
        }
        for i, sample in enumerate(samples, start=1)
    ]
    return Parallel(n_jobs=workers, prefer="threads")(delayed(run_one_sample)(**j) for j in jobs)


def generate_responses(*, args, samples, persona_variants, logs_root):
    batch_dir = logs_root / utc_timestamp_slug()
    batch_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for model in args.models:
        for v in persona_variants:
            emotion, team, team_dir_name = v["emotion"], v["team_id"], v["team_dir_name"]
            baseline_type = v.get("baseline_type", "main")
            run_dir = batch_dir / f"{safe_dir_name(model)}_{safe_dir_name(emotion)}_{safe_dir_name(team_dir_name)}"
            samples_dir = run_dir / "samples"
            run_dir.mkdir(parents=True, exist_ok=True)

            write_json(run_dir / "config.json", build_config(
                args, emotion=emotion, team=team, team_dir_name=team_dir_name,
                model=model, baseline_type=baseline_type,
            ))
            results = run_samples(
                args=args, samples=samples, samples_dir=samples_dir,
                model=model, emotion=emotion, team_dir_name=team_dir_name,
            )
            write_json(run_dir / "generation_summary.json", summarize_generation(results))
            runs.append({
                "run_dir": run_dir, "emotion": emotion, "team": team,
                "team_dir_name": team_dir_name, "baseline_type": baseline_type, "model": model,
            })

    return runs
