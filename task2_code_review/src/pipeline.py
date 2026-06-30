"""R1/R2 write reviews; R3 integrates them and can request revisions."""
from concurrent.futures import ThreadPoolExecutor

from . import config, judge, personas, reviewer, utils


ROLES = ("R1", "R2", "R3")


def trace_event(role: str, round_k: int, call: dict) -> dict:
    return {
        "role": role,
        "round_index": round_k,
        "model": call["model"],
        "system_prompt": call["system_prompt"],
        "user_prompt": call["user_prompt"],
        "response_text": call["text"],
        "metrics": call["metrics"],
        "duration_ms": call["duration_ms"],
    }


def sum_token_usage(traces: list[dict]) -> dict:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {key: sum(event.get("metrics", {}).get(key, 0) for event in traces) for key in keys}


def run_writer_round(
    *,
    patch_with_info: str,
    persona: dict[str, str],
    feedback: dict[str, str | None],
    targets: set[str],
    current: dict[str, str | None],
    sample_idx: int,
    round_k: int,
    persona_cell: str,
) -> tuple[dict[str, str], list[dict]]:
    outputs = dict(current)

    def run_role(role: str) -> tuple[str, dict]:
        call = reviewer.write_review(
            persona_desc=persona[role],
            role=role,
            patch_with_info=patch_with_info,
            prior_feedback=feedback.get(role),
        )
        return call["text"], trace_event(role, round_k, call)

    events = []
    runnable = sorted(role for role in ("R1", "R2") if role in targets)
    if runnable:
        with ThreadPoolExecutor(max_workers=min(config.REVIEWER_CONCURRENCY, len(runnable))) as pool:
            futures = {role: pool.submit(run_role, role) for role in runnable}
            for role in runnable:
                outputs[role], event = futures[role].result()
                events.append(event)

    if not outputs.get("R1") or not outputs.get("R2"):
        missing = [role for role in ("R1", "R2") if not outputs.get(role)]
        raise RuntimeError(f"missing reviewer output for {missing}")
    return {"R1": outputs["R1"], "R2": outputs["R2"]}, events


def run_review_pipeline_with_personas(
    sample: dict,
    sample_idx: int,
    persona_cell: str,
    persona: dict[str, str],
) -> dict:
    patch_with_info = utils.build_input(sample)

    r1_history = []
    r2_history = []
    r3_history = []
    trace = []
    judge_decisions = []
    decisions = []

    feedback = {"R1": None, "R2": None}
    targets = {"R1", "R2"}
    current = {"R1": None, "R2": None}
    final_review = ""

    for round_k in range(config.MAX_REVISE_ROUNDS + 1):
        current, writer_events = run_writer_round(
            patch_with_info=patch_with_info,
            persona=persona,
            feedback=feedback,
            targets=targets,
            current=current,
            sample_idx=sample_idx,
            round_k=round_k,
            persona_cell=persona_cell,
        )
        trace.extend(writer_events)
        r1_history.append(current["R1"])
        r2_history.append(current["R2"])

        r3_call = judge.supervise_reviews(
            persona_desc=persona["R3"],
            patch_with_info=patch_with_info,
            r1_output=current["R1"],
            r2_output=current["R2"],
        )
        r3_raw = r3_call["text"]
        parsed = judge.parse_supervisor_output(r3_raw)
        trace.append(trace_event("R3", round_k, r3_call))
        r3_history.append(r3_raw)
        judge_decisions.append(parsed)
        decisions.append(parsed["decision"])
        final_review = parsed["final_review"]

        if parsed["decision"] != "REVISE" or round_k == config.MAX_REVISE_ROUNDS:
            break

        targets = set(parsed["targets"])
        feedback = {
            "R1": parsed.get("feedback_R1") if "R1" in targets else None,
            "R2": parsed.get("feedback_R2") if "R2" in targets else None,
        }

    return {
        "patch_with_info": patch_with_info,
        "persona_cell": persona_cell,
        "sample_idx": sample_idx,
        "rounds_executed": len(decisions),
        "decisions": decisions,
        "judge_decisions": judge_decisions,
        "r1_history": r1_history,
        "r2_history": r2_history,
        "r3_history": r3_history,
        "trace": trace,
        "token_usage": sum_token_usage(trace),
        "final": final_review,
    }


def run_review_pipeline(sample: dict, sample_idx: int, persona_cell: str) -> dict:
    persona = {role: personas.load_persona(persona_cell, role) for role in ROLES}
    return run_review_pipeline_with_personas(
        sample=sample,
        sample_idx=sample_idx,
        persona_cell=persona_cell,
        persona=persona,
    )
