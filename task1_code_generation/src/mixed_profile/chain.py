from execution.feedback import get_public_tests_feedback
from llm.openai_compatible import call
from processing.assets import load_io_prompt, load_task_prompt
from processing.parsing import extract_longest_code_block, extract_review_decision
from processing.prompting import build_user_prompt
from utils import RunResult


ROLES = ("planner", "implementer", "reviewer")


class MixedProfileChain:
    def __init__(
        self,
        *,
        personas: dict[str, str],
        model: str,
        max_revise_rounds: int,
        max_tokens: int,
        temperature: float,
        timeout: int,
        llm_url: str | None,
        api_key: str | None,
        prompt_dir: str,
    ):
        self.personas = personas
        self.model = model
        self.max_revise_rounds = max_revise_rounds
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.llm_url = llm_url
        self.api_key = api_key
        self.task_prompts = {role: load_task_prompt(role, prompt_dir) for role in ROLES}
        self.io_prompts = {testtype: load_io_prompt(testtype, prompt_dir) for testtype in ("stdin", "functional")}

    def _invoke(self, role, user_prompt):
        return call(
            self.personas[role], user_prompt,
            model=self.model, max_tokens=self.max_tokens, temperature=self.temperature,
            timeout=self.timeout, llm_url=self.llm_url, api_key=self.api_key,
        )

    def _trace(self, role, round_index, user_prompt, response, **extra):
        event = {
            "role": role,
            "round_index": round_index,
            "model": response.model,
            "provider": response.provider,
            "system_prompt": self.personas[role],
            "user_prompt": user_prompt,
            "response_text": response.text,
            "metrics": response.metrics,
            "duration_ms": response.duration_ms,
        }
        for k, v in extra.items():
            if v is not None:
                event[k] = v
        return event

    def run(self, sample):
        traces = []
        reviewer_rounds = 0
        testtype = sample.public_tests[0].testtype if sample.public_tests else None
        io_prompt = self.io_prompts.get(testtype)

        planner_user = build_user_prompt(self.task_prompts["planner"], sample, io_prompt=io_prompt)
        planner_resp = self._invoke("planner", planner_user)
        traces.append(self._trace("planner", 0, planner_user, planner_resp))
        plan_text = planner_resp.text

        impl_user = build_user_prompt(
            self.task_prompts["implementer"], sample,
            io_prompt=io_prompt, plan_text=plan_text,
        )
        impl_resp = self._invoke("implementer", impl_user)
        sol_text = impl_resp.text
        sol_code = extract_longest_code_block(impl_resp.text)
        traces.append(self._trace("implementer", 0, impl_user, impl_resp, solution_code=sol_code))
        implementer_rounds = 1
        final_decision = "UNKNOWN"

        for round_index in range(self.max_revise_rounds):
            feedback = get_public_tests_feedback(sol_code, sample, timeout_seconds=min(self.timeout, 6))
            reviewer_user = build_user_prompt(
                self.task_prompts["reviewer"], sample,
                io_prompt=io_prompt, plan_text=plan_text,
                solution_text=sol_text, solution_section_name="Current Solution",
                execution_feedback_text=feedback,
            )
            reviewer_resp = self._invoke("reviewer", reviewer_user)
            raw_decision = extract_review_decision(reviewer_resp.text)
            decision = "ACCEPT" if raw_decision == "UNKNOWN" else raw_decision
            traces.append(self._trace(
                "reviewer", round_index, reviewer_user, reviewer_resp,
                raw_decision=raw_decision, decision=decision,
                public_testcases_feedback=feedback,
            ))
            reviewer_rounds += 1

            final_decision = decision
            if decision == "ACCEPT":
                break

            revise_user = build_user_prompt(
                self.task_prompts["implementer"], sample,
                io_prompt=io_prompt, plan_text=plan_text,
                solution_text=sol_text, solution_section_name="Prior Solution",
                execution_feedback_text=feedback,
                reviewer_feedback_text=reviewer_resp.text,
            )
            revise_resp = self._invoke("implementer", revise_user)
            sol_text = revise_resp.text
            sol_code = extract_longest_code_block(revise_resp.text)
            traces.append(self._trace(
                "implementer", round_index + 1, revise_user, revise_resp,
                solution_code=sol_code,
            ))
            implementer_rounds += 1

        return RunResult(
            plan_text=plan_text,
            final_solution_text=sol_text,
            final_solution_code=sol_code,
            final_decision=final_decision,
            traces=traces,
            reviewer_rounds=reviewer_rounds,
            implementer_rounds=implementer_rounds,
        )
