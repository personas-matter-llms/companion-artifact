from dataclasses import dataclass


@dataclass(frozen=True)
class LCBTestCase:
    input: str
    output: str
    testtype: str


@dataclass(frozen=True)
class LCBSample:
    question_id: str
    question_title: str
    difficulty: str
    prompt_text: str
    public_tests: list[LCBTestCase]
    private_tests: list[LCBTestCase]
    starter_code: str
    fn_name: str | None = None


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    metrics: dict[str, int]
    duration_ms: int


@dataclass
class RunResult:
    plan_text: str
    final_solution_text: str
    final_solution_code: str
    final_decision: str
    traces: list[dict]
    reviewer_rounds: int
    implementer_rounds: int
