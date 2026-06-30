"""Prompt loading for Task 2 code review."""
from pathlib import Path


PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts"
TASK_PROMPT_DIR = PROMPT_ROOT / "task_prompt"


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_task_prompts(prompt_root: Path) -> dict[str, str]:
    task_dir = prompt_root / "task_prompt"
    return {
        "R1": _load_text(task_dir / "r1_writer_task.md"),
        "R2": _load_text(task_dir / "r2_writer_task.md"),
        "R3": _load_text(task_dir / "supervisor_task.md"),
    }


TASK_PROMPTS: dict[str, str] = load_task_prompts(PROMPT_ROOT)

PATCH_VS_CONTEXT_INSTRUCTION = (
    "Please comment on the patch after fully understanding any additional information "
    "and the patch itself, and avoid commenting on the additional information."
)
