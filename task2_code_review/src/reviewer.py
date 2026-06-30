"""R1/R2 reviewer agents."""
from . import config, llm, prompts


def build_writer_prompt(role: str, patch_with_info: str, prior_feedback: str | None = None) -> str:
    parts = [
        prompts.TASK_PROMPTS[role],
        "",
        prompts.PATCH_VS_CONTEXT_INSTRUCTION,
        "",
        patch_with_info,
    ]
    if prior_feedback:
        parts.extend([
            "",
            "Senior supervisor revision feedback:",
            prior_feedback,
            "",
            "Regenerate your review based on the patch and this feedback. Output only the revised review.",
        ])
    return "\n".join(parts)


def write_review(
    persona_desc: str,
    role: str,
    patch_with_info: str,
    prior_feedback: str | None = None,
) -> dict:
    system_prompt = persona_desc.strip()
    user_prompt = build_writer_prompt(role, patch_with_info, prior_feedback)
    return llm.chat(
        system_prompt=system_prompt,
        user_message=user_prompt,
        max_tokens=config.MAX_TOKENS_REVIEWER,
    )
