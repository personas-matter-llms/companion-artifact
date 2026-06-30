def build_user_prompt(task_prompt, sample, *, io_prompt=None, plan_text=None,
                      solution_text=None, solution_section_name="Solution",
                      reviewer_feedback_text=None, execution_feedback_text=None):
    parts = [task_prompt.strip()]
    if io_prompt:
        parts += ["", io_prompt.strip()]
    parts += ["", "[Question]", sample.prompt_text.strip()]
    if plan_text:
        parts += ["", "[Plan]", plan_text.strip()]
    if solution_text:
        parts += ["", f"[{solution_section_name}]", solution_text.strip()]
    if execution_feedback_text:
        parts += ["", "[Public Test Execution Summary]", execution_feedback_text.strip()]
    if reviewer_feedback_text:
        parts += ["", "[Reviewer Feedback]", reviewer_feedback_text.strip()]
    return "\n".join(parts).strip()
