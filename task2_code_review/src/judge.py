"""R3 supervisor agent."""
import re

from . import config, llm, prompts

LABELS = {
    "decision": "Decision",
    "final_review": "Final Review",
    "writer1": "Writer 1 revision suggestions",
    "writer2": "Writer 2 revision suggestions",
}

NUMBERED_SUGGESTION = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$")
EMPTY_SUGGESTION = re.compile(
    r"^\s*(?:\d+[.)]\s*)?\*{0,2}(?:none|n/a|na|not needed|no revision needed)\*{0,2}\s*[.!。]?\s*$",
    re.IGNORECASE,
)


def build_supervisor_prompt(patch_with_info: str, r1_output: str, r2_output: str) -> str:
    return "\n".join([
        prompts.TASK_PROMPTS["R3"],
        "",
        prompts.PATCH_VS_CONTEXT_INSTRUCTION,
        "",
        patch_with_info,
        "",
        "Reviewer R1's review:",
        r1_output,
        "",
        "Reviewer R2's review:",
        r2_output,
    ])


def supervise_reviews(
    persona_desc: str,
    patch_with_info: str,
    r1_output: str,
    r2_output: str,
) -> dict:
    system_prompt = persona_desc.strip()
    user_prompt = build_supervisor_prompt(patch_with_info, r1_output, r2_output)
    return llm.chat(
        system_prompt=system_prompt,
        user_message=user_prompt,
        max_tokens=config.MAX_TOKENS_JUDGE,
    )


def _heading_pattern(label: str) -> str:
    # Model outputs may decorate the canonical headers with markdown or bullets.
    escaped = re.escape(label).replace(r"\ ", r"\s+").replace("_", r"[\s_]+")
    return (
        r"^\s*(?:[-*]\s*|\d+[.)]\s*)?"
        r"(?:#+\s*)?"
        r"\*{0,2}\[?\s*"
        rf"{escaped}"
        r"\s*\]?\*{0,2}\s*:?\s*"
        r"\*{0,2}\s*"
    )


def _find_headings(text: str) -> list[tuple[int, int, str]]:
    found = []
    for name, label in LABELS.items():
        pattern = re.compile(_heading_pattern(label), re.IGNORECASE | re.MULTILINE)
        for match in pattern.finditer(text):
            found.append((match.start(), match.end(), name))
    return sorted(found, key=lambda item: item[0])


def _block(text: str, name: str) -> str | None:
    headings = _find_headings(text)
    for i, (start, end, key) in enumerate(headings):
        if key != name:
            continue
        next_start = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        return text[end:next_start].strip()
    return None


def _extract_feedback_suggestions(text: str | None) -> str | None:
    if not text:
        return None

    suggestions = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if EMPTY_SUGGESTION.match(line):
            continue
        match = NUMBERED_SUGGESTION.match(line)
        if not match:
            break
        number = int(match.group(1))
        if number > 3:
            break
        body = match.group(2).strip()
        if EMPTY_SUGGESTION.match(body):
            continue
        suggestions.append(body)
        if len(suggestions) >= 3:
            break

    if not suggestions:
        return None
    return "\n".join(f"{i}. {text}" for i, text in enumerate(suggestions, start=1))


def _normalize_decision(value: str | None) -> str | None:
    text = re.sub(r"[*`\[\]().:]", " ", value or "")
    text = re.sub(r"\s+", " ", text).strip()
    low = text.lower()
    if not low:
        return None
    if re.search(r"\baccept\b", low) and low.startswith(("accept or revi", "accept / revi")):
        return "REVISE"
    if re.search(r"\baccept\b", low):
        return "ACCEPT"
    if low.startswith("revi"):
        return "REVISE"
    return None


def _parse_decision(raw: str) -> str:
    block = _block(raw, "decision")
    if block:
        first_line = block.splitlines()[0] if block.splitlines() else block
        decision = _normalize_decision(first_line)
        if decision:
            return decision

    patterns = (
        r"review\s+decision\s*(?:\([^)]*\))?\s*:\s*\*{0,2}\s*([^\n\r]+)",
        r"\bdecision\s*(?:\([^)]*\))?\s*:\s*\*{0,2}\s*([^\n\r]+)",
        r"^\s*(?:[-*]\s*|\d+[.)]\s*)?\*{0,2}([^\n\r]+?)\*{0,2}\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE | re.MULTILINE)
        if match:
            decision = _normalize_decision(match.group(1))
            if decision:
                return decision

    for line in raw.splitlines():
        value = _normalize_decision(line)
        if value in {"ACCEPT", "REVISE"}:
            return value
    return "ACCEPT"


def _needs_revision(text: str | None) -> bool:
    if not text:
        return False
    empty_markers = {
        "none",
        "n/a",
        "na",
        "no",
        "no revision needed",
        "not needed",
    }
    lines = []
    for line in text.splitlines():
        cleaned = line.strip().strip("*`[] \t\r\n-.。").strip()
        cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned).strip()
        if cleaned:
            lines.append(cleaned.lower())
    if not lines:
        return False
    return any(line not in empty_markers for line in lines)


def parse_supervisor_output(raw: str) -> dict:
    decision = _parse_decision(raw)
    final_review = _block(raw, "final_review") or raw.strip()

    targets: set[str] = set()
    feedback_r1 = _extract_feedback_suggestions(_block(raw, "writer1"))
    feedback_r2 = _extract_feedback_suggestions(_block(raw, "writer2"))

    if decision == "REVISE":
        if _needs_revision(feedback_r1):
            targets.add("R1")
        else:
            feedback_r1 = None
        if _needs_revision(feedback_r2):
            targets.add("R2")
        else:
            feedback_r2 = None
        if not targets:
            decision = "ACCEPT"
    else:
        feedback_r1 = None
        feedback_r2 = None

    return {
        "decision": decision,
        "targets": sorted(targets),
        "feedback_R1": feedback_r1,
        "feedback_R2": feedback_r2,
        "final_review": final_review,
    }
