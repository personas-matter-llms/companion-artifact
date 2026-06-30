import re

_PY_FENCE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_GENERIC_FENCE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_longest_code_block(text):
    text = text or ""
    py = [m.strip() for m in _PY_FENCE.findall(text) if m.strip()]
    if py:
        return max(py, key=len)
    generic = [m.strip() for m in _GENERIC_FENCE.findall(text) if m.strip()]
    if generic:
        return max(generic, key=len)
    return ""


# Patterns for extracting reviewer decisions from structured model outputs.
# Order matters: more-specific patterns first.
_DECISION_PATTERNS = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in (
    r"review\s*decision\s*(?:\([^)]*\))?\s*[:：]\s*(ACCEPT|REVISE)\b",
    r"review\s*decision\s*(?:\([^)]*\))?\s*[:：]\s*\*+\s*(ACCEPT|REVISE)\s*\*+",
    r"^\s*\*+\s*review\s+decision\s*:?\s*\*+\s*(ACCEPT|REVISE)\b",
    r"^\s*\d+\.\s*(?:\*+\s*)?decision(?:\s*\*+)?\s*[:：]\s*(ACCEPT|REVISE)\b",
    r"^\s*\d+\.\s*(?:\*+\s*)?decision\s*\((ACCEPT|REVISE)\)(?:\s*\*+)?\s*$",
    r"\bdecision\s*[:：]\s*(ACCEPT|REVISE)\b",
    r"\bdecision\s*[:：]\s*\*+\s*(ACCEPT|REVISE)\s*\*+",
    r"\bdecision\s*\((?:ACCEPT|REVISE)(?:\s*/\s*(?:ACCEPT|REVISE))?\)\s*[:：]\s*(ACCEPT|REVISE)\b",
    r"\bdecision\s*\((ACCEPT|REVISE)\)(?=[\s,;:.\)]|$)",
    r"^\s*\d+\.\s*decision\s*\((ACCEPT|REVISE)\)\s*$",
    r"^\s*\d+\.\s*decision\s*\((?:ACCEPT|REVISE)(?:\s*/\s*(?:ACCEPT|REVISE))?\)\s*[:：]\s*(ACCEPT|REVISE)\s*$",
    r"^\s*\d+\.\s*decision\s*[:：]\s*\*+\s*(ACCEPT|REVISE)\s*\*+",
    r"^\s*\*+\s*decision\s*\*+\s*[:：]\s*(ACCEPT|REVISE)\b",
    r"^\s*\*+\s*decision\s*:?\s*\*+\s*(ACCEPT|REVISE)\b",
    r"^\s*\*+\s*decision\s*\((ACCEPT|REVISE)\)\s*\*+\s*$",
    r"^\s*\d+\.\s*(ACCEPT|REVISE)\s*$",
    r"^\s*#+\s*decision\s*$[\s\S]*?^\s*\*+\s*(ACCEPT|REVISE)\s*\*+\s*$",
)]


def extract_review_decision(text):
    text = text or ""
    for p in _DECISION_PATTERNS:
        m = p.search(text)
        if m:
            return m.group(1).upper()
    for line in text.splitlines():
        s = line.strip().upper()
        if s in {"ACCEPT", "REVISE"}:
            return s
    return "UNKNOWN"
