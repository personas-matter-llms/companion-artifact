"""Persona loading for mixed-profile code review."""

from . import prompts


PERSONA_MARKER = "Persona description:"


def persona_text(text: str) -> str:
    if PERSONA_MARKER not in text:
        raise ValueError("persona file is missing Persona description marker")
    return text.split(PERSONA_MARKER, 1)[1].strip()


def _persona_path(persona_cell: str, role: str):
    return (
        prompts.PROMPT_ROOT
        / "actual_output_persona_description"
        / persona_cell
        / f"{role}.md"
    )


def load_persona(persona_cell: str, role: str) -> str:
    """Load the generated persona paragraph for one role in one cell."""
    path = _persona_path(persona_cell, role)
    return persona_text(path.read_text(encoding="utf-8"))
