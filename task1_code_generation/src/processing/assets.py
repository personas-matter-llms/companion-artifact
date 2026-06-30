from pathlib import Path

ROLE_TO_PROMPT_FILE = {
    "planner": "task_prompt_planner_livecodebench",
    "implementer": "task_prompt_implementer_livecodebench",
    "reviewer": "task_prompt_reviewer_livecodebench",
}
REQUIRED_PERSONA_FILES = ("planner.md", "implementer.md", "reviewer.md")
SPECIAL_PERSONA_DIRS = {"baseline", "emotion_baseline", "persona_baseline"}
PERSONA_MARKERS = ("Persona description:", "Engineer description:", "Description:")


def _prompt_dir(prompt_dir):
    if prompt_dir is None:
        raise ValueError("prompt_dir is required")
    return Path(prompt_dir)


def persona_root(prompt_dir):
    return _prompt_dir(prompt_dir) / "actual_output_persona_description"


def task_prompt_root(prompt_dir):
    return _prompt_dir(prompt_dir) / "task_prompt"


def io_prompt_root(prompt_dir):
    return _prompt_dir(prompt_dir) / "io_prompt"


def _is_complete(d):
    return d.is_dir() and all((d / f).is_file() for f in REQUIRED_PERSONA_FILES)


def _team_dirs(parent):
    if not parent.is_dir():
        return []
    return sorted(
        p for p in parent.iterdir()
        if p.is_dir() and p.name.startswith("team") and _is_complete(p)
    )


def discover_persona_variants(prompt_dir):
    variants = []
    root = persona_root(prompt_dir)

    # Main: <emotion>/<team_dir>/<role>.md
    for emotion_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if emotion_dir.name in SPECIAL_PERSONA_DIRS:
            continue
        for team_dir in _team_dirs(emotion_dir):
            variants.append({
                "emotion": emotion_dir.name,
                "team_id": team_dir.name.split("_", 1)[0],
                "team_dir_name": team_dir.name,
                "baseline_type": "main",
            })

    # Persona-only: persona_baseline/<team_dir>/<role>.md
    for team_dir in _team_dirs(root / "persona_baseline"):
        variants.append({
            "emotion": "persona_baseline",
            "team_id": team_dir.name.split("_", 1)[0],
            "team_dir_name": team_dir.name,
            "baseline_type": "persona_only",
        })

    # Emotion-only: emotion_baseline/<emotion>/<role>.md
    eb_dir = root / "emotion_baseline"
    if eb_dir.is_dir():
        for emotion_dir in sorted(p for p in eb_dir.iterdir() if _is_complete(p)):
            variants.append({
                "emotion": emotion_dir.name,
                "team_id": "emotion_baseline",
                "team_dir_name": "emotion_baseline",
                "baseline_type": "emotion_only",
            })

    # Vanilla: baseline/<role>.md
    if _is_complete(root / "baseline"):
        variants.append({
            "emotion": "baseline",
            "team_id": "baseline",
            "team_dir_name": "baseline",
            "baseline_type": "vanilla",
        })

    return variants


def persona_output_path(team_dir_name, role, emotion, prompt_dir):
    filename = f"{role.lower()}.md"
    root = persona_root(prompt_dir)

    main_path = root / emotion / team_dir_name / filename
    if main_path.exists():
        return main_path
    if team_dir_name == "emotion_baseline":
        return root / "emotion_baseline" / emotion / filename
    if emotion == "baseline" and team_dir_name == "baseline":
        return root / "baseline" / filename
    return main_path


def task_prompt_path(role, prompt_dir):
    return task_prompt_root(prompt_dir) / ROLE_TO_PROMPT_FILE[role.lower()]


def load_persona(team_dir_name, role, emotion, prompt_dir):
    path = persona_output_path(team_dir_name, role, emotion, prompt_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing persona output file: {path}")
    text = path.read_text(encoding="utf-8")
    # Try markers most-specific first (avoid 'Description:' matching inside 'Persona description:')
    for marker in PERSONA_MARKERS:
        if marker in text:
            return text.split(marker, 1)[1].strip()
    raise ValueError(f"No known marker {PERSONA_MARKERS} in {path}")


def load_task_prompt(role, prompt_dir):
    path = task_prompt_path(role, prompt_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing task prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_io_prompt(testtype, prompt_dir):
    path = io_prompt_root(prompt_dir) / f"io_prompt_{testtype}"
    if not path.exists():
        raise FileNotFoundError(f"Missing I/O prompt file: {path}")
    return path.read_text(encoding="utf-8").strip()
