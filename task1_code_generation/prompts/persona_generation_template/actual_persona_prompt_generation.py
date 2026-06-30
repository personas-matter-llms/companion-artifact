import argparse
import re
from pathlib import Path

TEMPLATE_ROOT = Path(__file__).resolve().parent
PROMPTS_ROOT = TEMPLATE_ROOT.parent
PERSONA_PROMPT_PATH = TEMPLATE_ROOT / "persona_prompt" / "persona_prompt"
PERSONA_PROMPT_DIR = TEMPLATE_ROOT / "persona_prompt"
TASK_PROMPT_DIR = PROMPTS_ROOT / "task_prompt"
OUTPUT_DIR = PROMPTS_ROOT / "actual_input_persona_description"
DEFAULT_EMOTIONS = [
    "neutral",
    "anger",
    "fear",
    "disgust",
    "sadness",
    "happiness",
]

ROLE_TO_TASK_TYPE = {
    "Planner": "Code Planning",
    "Implementer": "Code Generation",
    "Reviewer": "Code Review",
}
ROLE_TO_TASK_FILE = {
    "Planner": "task_prompt_planner_livecodebench",
    "Implementer": "task_prompt_implementer_livecodebench",
    "Reviewer": "task_prompt_reviewer_livecodebench",
}
NEUTRAL_TRAIT_NOTE = (
    "- Treat NEUTRAL trait values as a balanced state with no strong tendency in that dimension."
)


def _extract_task_desc(text):
    match = re.search(
        r"^Task description:\s*\n(?P<body>.*?)(?:^Output format:\s*$|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError("Could not find 'Task description:' block in task prompt")
    body = match.group("body").rstrip()
    return f"Task description:\n{body}".strip()


def parse_team(md_path):
    text = md_path.read_text()
    sections = re.split(r"^## ", text, flags=re.MULTILINE)[1:]
    out = {}
    for section in sections:
        head, body = section.split("\n", 1)
        role = head.strip()
        out[role] = {
            "Conscientiousness": re.search(r"Conscientiousness:\s*(HIGH|LOW|NEUTRAL)", body).group(1),
            "Openness": re.search(r"Openness:\s*(HIGH|LOW|NEUTRAL)", body).group(1),
            "Extraversion": re.search(r"Extraversion:\s*(HIGH|LOW|NEUTRAL)", body).group(1),
            "Emotion": re.search(r"Emotion:\s*(\S+)", body).group(1),
        }
    return out


def team_id(md_path):
    # "01_team.md" -> "team01"
    m = re.match(r"(\d+)_team", md_path.stem)
    if not m:
        raise ValueError(f"unexpected team filename: {md_path.name}")
    return f"team{m.group(1)}"


def trait_code(profile):
    code_map = {"HIGH": "H", "LOW": "L", "NEUTRAL": "N"}
    return "".join(
        code_map[profile[key]]
        for key in ("Conscientiousness", "Openness", "Extraversion")
    )


def emotion_dir_name(emotion):
    return (emotion or "unknown").strip().lower()


def team_dir_name(team_id, profile):
    return f"{team_id}_{trait_code(profile)}"


def add_neutral_trait_note(text, cons, openness, extra):
    if "NEUTRAL" not in (cons, openness, extra):
        return text
    if NEUTRAL_TRAIT_NOTE in text:
        return text
    return text.replace(
        "- Tone: clear, natural, professional, realistic, and behaviorally distinctive.\n\n"
        "Personality trait interpretation:",
        "- Tone: clear, natural, professional, realistic, and behaviorally distinctive.\n"
        f"{NEUTRAL_TRAIT_NOTE}\n\n"
        "Personality trait interpretation:",
    )


def fill_prompt(template, role, task_type, task_desc, cons, openness, extra, emotion):
    filled = (
        template
        .replace("[ROLE]", role)
        .replace("[TARGET TASK TYPE]", task_type)
        .replace("{ROLE}", role)
        .replace("{TASK_TYPE}", task_type)
        .replace("{TASK_DESC}", task_desc.strip())
        .replace("- Conscientiousness: {HIGH/LOW}", f"- Conscientiousness: {cons}")
        .replace("- Openness: {HIGH/LOW}", f"- Openness: {openness}")
        .replace("- Extraversion: {HIGH/LOW}", f"- Extraversion: {extra}")
        .replace("{EMOTION}", emotion)
    )
    return add_neutral_trait_note(filled, cons=cons, openness=openness, extra=extra)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=DEFAULT_EMOTIONS,
        help="Emotion folders to generate. Default: all supported emotions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    template = PERSONA_PROMPT_PATH.read_text()
    task_descs = {
        role: _extract_task_desc((TASK_PROMPT_DIR / fname).read_text())
        for role, fname in ROLE_TO_TASK_FILE.items()
    }
    team_files = sorted(PERSONA_PROMPT_DIR.glob("*_team.md"))
    if not team_files:
        raise FileNotFoundError(f"no *_team.md files found under {PERSONA_PROMPT_DIR}")

    for team_path in team_files:
        tid = team_id(team_path)
        team = parse_team(team_path)
        for role, profile in team.items():
            if role not in ROLE_TO_TASK_TYPE:
                continue
            for emotion in args.emotions:
                filled = fill_prompt(
                    template,
                    role=role,
                    task_type=ROLE_TO_TASK_TYPE[role],
                    task_desc=task_descs[role],
                    cons=profile["Conscientiousness"],
                    openness=profile["Openness"],
                    extra=profile["Extraversion"],
                    emotion=emotion,
                )
                out_dir = OUTPUT_DIR / emotion_dir_name(emotion) / team_dir_name(tid, profile)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{role.lower()}.md"
                if out_path.exists():
                    print(f"skip {out_path}")
                    continue
                out_path.write_text(filled)
                print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
