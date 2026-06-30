import base64
import glob
import json
import pickle
import zlib
from pathlib import Path

from utils import LCBSample, LCBTestCase


def _resolve_jsonl(path_or_glob):
    raw = str(path_or_glob)
    p = Path(raw)
    if any(ch in raw for ch in "*?[]"):
        return [Path(m) for m in sorted(glob.glob(raw))]
    if p.is_dir():
        return sorted(p.glob("*.jsonl"))
    if p.exists():
        return [p]
    raise FileNotFoundError(f"Could not resolve dataset path: {path_or_glob}")


def _to_testcase(item):
    return LCBTestCase(
        input=str(item.get("input", "")),
        output=str(item.get("output", "")),
        testtype=str(item.get("testtype", "")),
    )


def _decode_public_tests(raw):
    return [_to_testcase(it) for it in json.loads(raw or "[]")]


def _decode_private_tests(raw):
    raw = raw.strip()
    if not raw or raw == "[]":
        return []
    inflated = zlib.decompress(base64.b64decode(raw))
    return [_to_testcase(it) for it in json.loads(pickle.loads(inflated))]


def _format_test(test, idx):
    label = "Input arguments" if test.testtype == "functional" else "Standard input"
    return f"Public test {idx}:\n{label}:\n{test.input}\nExpected output:\n{test.output}"


def _build_prompt_text(item, public_tests):
    parts = []
    title = str(item.get("question_title", "") or "").strip()
    if title:
        parts += ["Question title:", title]
    content = str(item.get("question_content", "") or "").strip()
    if content:
        parts += ["Problem statement:", content]
    starter = str(item.get("starter_code", "") or "").rstrip()
    if starter:
        parts += ["Starter code:", f"```python\n{starter}\n```"]
    if public_tests:
        parts.append("Public tests:")
        for i, t in enumerate(public_tests, start=1):
            parts.append(_format_test(t, i))
    return "\n\n".join(p for p in parts if p.strip()).strip()


def _build_sample(item):
    public_tests = _decode_public_tests(str(item.get("public_test_cases", "") or "[]"))
    private_tests = _decode_private_tests(str(item.get("private_test_cases", "") or "[]"))
    metadata = json.loads(str(item.get("metadata", "") or "{}"))
    qid = str(item.get("question_id", "") or "") or "unknown"
    return LCBSample(
        question_id=qid,
        question_title=str(item.get("question_title", "") or ""),
        difficulty=str(item.get("difficulty", "") or ""),
        prompt_text=_build_prompt_text(item, public_tests),
        public_tests=public_tests,
        private_tests=private_tests,
        starter_code=str(item.get("starter_code", "") or ""),
        fn_name=metadata.get("func_name") or None,
    )


def load_samples(jsonl_path, *, difficulty=None, limit=None):
    allowed = None if difficulty in (None, "", "all") else str(difficulty).lower()
    emitted = 0
    for path in _resolve_jsonl(jsonl_path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if limit is not None and emitted >= limit:
                    return
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if allowed and str(item.get("difficulty", "")).lower() != allowed:
                    continue
                yield _build_sample(item)
                emitted += 1
