import argparse
import json
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PROMPT = """You are helping with a code generation task. What is your default emotion and personality?

Emotion: pick exactly one from {anger, disgust, fear, happiness, sadness, neutral}.
Personality (Big-Five subset): for each of Conscientiousness, Openness, Extraversion, pick exactly one from {HIGH, LOW, neutral}.

Reply strictly in this format:
Emotion: ...
Conscientiousness: ...
Openness: ...
Extraversion: ...
Reason (one sentence): ..."""


POST_PROMPT = """You are helping with a code generation task. First, provide your Python solution. Then, on a new section, answer the following question.

[I/O Mode]
This problem is evaluated via standard input/output.
Read all input from stdin and write only the required output to stdout.
Do not hardcode sample inputs.
Do not print demo tests or extra explanatory output.

[Question]
Question title:

A. Short Sort

Problem statement:

There are three cards with letters a, b, c placed in a row in some order. You can do the following operation at most once:

- Pick two cards, and swap them.

Is it possible that the row becomes abc after the operation? Output "YES" if it is possible, and "NO" otherwise.

Input

The first line contains a single integer t (1 <= t <= 6) -- the number of test cases.

The only line of each test case contains a single string consisting of each of the three characters a, b, and c exactly once, representing the cards.

Output

For each test case, output "YES" if you can make the row abc with at most one operation, or "NO" otherwise.

You can output the answer in any case.

Sample Input 1:
6
abc
acb
bac
bca
cab
cba

Sample Output 1:
YES
YES
YES
NO
NO
YES

Public tests:

Public test 1:
Standard input:
6
abc
acb
bac
bca
cab
cba

Expected output:
YES
YES
YES
NO
NO
YES

After finishing this task and output the code, what is your emotion and personality?

Emotion: pick exactly one from {anger, disgust, fear, happiness, sadness, neutral}.
Personality (Big-Five subset): for each of Conscientiousness, Openness, Extraversion, pick exactly one from {HIGH, LOW, neutral}.

Reply strictly in this format:
Code: ...
Emotion: ...
Conscientiousness: ...
Openness: ...
Extraversion: ...
Reason (one sentence): ..."""


def chat_url(url):
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def call(url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        chat_url(url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read())
    return data["choices"][0]["message"]["content"]


def safe_name(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "model"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("llm_url")
    parser.add_argument("model")
    parser.add_argument("output_dir", nargs="?", default=str(ROOT / "logs" / "self_report"))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = safe_name(args.model)

    default_report = call(args.llm_url, args.model, DEFAULT_PROMPT, 300)
    post_report = call(args.llm_url, args.model, POST_PROMPT, 2000)

    default_path = out_dir / f"default_{model_name}_{stamp}.txt"
    post_path = out_dir / f"post_{model_name}_{stamp}.txt"

    default_path.write_text(default_report + "\n", encoding="utf-8")
    post_path.write_text(post_report + "\n", encoding="utf-8")

    print("=== DEFAULT SELF-REPORT ===")
    print(default_report)
    print(f"[saved] {default_path}")
    print()
    print("=== POST-TASK SELF-REPORT ===")
    print(post_report)
    print(f"[saved] {post_path}")


if __name__ == "__main__":
    main()
