"""OpenAI-compatible chat client."""

import time

from . import config

_client = None


def get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
    return _client


def _metrics(response) -> dict:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def chat(system_prompt: str, user_message: str, max_tokens: int) -> dict:
    """Send one chat-completion request."""
    client = get_client()
    kwargs = {
        "model": config.MODEL,
        "temperature": config.TEMPERATURE,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }

    started = time.monotonic()
    response = client.chat.completions.create(**kwargs)
    duration_ms = int((time.monotonic() - started) * 1000)
    return {
        "text": response.choices[0].message.content or "",
        "model": getattr(response, "model", None) or config.MODEL,
        "system_prompt": system_prompt,
        "user_prompt": user_message,
        "metrics": _metrics(response),
        "duration_ms": duration_ms,
    }
