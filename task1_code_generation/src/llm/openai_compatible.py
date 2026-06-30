import json
import time
import urllib.error
import urllib.request

from utils import LLMResponse

RETRY_STATUS = {429, 500, 502, 503, 504}


def _retry_delay(error):
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass
    return 3.0


def _provider_tag(url):
    if "dashscope" in url:
        return "aliyun-dashscope"
    if "api.openai.com" in url:
        return "openai"
    return "remote-openai-compatible"


def call(system, user, *, model, max_tokens, temperature=0.0, timeout=300,
         llm_url=None, api_key=None):
    """Send one OpenAI-compatible /v1/chat/completions request."""
    if not llm_url:
        raise ValueError("llm_url is required")
    url = llm_url.rstrip("/")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": False,
    }

    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )

    last_error = None
    for attempt in range(3):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            choice = (raw.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content") or "").strip()
            usage = raw.get("usage") or {}
            pt = int(usage.get("prompt_tokens", 0) or 0)
            ct = int(usage.get("completion_tokens", 0) or 0)
            tt = int(usage.get("total_tokens", pt + ct) or 0)
            return LLMResponse(
                text=text, model=model, provider=_provider_tag(url),
                metrics={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt},
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code in RETRY_STATUS and attempt < 2:
                time.sleep(_retry_delay(error))
                continue
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code}: {details}") from error
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(3.0)
                continue
            raise RuntimeError(f"request failed: {error}") from error
    raise RuntimeError(f"request failed: {last_error}")
