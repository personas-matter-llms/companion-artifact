import contextlib
import json
import multiprocessing
import os
import queue as queue_module
import threading

from .eval_testcase import run_test

_TESTCASE_EVAL_LOCK = threading.Lock()


def _lcb_payload(sample, testcase):
    """Build LCB-format payload for a single testcase. Returns (payload, error_message)."""
    if testcase.testtype == "stdin":
        fn_name = None
    elif testcase.testtype == "functional":
        fn_name = sample.fn_name
        if fn_name is None:
            return None, "Missing function name in dataset metadata."
    else:
        return None, f"Unsupported public test type: {testcase.testtype}"
    payload = {"inputs": [testcase.input], "outputs": [testcase.output], "fn_name": fn_name}
    return {"input_output": json.dumps(payload)}, ""


def _jsonable(value):
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def _worker(lcb_sample, solution_code, timeout_seconds, queue):
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                results, metadata = run_test(lcb_sample, test=solution_code, timeout=timeout_seconds)
        queue.put({"results": _jsonable(results), "metadata": _jsonable(metadata or {})})
    except BaseException as error:
        queue.put({"results": [-4], "metadata": {"error_message": str(error)}})


def _is_pass(value):
    return value is True


def run_one_testcase(solution_code, sample, testcase, timeout_seconds):
    payload, setup_error = _lcb_payload(sample, testcase)
    if payload is None:
        return False, {"error_message": setup_error}

    with _TESTCASE_EVAL_LOCK:
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = multiprocessing.get_context()

        queue = ctx.Queue()
        proc = ctx.Process(target=_worker, args=(payload, solution_code, timeout_seconds, queue))
        proc.start()
        proc.join(timeout_seconds + 5)
        if proc.is_alive():
            proc.kill()
            proc.join()
            return False, {"error_message": f"timeout after {timeout_seconds}s"}

        try:
            result = queue.get_nowait()
        except queue_module.Empty:
            return False, {"error_message": "empty test result"}

    results = result.get("results") or []
    metadata = result.get("metadata") or {}
    passed = bool(results) and _is_pass(results[0])
    return passed, metadata


def _failure_message(index, testcase, metadata):
    lines = [f"First failed public test: #{index}"]
    if "output" in metadata:
        lines += [
            "Input:", str(testcase.input),
            "Expected:", testcase.output.rstrip(),
            "Actual:", str(metadata["output"]).rstrip(),
        ]
        return "\n".join(lines)
    error = metadata.get("error_message") or metadata.get("error") or "Unknown public test failure"
    lines += ["Error message:", str(error)]
    return "\n".join(lines)


def get_public_tests_feedback(solution_code, sample, *, timeout_seconds=6):
    if not sample.public_tests:
        return "No public tests found."
    for i, tc in enumerate(sample.public_tests, start=1):
        passed, metadata = run_one_testcase(solution_code, sample, tc, timeout_seconds)
        if not passed:
            return _failure_message(i, tc, metadata)
    return "All public tests passed."
