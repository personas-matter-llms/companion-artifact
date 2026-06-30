"""LiveCodeBench execution is not bundled in this artifact.

Use the official LiveCodeBench evaluator for testcase execution:
https://github.com/LiveCodeBench/LiveCodeBench
"""


def run_test(*args, **kwargs):
    raise RuntimeError(
        "LiveCodeBench testcase execution is not bundled in this artifact; "
        "please use the official LiveCodeBench evaluator."
    )
