"""Runtime settings filled by the runner scripts."""

import os

MODEL = ""
BASE_URL = ""
API_KEY = os.environ.get("OPENAI_API_KEY", "")

TEMPERATURE = 0.0
MAX_TOKENS_REVIEWER = 600
MAX_TOKENS_JUDGE = 1000
MAX_REVISE_ROUNDS = 3

REVIEWER_CONCURRENCY = 2
