"""TypeSafe client construction."""

from __future__ import annotations

import os

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from .errors import MissingAPIKeyError

API_KEY_ENV = "TYPESAFE_API_KEY"
DEFAULT_MODEL = "jev-latest"
#: A 30k-token batch takes longer than the SDK's 10s default.
REQUEST_TIMEOUT_SECONDS = 90.0

#: 408 and 429 are the documented transient client statuses; 5xx covers 529 Overloaded.
RETRY_POLICY = RetryPolicy(
    max_retries=4,
    backoff_initial=0.5,
    backoff_max=8.0,
    http_statuses={408, 429, *range(500, 600)},
    respect_retry_after=True,
    timeout=300.0,
)


def build_client(api_key: str | None = None, model: str = DEFAULT_MODEL) -> AsyncTypeSafeClient:
    """An async TypeSafe client, or a clear error when the key is missing."""
    key = api_key or os.environ.get(API_KEY_ENV)
    if not key:
        raise MissingAPIKeyError(
            f"{API_KEY_ENV} is not set. sieve reads it from the environment; the "
            f"bin/sieve-mcp launcher also sources $SIEVE_ENV_FILE, which defaults "
            f"to ~/.config/sieve/env. Set it and retry."
        )
    return AsyncTypeSafeClient(
        api_key=key,
        model=model,
        retry=RETRY_POLICY,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
