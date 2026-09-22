"""Live citation check against this repository's README."""

import os
from pathlib import Path

import pytest

from sieve.cache import NullCache
from sieve.client import API_KEY_ENV
from sieve.verify import jev_verify

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get(API_KEY_ENV), reason=f"{API_KEY_ENV} is not set"),
]

README = str(Path(__file__).resolve().parent.parent / "README.md")
QUOTE = "Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/)."


async def test_readme_correct_and_changed_number():
    out = await jev_verify(records=[
        {"claim": "sieve requires Python 3.12 or later.", "citations": [{"locator": README, "quote": QUOTE}]},
        {"claim": "sieve requires Python 3.10 or later.", "citations": [{"locator": README, "quote": QUOTE}]},
    ], cache=NullCache())
    correct, changed = [row["verdict"] for row in out["results"]]
    print(f"\ncorrect: {correct}; changed: {changed}; usage: {out['usage']}")
    assert max(correct, key=correct.get) == "supports_fully"
    assert max(changed, key=changed.get) in {"contradicts", "does_not_address"}
