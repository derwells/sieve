"""Live route selection over a small synthetic set."""

import os

import pytest

from sieve.cache import NullCache
from sieve.client import API_KEY_ENV
from sieve.route import jev_route

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get(API_KEY_ENV), reason=f"{API_KEY_ENV} is not set"),
]

ROUTES = [
    {"id": "billing", "description": "Invoices, payments, and account charges", "aliases": ["payments"]},
    {"id": "deploy", "description": "Release and deploy software services", "aliases": ["release"]},
    {"id": "docs", "description": "Write or update product documentation", "aliases": ["guides"]},
    {"id": "data-pipeline", "description": "Move and transform data in scheduled pipelines", "aliases": ["ETL"]},
    {"id": "mobile-app", "description": "Build and debug mobile applications", "aliases": ["iOS", "Android"]},
    {"id": "security", "description": "Investigate vulnerabilities and access controls", "aliases": ["infosec"]},
]

ASKS = [
    ("I was charged twice on my invoice. Please fix it.", "billing"),
    ("Roll out the new service version to staging.", "deploy"),
    ("Update the getting started guide for new users.", "docs"),
    ("The nightly ETL job stopped loading warehouse rows.", "data-pipeline"),
    ("What's the weather in Lisbon?", "none"),
]


async def test_live_route_choices():
    for ask, expected in ASKS:
        out = await jev_route(ask, ROUTES, cache=NullCache())
        print(f"\n{ask!r}: {out['probabilities']}")
        assert out["choice"] == expected
        assert out["probabilities"][expected] == max(out["probabilities"].values())
