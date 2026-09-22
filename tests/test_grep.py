"""jev_grep: both modes, sorting, thresholds, and the survivor cut."""

from pathlib import Path

import pytest

from sieve.cache import NullCache
from sieve.errors import InvalidAnswerError
from sieve.grep import SURVIVOR_MULTIPLE, UNITS_PER_REQUEST, jev_grep

from .conftest import FakeChoiceAnswer, FakeClient, FakeResponse, FakeUsage


def build(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return tmp_path


def by_path(scores: dict[str, float], default: float = 0.05):
    """A fake scorer that reads the unit's path out of the shared state."""

    def score(state, index):
        unit = state["units"][index]
        return scores.get(unit["path"], default)

    return score


async def test_files_mode_ranks_and_applies_the_threshold(tmp_path):
    root = build(tmp_path, {"a.py": "x=1\n", "b.py": "y=2\n", "c.py": "z=3\n"})
    client = FakeClient(scorer=by_path({"a.py": 0.9, "b.py": 0.6, "c.py": 0.2}))
    out = await jev_grep("q", str(root), client=client, cache=NullCache(), threshold=0.5)
    assert [r["path"] for r in out["results"]] == ["a.py", "b.py"]
    assert out["results"][0] == {
        "path": "a.py",
        "line_start": 1,
        "line_end": 1,
        "kind": "file",
        "probability": 0.9,
    }
    assert out["units_scored"] == 3
    assert out["budget_exhausted"] is False
    assert out["cost_usd"] > 0


async def test_top_k_caps_the_result_list(tmp_path):
    root = build(tmp_path, {f"f{i}.py": "x=1\n" for i in range(10)})
    client = FakeClient(scorer=lambda state, i: 0.9)
    out = await jev_grep("q", str(root), top_k=3, client=client, cache=NullCache())
    assert len(out["results"]) == 3
    assert out["units_scored"] == 10


async def test_functions_mode_splits_only_the_surviving_files(tmp_path):
    files = {f"f{i}.py": f"def a{i}():\n    return {i}\n\ndef b{i}():\n    return {i}\n" for i in range(8)}
    root = build(tmp_path, files)
    strong = {f"f{i}.py": 0.9 for i in range(3)}
    client = FakeClient(scorer=by_path(strong, default=0.1))
    out = await jev_grep("q", str(root), mode="functions", top_k=2, client=client, cache=NullCache())
    # 3 files clear the threshold, under the cut of SURVIVOR_MULTIPLE * top_k = 4;
    # each splits into 2 functions.
    assert SURVIVOR_MULTIPLE * 2 == 4
    assert out["units_scored"] == 8 + 3 * 2
    assert all(r["kind"] == "function" for r in out["results"])
    assert {r["path"] for r in out["results"]} <= set(strong)
    assert len(out["results"]) == 2


async def test_functions_mode_with_no_survivors_returns_nothing(tmp_path):
    root = build(tmp_path, {"a.py": "def f():\n    return 1\n"})
    client = FakeClient(scorer=lambda state, i: 0.1)
    out = await jev_grep("q", str(root), mode="functions", threshold=0.5, client=client, cache=NullCache())
    assert out["results"] == []
    assert out["units_scored"] == 1


async def test_results_are_sorted_by_probability_then_path(tmp_path):
    root = build(tmp_path, {"b.py": "1\n", "a.py": "1\n", "c.py": "1\n"})
    client = FakeClient(scorer=by_path({"a.py": 0.8, "b.py": 0.8, "c.py": 0.9}))
    out = await jev_grep("q", str(root), client=client, cache=NullCache())
    assert [r["path"] for r in out["results"]] == ["c.py", "a.py", "b.py"]


async def test_an_unknown_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode must be"):
        await jev_grep("q", str(tmp_path), mode="lines", client=FakeClient(), cache=NullCache())


async def test_a_malformed_answer_aborts_the_call(tmp_path):
    root = build(tmp_path, {"a.py": "x=1\n"})

    class WrongTypeClient(FakeClient):
        async def system_one(self, state, questions, *, model=None, **kwargs):
            return FakeResponse(
                answers={qid: FakeChoiceAnswer("a", {"a": 0.4, "b": 0.4}) for qid in questions},
                usage=FakeUsage(10, 2),
            )

    with pytest.raises(InvalidAnswerError, match="expected a noul"):
        await jev_grep("q", str(root), client=WrongTypeClient(), cache=NullCache())


async def test_the_cache_makes_a_repeat_search_free(tmp_path):
    root = build(tmp_path / "src", {"a.py": "x=1\n", "b.py": "y=2\n"})
    from sieve.cache import AnswerCache

    cache = AnswerCache(tmp_path / "answers.sqlite3")
    client = FakeClient(scorer=by_path({"a.py": 0.9}, default=0.9))
    first = await jev_grep("q", str(root), client=client, cache=cache)
    second = await jev_grep("q", str(root), client=client, cache=cache)
    assert first["results"] == second["results"]
    assert second["cache_hits"] == 2
    assert second["requests"] == 0
    cache.close()


async def test_units_are_capped_per_request(tmp_path):
    root = build(tmp_path, {f"f{i}.py": "x=1\n" for i in range(11)})
    client = FakeClient(scorer=lambda state, i: 0.9)
    await jev_grep("q", str(root), client=client, cache=NullCache())
    assert [len(call.state["units"]) for call in client.calls] == [8, 3]
    assert UNITS_PER_REQUEST == 8


async def test_units_per_request_can_be_overridden(tmp_path):
    root = build(tmp_path, {f"f{i}.py": "x=1\n" for i in range(5)})
    client = FakeClient(scorer=lambda state, i: 0.9)
    await jev_grep("q", str(root), client=client, cache=NullCache(), units_per_request=2)
    assert [len(call.state["units"]) for call in client.calls] == [2, 2, 1]
