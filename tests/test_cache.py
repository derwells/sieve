"""The answer cache keys on (model, question, unit text) and stops repeat calls."""

import json
from dataclasses import replace

from sieve.cache import IGNORE_PATTERN, AnswerCache, answer_key, cache_dir_for, ensure_ignored
from sieve.jev import JevScorer, QuestionSpec, ScoreItem

from .conftest import FakeClient

SPEC = QuestionSpec(
    state_field="units",
    instructions="Is {ref} relevant to `question`?",
    criteria_true="yes",
    criteria_false="no",
)


def item(index: int) -> ScoreItem:
    payload = {"path": f"f{index}.py", "content": f"body {index}"}
    return ScoreItem(id=str(index), text=json.dumps(payload, sort_keys=True), payload=payload)


def test_key_changes_with_model_question_and_text():
    base = answer_key("jev-latest", "q", "text")
    assert base == answer_key("jev-latest", "q", "text")
    assert base != answer_key("jev-1.13.0", "q", "text")
    assert base != answer_key("jev-latest", "other", "text")
    assert base != answer_key("jev-latest", "q", "other")


def test_key_changes_with_the_prompt_spec():
    base = answer_key("jev-latest", "q", "text", "spec")
    assert base == answer_key("jev-latest", "q", "text", "spec")
    assert base != answer_key("jev-latest", "q", "text", "another spec")


def test_key_is_not_confused_by_field_boundaries():
    assert answer_key("a", "b", "c") != answer_key("ab", "", "c")


def test_cache_round_trips(tmp_path):
    with AnswerCache(tmp_path / "answers.sqlite3") as cache:
        assert cache.get("k") is None
        cache.put("k", 0.42)
        assert cache.get("k") == 0.42
    with AnswerCache(tmp_path / "answers.sqlite3") as reopened:
        assert reopened.get("k") == 0.42


async def test_second_run_is_served_from_cache(tmp_path):
    client = FakeClient(scorer=lambda state, i: 0.6)
    cache = AnswerCache(tmp_path / "answers.sqlite3")
    items = [item(i) for i in range(3)]

    first = await JevScorer(client, cache=cache).score("q", items, SPEC)
    assert first.usage.cache_hits == 0
    assert len(client.calls) == 1

    second = await JevScorer(client, cache=cache).score("q", items, SPEC)
    assert second.usage.cache_hits == 3
    assert second.usage.requests == 0
    assert len(client.calls) == 1, "a full cache hit must not reach the API"
    assert second.scores == first.scores
    cache.close()


async def test_a_changed_question_misses_the_cache(tmp_path):
    client = FakeClient()
    cache = AnswerCache(tmp_path / "answers.sqlite3")
    items = [item(0)]
    await JevScorer(client, cache=cache).score("q", items, SPEC)
    await JevScorer(client, cache=cache).score("a different question", items, SPEC)
    assert len(client.calls) == 2
    cache.close()


async def test_a_changed_prompt_misses_the_cache(tmp_path):
    """Editing the instructions or criteria must not return answers to the old prompt."""
    client = FakeClient()
    cache = AnswerCache(tmp_path / "answers.sqlite3")
    items = [item(0)]
    await JevScorer(client, cache=cache).score("q", items, SPEC)
    sharper = replace(SPEC, criteria_true="yes, and it says how")
    await JevScorer(client, cache=cache).score("q", items, sharper)
    assert len(client.calls) == 2
    cache.close()


async def test_partial_hits_only_send_the_misses(tmp_path):
    client = FakeClient()
    cache = AnswerCache(tmp_path / "answers.sqlite3")
    await JevScorer(client, cache=cache).score("q", [item(0), item(1)], SPEC)
    await JevScorer(client, cache=cache).score("q", [item(0), item(1), item(2)], SPEC)
    assert len(client.calls) == 2
    assert [u["path"] for u in client.calls[1].state["units"]] == ["f2.py"]
    cache.close()


def test_cache_lives_in_a_repo_that_has_a_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    directory = cache_dir_for(tmp_path)
    assert directory == tmp_path.resolve() / ".sieve-cache"
    assert IGNORE_PATTERN in (tmp_path / ".gitignore").read_text(encoding="utf-8")


def test_an_existing_ignore_entry_is_not_duplicated(tmp_path):
    (tmp_path / ".gitignore").write_text(f"*.pyc\n{IGNORE_PATTERN}\n", encoding="utf-8")
    assert ensure_ignored(tmp_path) is True
    assert (tmp_path / ".gitignore").read_text(encoding="utf-8").count(".sieve-cache") == 1


def test_a_repo_without_a_gitignore_uses_the_user_cache(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr("sieve.cache.USER_CACHE_ROOT", home / ".cache" / "sieve")
    directory = cache_dir_for(tmp_path)
    assert home in directory.parents
    assert not (tmp_path / ".sieve-cache").exists()
    assert not (tmp_path / ".gitignore").exists()
