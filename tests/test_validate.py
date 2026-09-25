"""Answer validation rejects anything sieve must not act on."""

import pytest
from typesafe_sdk import Choice, Noul, Score

from sieve.errors import InvalidAnswerError
from sieve.validate import validate_choice, validate_noul, validate_response, validate_score

from .conftest import FakeChoiceAnswer, FakeNoulAnswer, FakeResponse, FakeScoreAnswer, FakeUsage


def test_noul_in_range_passes():
    assert validate_noul("q0", FakeNoulAnswer(0.73)) == 0.73


@pytest.mark.parametrize("value", [-0.01, 1.01, 42.0])
def test_noul_outside_range_rejected(value):
    with pytest.raises(InvalidAnswerError, match="outside"):
        validate_noul("q0", FakeNoulAnswer(value))


def test_noul_of_wrong_type_rejected():
    with pytest.raises(InvalidAnswerError, match="expected a noul"):
        validate_noul("q0", FakeChoiceAnswer("a", {"a": 1.0}))


def test_choice_covering_its_options_passes():
    answer = FakeChoiceAnswer("a", {"a": 0.7, "b": 0.3})
    assert validate_choice("q0", answer, ["a", "b"]) is answer


def test_choice_missing_an_offered_option_rejected():
    with pytest.raises(InvalidAnswerError, match="omit offered options"):
        validate_choice("q0", FakeChoiceAnswer("a", {"a": 1.0}), ["a", "b"])


def test_choice_with_an_unoffered_option_rejected():
    with pytest.raises(InvalidAnswerError, match="not offered"):
        validate_choice("q0", FakeChoiceAnswer("a", {"a": 0.5, "b": 0.3, "c": 0.2}), ["a", "b"])


def test_choice_probabilities_not_summing_to_one_rejected():
    with pytest.raises(InvalidAnswerError, match="sum to"):
        validate_choice("q0", FakeChoiceAnswer("a", {"a": 0.5, "b": 0.3}), ["a", "b"])


def test_choice_sum_inside_tolerance_passes():
    validate_choice("q0", FakeChoiceAnswer("a", {"a": 0.6, "b": 0.385}), ["a", "b"])


def test_choice_not_picking_the_maximum_rejected():
    with pytest.raises(InvalidAnswerError, match="but another option scores"):
        validate_choice("q0", FakeChoiceAnswer("b", {"a": 0.8, "b": 0.2}), ["a", "b"])


def test_choice_one_reporting_unit_below_the_maximum_passes():
    """Recorded live: the API picked 0.39 while reporting another option at 0.4."""
    probabilities = {"partially_supports": 0.4, "does_not_address": 0.39, "supports_fully": 0.13, "contradicts": 0.08}
    answer = FakeChoiceAnswer("does_not_address", probabilities)
    assert validate_choice("q0", answer, list(probabilities)).choice == "does_not_address"


def test_choice_float_noise_at_a_tie_passes():
    probabilities = {"a": 0.39999999999999997, "b": 0.4, "c": 0.2}
    validate_choice("q0", FakeChoiceAnswer("a", probabilities), ["a", "b", "c"])


@pytest.mark.parametrize("picked", [0.38, 0.37])
def test_choice_more_than_one_unit_below_the_maximum_rejected(picked):
    probabilities = {"a": picked, "b": 0.4, "c": round(1 - 0.4 - picked, 2)}
    with pytest.raises(InvalidAnswerError, match="another option scores 0.4"):
        validate_choice("q0", FakeChoiceAnswer("a", probabilities), ["a", "b", "c"])


def test_score_levels_must_all_be_present():
    good = FakeScoreAnswer(1.0, {0: 0.2, 1: 0.5, 2: 0.3})
    assert validate_score("q0", good, ["low", "mid", "high"]) is good
    with pytest.raises(InvalidAnswerError, match="omit offered options"):
        validate_score("q0", FakeScoreAnswer(1.0, {0: 0.5, 1: 0.5}), ["low", "mid", "high"])


def test_validate_response_dispatches_by_question_type():
    questions = {
        "n": Noul(instructions="yes?"),
        "c": Choice(instructions="which?", criteria={"a": None, "b": None}),
        "s": Score(instructions="how much?", criteria=["low", "high"]),
    }
    response = FakeResponse(
        answers={
            "n": FakeNoulAnswer(0.4),
            "c": FakeChoiceAnswer("a", {"a": 0.9, "b": 0.1}),
            "s": FakeScoreAnswer(0.5, {0: 0.5, 1: 0.5}),
        },
        usage=FakeUsage(10, 2),
    )
    answers = validate_response(response, questions)
    assert answers["n"] == 0.4
    assert answers["c"].choice == "a"


def test_validate_response_reads_the_real_sdk_answer_objects():
    """The fakes above mimic the wire shape; this pins the attribute names sieve reads."""
    from typesafe_sdk._core.response_types import (
        ChoiceAnswer,
        NoulAnswer,
        ScoreAnswer,
        SystemOneResponse,
        Usage,
    )

    questions = {
        "n": Noul(instructions="is it spam?"),
        "c": Choice(instructions="mood?", criteria={"angry": None, "calm": None}),
        "s": Score(instructions="how bad?", criteria=["low", "mid", "high"]),
    }
    response = SystemOneResponse(
        model="jev-1.13.0",
        usage=Usage(input_tokens=360, output_tokens=39),
        answers={
            "n": NoulAnswer(type="noul", noul=0.99),
            "c": ChoiceAnswer(type="choice", choice="angry", confidence=0.9, probabilities={"angry": 0.8, "calm": 0.2}),
            "s": ScoreAnswer(
                type="score",
                score=1.2,
                confidence=0.7,
                legend={0: "low", 1: "mid", 2: "high"},
                probabilities={0: 0.1, 1: 0.6, 2: 0.3},
            ),
        },
    )
    answers = validate_response(response, questions)
    assert answers["n"] == 0.99
    assert answers["c"].choice == "angry"
    assert answers["s"].probabilities == {0: 0.1, 1: 0.6, 2: 0.3}

    not_the_maximum = SystemOneResponse(
        model="jev-1.13.0",
        usage=Usage(),
        answers={"c": ChoiceAnswer(type="choice", choice="calm", confidence=0.9, probabilities={"angry": 0.8, "calm": 0.2})},
    )
    with pytest.raises(InvalidAnswerError, match="but another option scores"):
        validate_response(not_the_maximum, {"c": questions["c"]})


def test_validate_response_rejects_a_missing_answer():
    questions = {"n": Noul(instructions="yes?"), "m": Noul(instructions="also?")}
    response = FakeResponse(answers={"n": FakeNoulAnswer(0.4)}, usage=FakeUsage(10, 2))
    with pytest.raises(InvalidAnswerError, match="no answer returned"):
        validate_response(response, questions)
