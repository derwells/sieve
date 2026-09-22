"""Client-side validation of Jev answers.

Typed output guarantees the interface, not the contents. Every answer sieve acts
on goes through here first: a malformed distribution is rejected rather than
returned. See https://docs.typesafe.ai/api#answer-types for the wire shapes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .errors import InvalidAnswerError

#: How far the probabilities of one answer may sum away from 1 before rejection.
SUM_TOLERANCE = 0.02


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidAnswerError(message)


def validate_noul(qid: str, answer: Any) -> float:
    """Return the noul of `answer`, or raise if it is not a probability."""
    _require(getattr(answer, "type", None) == "noul", f"{qid}: expected a noul answer, got {answer!r}")
    value = answer.noul
    _require(isinstance(value, (int, float)), f"{qid}: noul is not a number: {value!r}")
    _require(math.isfinite(value) and 0.0 <= value <= 1.0, f"{qid}: noul {value} is outside [0, 1]")
    return float(value)


def _validate_distribution(qid: str, probabilities: Mapping[Any, float], offered: Sequence[Any]) -> None:
    missing = [option for option in offered if option not in probabilities]
    _require(not missing, f"{qid}: probabilities omit offered options {missing!r}")
    extra = [option for option in probabilities if option not in offered]
    _require(not extra, f"{qid}: probabilities include options that were not offered: {extra!r}")
    for option, value in probabilities.items():
        _require(isinstance(value, (int, float)), f"{qid}: probability for {option!r} is not a number: {value!r}")
        _require(math.isfinite(value) and 0.0 <= value <= 1.0, f"{qid}: probability for {option!r} is {value}, outside [0, 1]")
    total = sum(probabilities.values())
    _require(
        abs(total - 1.0) <= SUM_TOLERANCE,
        f"{qid}: probabilities sum to {total}, more than {SUM_TOLERANCE} away from 1",
    )


def validate_choice(qid: str, answer: Any, options: Sequence[str]) -> Any:
    """Check a Choice answer covers `options`, sums to ~1, and picks its maximum."""
    _require(getattr(answer, "type", None) == "choice", f"{qid}: expected a choice answer, got {answer!r}")
    probabilities = answer.probabilities
    _require(isinstance(probabilities, Mapping), f"{qid}: probabilities is not a map: {probabilities!r}")
    _validate_distribution(qid, probabilities, options)
    _require(answer.choice in probabilities, f"{qid}: chose {answer.choice!r}, which is not an offered option")
    best = max(probabilities.values())
    _require(
        probabilities[answer.choice] >= best,
        f"{qid}: chose {answer.choice!r} at {probabilities[answer.choice]}, "
        f"but another option scores {best}",
    )
    return answer


def validate_score(qid: str, answer: Any, levels: Sequence[Any]) -> Any:
    """Check a Score answer covers every level and sums to ~1."""
    _require(getattr(answer, "type", None) == "score", f"{qid}: expected a score answer, got {answer!r}")
    _validate_distribution(qid, answer.probabilities, list(range(len(levels))))
    return answer


def validate_response(response: Any, questions: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every answer to `questions` and return them keyed by question id."""
    answers = getattr(response, "answers", None)
    _require(isinstance(answers, Mapping), f"response carries no answers map: {response!r}")
    validated: dict[str, Any] = {}
    for qid, question in questions.items():
        _require(qid in answers, f"{qid}: no answer returned for this question")
        answer = answers[qid]
        kind = getattr(question, "type", None) or (question.get("type") if isinstance(question, Mapping) else None)
        if kind == "noul":
            validated[qid] = validate_noul(qid, answer)
        elif kind == "choice":
            criteria = question.criteria if hasattr(question, "criteria") else question["criteria"]
            validated[qid] = validate_choice(qid, answer, list(criteria))
        elif kind == "score":
            criteria = question.criteria if hasattr(question, "criteria") else question["criteria"]
            validated[qid] = validate_score(qid, answer, list(criteria))
        else:
            raise InvalidAnswerError(f"{qid}: unknown question type {kind!r}")
    return validated
