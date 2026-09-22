"""Typed errors raised by sieve."""

from __future__ import annotations


class SieveError(Exception):
    """Base class for every error sieve raises on purpose."""


class MissingAPIKeyError(SieveError):
    """TYPESAFE_API_KEY is not set in the environment."""


class InvalidAnswerError(SieveError):
    """A Jev answer failed client-side validation and must not be used.

    Raised when an answer is missing, has the wrong type, offers a probability
    distribution that does not cover the options that were asked about, does not
    sum to 1, or names a pick that is not the maximum-probability option.
    """


class EnumerationError(SieveError):
    """The target path could not be enumerated."""
