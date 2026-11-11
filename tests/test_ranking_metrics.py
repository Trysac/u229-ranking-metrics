"""Behavior tests for the public API of ranking_metrics."""

from __future__ import annotations

import pytest

import ranking_metrics


def test_normalize_trims_and_lowercases() -> None:
    assert ranking_metrics.normalize("  Hello World ") == "hello world"


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_normalize_rejects_blank_input(value: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        ranking_metrics.normalize(value)


def test_version_is_semver() -> None:
    parts = ranking_metrics.__version__.split(".")
    assert len(parts) == 3
    assert all(part.isdigit() for part in parts)
