from __future__ import annotations

from typing import Callable

import pandas as pd


Filter = Callable[[pd.DataFrame], pd.Series]


def minimum_experience_filter(min_years: float) -> Filter:
    """Create a filter that keeps candidates with at least min_years experience."""

    def filter_fn(candidates: pd.DataFrame) -> pd.Series:
        return candidates["years_of_experience"] >= min_years

    return filter_fn


def minimum_technical_score_filter(min_score: int) -> Filter:
    """Create a filter that keeps candidates above a technical score threshold."""

    def filter_fn(candidates: pd.DataFrame) -> pd.Series:
        return candidates["technical_score"] >= min_score

    return filter_fn


def apply_filters(candidates: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    """Apply all configured filters to a candidate chunk."""
    if not filters:
        return candidates

    keep_mask = pd.Series(True, index=candidates.index)
    for filter_fn in filters:
        keep_mask &= filter_fn(candidates)

    return candidates[keep_mask]


def build_filters(min_years_experience: float, min_technical_score: int) -> list[Filter]:
    """Configure all active candidate filters in one place."""
    return [
        minimum_experience_filter(min_years_experience),
        minimum_technical_score_filter(min_technical_score),
    ]
