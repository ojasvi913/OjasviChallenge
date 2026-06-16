from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import pandas as pd

from technical_vocabulary import (
    TECHNICAL_SKILL_TERMS,
    TECHNICAL_TITLE_TERMS,
    TECHNICAL_WORK_TERMS,
)


TITLE_MATCH_WEIGHT = 5
SKILL_MATCH_WEIGHT = 3
EXPERIENCE_MATCH_WEIGHT = 1


def read_candidate_chunks(input_path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    return pd.read_json(input_path, lines=True, chunksize=chunk_size)


def get_nested_value(value: object, key: str) -> object:
    """Return value[key] when value is a dict, otherwise None."""
    if not isinstance(value, dict):
        return None
    return value.get(key)


def normalize_text(value: object) -> str:
    """Normalize arbitrary text-like values for vocabulary matching."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def count_term_matches(text: str, terms: list[str]) -> int:
    """Count unique vocabulary terms present in text."""
    matches = 0
    for term in terms:
        pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
        if re.search(pattern, text):
            matches += 1
    return matches


def join_text_values(values: Iterable[object]) -> str:
    """Join text-like values into one normalized search string."""
    return " ".join(normalize_text(value) for value in values if normalize_text(value))


def get_skill_names(skills: object) -> set[str]:
    """Return lowercase skill names from a candidate skills list."""
    if not isinstance(skills, list):
        return set()

    names = set()
    for skill in skills:
        if not isinstance(skill, dict):
            continue

        name = skill.get("name")
        if isinstance(name, str):
            names.add(name.strip().lower())

    return names


def get_career_values(career_history: object, key: str) -> list[object]:
    """Return values for one key across a candidate's career history."""
    if not isinstance(career_history, list):
        return []

    return [
        role.get(key)
        for role in career_history
        if isinstance(role, dict) and role.get(key) is not None
    ]


def add_numeric_profile_feature(
    candidates: pd.DataFrame,
    profile_key: str,
    column_name: str,
) -> pd.DataFrame:
    """Add a numeric column from a field inside the profile object."""
    result = candidates.copy()
    raw_values = result["profile"].apply(
        lambda profile: get_nested_value(profile, profile_key)
    )
    result[column_name] = pd.to_numeric(raw_values, errors="coerce")
    return result


def add_technical_score_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Add broad technical vocabulary match counts and total technical score."""
    result = candidates.copy()

    profile = result["profile"]
    title_text = profile.apply(
        lambda value: join_text_values(
            [
                get_nested_value(value, "headline"),
                get_nested_value(value, "current_title"),
            ]
        )
    )
    career_title_text = result["career_history"].apply(
        lambda value: join_text_values(get_career_values(value, "title"))
    )
    combined_title_text = title_text + " " + career_title_text

    skill_names = result["skills"].apply(get_skill_names)
    experience_text = profile.apply(
        lambda value: join_text_values(
            [
                get_nested_value(value, "summary"),
                get_nested_value(value, "headline"),
            ]
        )
    )
    career_description_text = result["career_history"].apply(
        lambda value: join_text_values(get_career_values(value, "description"))
    )
    combined_experience_text = experience_text + " " + career_description_text

    technical_skills = {term.lower() for term in TECHNICAL_SKILL_TERMS}
    result["title_matches"] = combined_title_text.apply(
        lambda text: count_term_matches(text, TECHNICAL_TITLE_TERMS)
    )
    result["skill_matches"] = skill_names.apply(
        lambda names: len(names.intersection(technical_skills))
    )
    result["experience_matches"] = combined_experience_text.apply(
        lambda text: count_term_matches(text, TECHNICAL_WORK_TERMS)
    )
    result["technical_score"] = (
        result["skill_matches"] * SKILL_MATCH_WEIGHT
        + result["title_matches"] * TITLE_MATCH_WEIGHT
        + result["experience_matches"] * EXPERIENCE_MATCH_WEIGHT
    )
    return result


def add_candidate_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Build columns used by filters and scorers.

    Add future feature builders here, for example:
    - skill counts
    - latest activity recency
    - AI/ML keyword indicators
    - education tier signals
    """
    candidates = add_numeric_profile_feature(
        candidates,
        profile_key="years_of_experience",
        column_name="years_of_experience",
    )
    candidates = add_technical_score_features(candidates)
    return candidates


def prepare_output(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Remove internal feature columns before writing candidate records.

    Keep this as the final output-shaping step so we can add temporary scoring
    columns freely without changing the source candidate JSON structure.
    """
    internal_columns = [
        "years_of_experience",
        "title_matches",
        "skill_matches",
        "experience_matches",
        "technical_score",
    ]
    return candidates.drop(columns=internal_columns, errors="ignore")


def write_jsonl_chunk(candidates: pd.DataFrame, output_file) -> None:
    """Append one candidate chunk to an already opened JSONL output file."""
    candidates.to_json(
        output_file,
        orient="records",
        lines=True,
        force_ascii=False,
    )
