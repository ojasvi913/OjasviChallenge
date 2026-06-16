#!/usr/bin/env python3
"""
BM25 relevance scoring for filtered candidates.

Reads the filtered candidate pool (output of resumescore.py) and scores
each candidate against job-relevant BM25 queries. Outputs a scored JSONL
sorted by combined BM25 score, ready for the next pipeline stage.

Usage:
    python bm25_score.py [--input PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from rank_bm25 import BM25Okapi

from bm25_queries import BM25_QUERIES, BM25_QUERY_WEIGHTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "candidates_min_4_years.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "candidates_bm25_scored.jsonl"


# ---------------------------------------------------------------------------
# Document building
# ---------------------------------------------------------------------------

_NON_ALPHA = re.compile(r"[^a-z0-9\s]")


def _clean_text(value: object) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    if not isinstance(value, str):
        return ""
    return _NON_ALPHA.sub(" ", value.strip().lower())


def build_candidate_document(row: pd.Series) -> str:
    """Build a single text document per candidate for BM25 indexing.

    Concatenates career descriptions, career titles, and profile summary —
    the fields where candidates describe what they actually built.
    """
    parts: list[str] = []

    profile = row.get("profile")
    if isinstance(profile, dict):
        for key in ("summary", "headline"):
            text = _clean_text(profile.get(key))
            if text:
                parts.append(text)

    career = row.get("career_history")
    if isinstance(career, list):
        for role in career:
            if not isinstance(role, dict):
                continue
            for key in ("description", "title"):
                text = _clean_text(role.get(key))
                if text:
                    parts.append(text)

    return " ".join(parts)


def tokenize(text: str) -> list[str]:
    """Whitespace tokenization for BM25."""
    return text.split()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_candidates(
    candidates: pd.DataFrame,
    queries: dict[str, str],
    weights: dict[str, float],
) -> pd.DataFrame:
    """Score every candidate against each BM25 query.

    Adds one column per query (``bm25_{name}``) plus a combined
    ``bm25_score`` column.
    """
    result = candidates.copy()

    # Build and tokenize documents
    documents = result.apply(build_candidate_document, axis=1)
    tokenized_docs = documents.apply(tokenize).tolist()

    # Build BM25 index over the full corpus
    bm25 = BM25Okapi(tokenized_docs)

    # Score against each query
    combined = pd.Series(0.0, index=result.index)

    for query_name, query_text in queries.items():
        query_tokens = tokenize(query_text.lower())
        scores = bm25.get_scores(query_tokens)
        col = f"bm25_{query_name}"
        result[col] = scores
        combined += scores * weights.get(query_name, 1.0)

    result["bm25_score"] = combined
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score filtered candidates using BM25 against job-relevant queries.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Filtered JSONL from resumescore.py. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL with BM25 scores. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading candidates from {args.input} ...")
    candidates = pd.read_json(args.input, lines=True)
    print(f"Loaded {len(candidates):,} candidates.")

    print("Building BM25 index and scoring ...")
    scored = score_candidates(candidates, BM25_QUERIES, BM25_QUERY_WEIGHTS)

    # Sort by combined BM25 score descending
    scored = scored.sort_values("bm25_score", ascending=False).reset_index(drop=True)

    # Write output — use json.dumps per row for safe escaping
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for record in scored.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # Summary stats
    print(f"\nBM25 score distribution:")
    print(scored["bm25_score"].describe().to_string())

    query_cols = [f"bm25_{name}" for name in BM25_QUERIES]
    display_cols = ["candidate_id"] + query_cols + ["bm25_score"]

    print(f"\nTop 10 candidates by BM25 score:")
    print(scored.head(10)[display_cols].to_string(index=False))

    print(f"\nWrote {len(scored):,} scored candidates to {args.output}")


if __name__ == "__main__":
    main()
