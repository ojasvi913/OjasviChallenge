#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from helpers import add_candidate_features, read_candidate_chunks
from resumescore import DEFAULT_CHUNK_SIZE, DEFAULT_INPUT, DEFAULT_MIN_EXPERIENCE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLOT_OUTPUT = PROJECT_ROOT / "data" / "technical_score_distribution.png"
DEFAULT_CSV_OUTPUT = PROJECT_ROOT / "data" / "technical_scores_min_4_years.csv"


def collect_scores(
    input_path: Path,
    min_years: float,
    chunk_size: int,
) -> pd.DataFrame:
    score_frames = []

    for chunk in read_candidate_chunks(input_path, chunk_size):
        candidates = add_candidate_features(chunk)
        candidates = candidates[candidates["years_of_experience"] >= min_years]
        score_frames.append(
            candidates[
                [
                    "candidate_id",
                    "years_of_experience",
                    "title_matches",
                    "skill_matches",
                    "experience_matches",
                    "technical_score",
                ]
            ]
        )

    if not score_frames:
        return pd.DataFrame()

    return pd.concat(score_frames, ignore_index=True)


def plot_score_distribution(
    scores: pd.DataFrame,
    output_path: Path,
    min_years: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(scores["technical_score"], bins=40, color="#2563eb", edgecolor="white")
    plt.axvline(3, color="#dc2626", linestyle="--", linewidth=2, label="threshold = 3")
    plt.title(
        "Technical Score Distribution for Candidates With "
        f">= {min_years:g} Years Experience"
    )
    plt.xlabel("technical_score")
    plt.ylabel("candidate_count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot broad technical scores for candidates above the experience bar."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--plot-output", type=Path, default=DEFAULT_PLOT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--min-years", type=float, default=DEFAULT_MIN_EXPERIENCE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = collect_scores(args.input, args.min_years, args.chunk_size)

    if scores.empty:
        print("No candidates found for the selected filters.")
        return

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.csv_output, index=False)
    plot_score_distribution(scores, args.plot_output, args.min_years)

    print(f"Wrote score CSV: {args.csv_output}")
    print(f"Wrote score plot: {args.plot_output}")
    print(scores["technical_score"].describe())


if __name__ == "__main__":
    main()
