#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from filters import apply_filters, build_filters
from helpers import (
    add_candidate_features,
    prepare_output,
    read_candidate_chunks,
    write_jsonl_chunk,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "candidates.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "candidates_min_4_years.jsonl"
DEFAULT_MIN_EXPERIENCE = 4.0
DEFAULT_MIN_TECHNICAL_SCORE = 20
DEFAULT_CHUNK_SIZE = 10_000


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path = DEFAULT_INPUT
    output_path: Path = DEFAULT_OUTPUT
    min_years_experience: float = DEFAULT_MIN_EXPERIENCE
    min_technical_score: int = DEFAULT_MIN_TECHNICAL_SCORE
    chunk_size: int = DEFAULT_CHUNK_SIZE


def run_pipeline(config: PipelineConfig) -> int:
    """Run the candidate pipeline and return the number of written candidates."""
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    filters = build_filters(config.min_years_experience, config.min_technical_score)
    total_written = 0

    with config.output_path.open("w", encoding="utf-8") as output_file:
        for chunk in read_candidate_chunks(config.input_path, config.chunk_size):
            candidates = add_candidate_features(chunk)
            candidates = apply_filters(candidates, filters)
            candidates = prepare_output(candidates)

            if candidates.empty:
                continue

            write_jsonl_chunk(candidates, output_file)
            total_written += len(candidates)

    return total_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter and score candidates from the Redrob JSONL dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--min-years",
        type=float,
        default=DEFAULT_MIN_EXPERIENCE,
        help="Minimum years of experience to keep. Default: 4.0",
    )
    parser.add_argument(
        "--min-technical-score",
        type=int,
        default=DEFAULT_MIN_TECHNICAL_SCORE,
        help="Minimum broad technical score to keep. Default: 3",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Number of JSONL rows to process per pandas chunk. Default: 10000",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        input_path=args.input,
        output_path=args.output,
        min_years_experience=args.min_years,
        min_technical_score=args.min_technical_score,
        chunk_size=args.chunk_size,
    )


def main() -> None:
    config = config_from_args(parse_args())
    written = run_pipeline(config)
    print(
        f"Wrote {written} candidates with >= {config.min_years_experience:g} "
        f"years experience and technical_score >= {config.min_technical_score}."
    )
    print(f"Output: {config.output_path}")


if __name__ == "__main__":
    main()
