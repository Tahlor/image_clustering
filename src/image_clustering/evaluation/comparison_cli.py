"""Command-line interface for generating a static clustering comparison page."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .comparison import (
    compare_result_paths,
    default_comparison_output,
    write_comparison_page,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two saved clustering runs by exact image-membership sets "
            "and write an aggregate HTML review page."
        )
    )
    parser.add_argument(
        "--run-a",
        type=Path,
        required=True,
        help="First result root or path to its clustering.json manifest.",
    )
    parser.add_argument(
        "--run-b",
        type=Path,
        required=True,
        help="Second result root or path to its clustering.json manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Comparison HTML path; defaults to run-a/review/compare.html.",
    )
    parser.add_argument("--label-a", help="Display label for the first run.")
    parser.add_argument("--label-b", help="Display label for the second run.")
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Machine-readable summary path; defaults beside the HTML page.",
    )
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    comparison = compare_result_paths(
        args.run_a,
        args.run_b,
        left_label=args.label_a,
        right_label=args.label_b,
    )
    output_path = (
        args.output or default_comparison_output(args.run_a)
    ).expanduser().resolve()
    summary_path = (
        args.summary_json or output_path.with_suffix(".json")
    ).expanduser().resolve()
    write_comparison_page(comparison, output_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote comparison page: {output_path}")
    print(f"Wrote comparison summary: {summary_path}")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
