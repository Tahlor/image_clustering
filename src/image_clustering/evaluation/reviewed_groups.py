"""Grouped evaluation for manually reviewed same-document proposals.

The evaluator treats ``original_cluster_id`` as the indivisible truth unit. A
review-oriented probability can rank a pair, but only explicit automatic-edge
fields participate in component construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from image_clustering.evaluation.reviewed_calibration import (
    apply_isotonic,
    compare_runs,
    fit_real_calibration,
)
from image_clustering.evaluation.reviewed_evaluate import evaluate_predictions
from image_clustering.evaluation.reviewed_models import (
    ALLOWED_SUBTYPES,
    CONTRACT,
    SCHEMA_VERSION,
    SPLIT_VERSION,
    DatasetContract,
    ManifestRow,
    PairTruth,
    SplitAssignment,
    expand_pairs,
    group_rows,
    load_csv_manifest,
    load_jsonl_manifest,
    pair_id,
)
from image_clustering.evaluation.reviewed_prepare import (
    load_subtypes,
    prepare_dataset,
    stage_review_images,
)
from image_clustering.evaluation.reviewed_split import make_grouped_splits
from image_clustering.evaluation.reviewed_validate import validate_manifest

__all__ = [
    "ALLOWED_SUBTYPES",
    "CONTRACT",
    "SCHEMA_VERSION",
    "SPLIT_VERSION",
    "DatasetContract",
    "ManifestRow",
    "PairTruth",
    "SplitAssignment",
    "apply_isotonic",
    "compare_runs",
    "evaluate_predictions",
    "expand_pairs",
    "fit_real_calibration",
    "group_rows",
    "load_csv_manifest",
    "load_jsonl_manifest",
    "load_subtypes",
    "make_grouped_splits",
    "pair_id",
    "prepare_dataset",
    "stage_review_images",
    "validate_manifest",
]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--assignments-csv", type=Path, required=True)
    prepare.add_argument("--assignments-jsonl", type=Path)
    prepare.add_argument("--package-root", type=Path)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--split-version", default=SPLIT_VERSION)
    prepare.add_argument("--stage-root", type=Path)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--prepared-dir", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--subtypes", type=Path)
    evaluate.add_argument("--calibrator", type=Path)
    evaluate.add_argument("--run-label", default="evaluation")

    calibrate = subparsers.add_parser("fit-calibration")
    calibrate.add_argument("--prepared-dir", type=Path, required=True)
    calibrate.add_argument("--pair-results", type=Path, required=True)
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--subtypes", type=Path)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--before", type=Path, required=True)
    compare.add_argument("--after", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        output = prepare_dataset(
            args.assignments_csv,
            args.output_dir,
            jsonl_path=args.assignments_jsonl,
            package_root=args.package_root,
            split_version=args.split_version,
            stage_root=args.stage_root,
        )
    elif args.command == "evaluate":
        output = evaluate_predictions(
            args.prepared_dir,
            args.predictions,
            args.output_dir,
            subtype_path=args.subtypes,
            calibrator_path=args.calibrator,
            run_label=args.run_label,
        )
    elif args.command == "fit-calibration":
        output = fit_real_calibration(
            args.prepared_dir,
            args.pair_results,
            args.output,
            subtype_path=args.subtypes,
        )
    else:
        output = compare_runs(args.before, args.after, args.output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
