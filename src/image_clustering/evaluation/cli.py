"""CLI for appending reviewed cluster and crop labels."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .labels import append_crop, load_cases, make_case_id, upsert_case, validate_cases


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add reviewed clusters, non-clusters, and crop targets to JSONL."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    pair = commands.add_parser("pair", help="Add a reviewed image pair")
    pair.add_argument("label_file", type=Path)
    pair.add_argument("images", nargs=2)
    relation = pair.add_mutually_exclusive_group(required=True)
    relation.add_argument("--same", action="store_true")
    relation.add_argument("--near-duplicate", action="store_true")
    relation.add_argument("--occlusion", action="store_true")
    relation.add_argument("--different", action="store_true")
    pair.add_argument("--notes", default="")
    pair.add_argument("--source", default="user_review")
    pair.add_argument("--status", default="confirmed")
    pair.add_argument("--expected-min-submissions", type=int)

    cluster = commands.add_parser(
        "cluster", help="Add a reviewed same-document cluster"
    )
    cluster.add_argument("label_file", type=Path)
    cluster.add_argument("images", nargs="+")
    cluster.add_argument("--near-duplicate", action="store_true")
    cluster.add_argument("--notes", default="")
    cluster.add_argument("--source", default="user_review")
    cluster.add_argument("--status", default="confirmed")
    cluster.add_argument("--expected-min-submissions", type=int, default=1)

    crop = commands.add_parser("crop", help="Append one expected crop to a case")
    crop.add_argument("label_file", type=Path)
    crop.add_argument("case_id")
    crop.add_argument("image")
    crop.add_argument("bbox", nargs=4, type=float)
    crop.add_argument("--pixels", action="store_true")
    crop.add_argument("--kind", default="base_page")
    crop.add_argument("--completeness", default="complete")
    crop.add_argument("--side", default="single")
    crop.add_argument("--notes", default="")

    validate = commands.add_parser("validate", help="Validate a label file")
    validate.add_argument("label_file", type=Path)

    listing = commands.add_parser("list", help="Print a compact case summary")
    listing.add_argument("label_file", type=Path)
    return parser


def run(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        cases = load_cases(args.label_file)
        validate_cases(cases)
        print(f"Validated {len(cases)} cases in {args.label_file}")
        return 0
    if args.command == "list":
        for case in load_cases(args.label_file):
            print(
                f"{case['case_id']}\t{case['relation']}\t"
                f"{len(case.get('expected_submissions', []))} crops"
            )
        return 0
    if args.command == "crop":
        crop = {
            "image": args.image,
            "bbox": list(args.bbox),
            "coordinates": "pixels" if args.pixels else "normalized",
            "kind": args.kind,
            "completeness": args.completeness,
            "side": args.side,
            "notes": args.notes,
        }
        case = append_crop(args.label_file, args.case_id, crop)
        print(json.dumps(case, indent=2))
        return 0

    images = list(args.images)
    if args.command == "pair":
        if args.different:
            relation = "different_document"
        elif args.near_duplicate:
            relation = "same_document_near_duplicate"
        elif args.occlusion:
            relation = "same_document_occlusion"
        else:
            relation = "same_document"
        minimum = args.expected_min_submissions
        if minimum is None:
            minimum = 0 if relation == "different_document" else 1
    else:
        relation = (
            "same_document_near_duplicate"
            if args.near_duplicate
            else "same_document"
        )
        minimum = args.expected_min_submissions
    case = {
        "case_id": make_case_id(images),
        "images": images,
        "relation": relation,
        "expected_cluster": relation != "different_document",
        "expected_min_submissions": minimum,
        "expected_submissions": [],
        "notes": args.notes,
        "source": args.source,
        "status": args.status,
    }
    print(json.dumps(upsert_case(args.label_file, case), indent=2))
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
