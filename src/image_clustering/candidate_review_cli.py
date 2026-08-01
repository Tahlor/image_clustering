"""CLI for sequence-aware occlusion candidate reports."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from image_clustering.clustering.candidate_review import rank_occlusion_candidates
from image_clustering.clustering.serialization import load_result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            encoded = dict(row)
            encoded["common_accepted_neighbors"] = json.dumps(
                encoded["common_accepted_neighbors"],
                separators=(",", ":"),
            )
            writer.writerow(encoded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank possible same-document occlusions for review.",
    )
    parser.add_argument("--clustering_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--include_accepted", action="store_true")
    parser.add_argument("--include_unflagged", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = load_result(args.clustering_json)
    candidates = rank_occlusion_candidates(
        result,
        include_accepted=args.include_accepted,
        include_unflagged=args.include_unflagged,
    )
    rows = [candidate.to_dict() for candidate in candidates]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "occlusion_candidates.csv", rows)
    (args.output_dir / "occlusion_candidates.jsonl").write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    tiers = Counter(candidate.review_tier for candidate in candidates)
    summary = {
        "candidate_count": len(candidates),
        "count_by_review_tier": {
            str(key): value for key, value in sorted(tiers.items())
        },
        "hard_contradiction_count": sum(
            candidate.hard_contradiction for candidate in candidates
        ),
        "common_neighbor_support_count": sum(
            bool(candidate.common_accepted_neighbors) for candidate in candidates
        ),
        "same_component_count": sum(
            candidate.same_component for candidate in candidates
        ),
        "fallback_registration_count": sum(
            candidate.registration_fallback_used for candidate in candidates
        ),
    }
    (args.output_dir / "occlusion_candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
