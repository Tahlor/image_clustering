"""Prepare canonical reviewed-real manifests and evaluation input staging."""

from __future__ import annotations

import csv
import json
import os
import shutil
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from image_clustering.evaluation.reviewed_models import (
    ALLOWED_SUBTYPES,
    SCHEMA_VERSION,
    SPLIT_VERSION,
    ManifestRow,
    expand_pairs,
    group_rows,
    load_csv_manifest,
)
from image_clustering.evaluation.reviewed_split import make_grouped_splits
from image_clustering.evaluation.reviewed_validate import (
    cluster_summary,
    validate_manifest,
    write_csv,
    write_jsonl,
)


def load_subtypes(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    output = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            cluster_id = row["original_cluster_id"].strip()
            subtype = row["occlusion_subtype"].strip()
            if subtype not in ALLOWED_SUBTYPES:
                raise ValueError(f"Unsupported subtype {subtype!r} for {cluster_id}")
            output[cluster_id] = {key: value for key, value in row.items() if key}
    return output


def _source_image(package_root: Path, relative_text: str) -> Path | None:
    relative = Path(relative_text)
    candidates = [package_root / relative]
    if relative.parts and relative.parts[0] == "images":
        candidates.append(package_root / Path(*relative.parts[1:]))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def stage_review_images(
    rows: Sequence[ManifestRow],
    package_root: Path,
    stage_root: Path,
) -> list[dict[str, Any]]:
    """Stage each reviewed proposal as an isolated evaluation sequence."""
    stage_root.mkdir(parents=True, exist_ok=True)
    staged = []
    seen: set[Path] = set()
    ordered = sorted(
        rows,
        key=lambda item: (item.original_cluster_id, item.sequence_index),
    )
    for row in ordered:
        source = _source_image(package_root, row.package_relative_path)
        if source is None:
            raise FileNotFoundError(row.package_relative_path)
        target = stage_root / row.original_cluster_id / Path(row.image_id).name
        if target in seen:
            raise ValueError(f"staging filename collision: {target}")
        seen.add(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != source.stat().st_size:
                raise ValueError(f"staged file differs from source: {target}")
            mode = "existing"
        else:
            try:
                os.link(source, target)
                mode = "hardlink"
            except OSError:
                try:
                    target.symlink_to(source.resolve())
                    mode = "symlink"
                except OSError:
                    shutil.copy2(source, target)
                    mode = "copy"
        staged.append(
            {
                "image_id": row.image_id,
                "original_cluster_id": row.original_cluster_id,
                "review_decision": row.review_decision,
                "sequence_id": row.sequence_id,
                "sequence_index": row.sequence_index,
                "evaluation_sequence_id": row.original_cluster_id,
                "prediction_image_id": target.relative_to(stage_root).as_posix(),
                "source_path": str(source),
                "staged_path": str(target),
                "storage_mode": mode,
            }
        )
    return staged


def prepare_dataset(
    csv_path: Path,
    output_dir: Path,
    *,
    jsonl_path: Path | None = None,
    package_root: Path | None = None,
    split_version: str = SPLIT_VERSION,
    stage_root: Path | None = None,
) -> dict[str, Any]:
    report = validate_manifest(
        csv_path,
        jsonl_path=jsonl_path,
        package_root=package_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "integrity_check_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    if report["status"] != "pass":
        raise ValueError("Reviewed dataset integrity validation failed")

    rows = load_csv_manifest(csv_path)
    groups = [cluster_summary(group) for group in group_rows(rows).values()]
    groups.sort(key=lambda row: row["original_cluster_id"])
    positive, negative = expand_pairs(rows)
    splits = make_grouped_splits(rows, split_version=split_version)
    write_jsonl(output_dir / "canonical_reviewed_groups.jsonl", groups)
    write_csv(output_dir / "canonical_reviewed_groups.csv", groups)
    positive_rows = [pair.as_dict() for pair in positive]
    negative_rows = [pair.as_dict() for pair in negative]
    write_jsonl(output_dir / "positive_pairs.jsonl", positive_rows)
    write_csv(output_dir / "positive_pairs.csv", positive_rows)
    write_jsonl(output_dir / "negative_pairs.jsonl", negative_rows)
    write_csv(output_dir / "negative_pairs.csv", negative_rows)
    split_rows = [assignment.as_dict() for assignment in splits]
    write_jsonl(output_dir / "split_manifest.jsonl", split_rows)
    write_csv(output_dir / "split_manifest.csv", split_rows)

    staged_rows: list[dict[str, Any]] = []
    if stage_root is not None:
        if package_root is None:
            raise ValueError("stage_root requires package_root")
        staged_rows = stage_review_images(rows, package_root, stage_root)
        write_csv(output_dir / "evaluation_input_manifest.csv", staged_rows)

    subtype_rows = [
        {
            "original_cluster_id": group["original_cluster_id"],
            "occlusion_subtype": "uncertain_occlusion_subtype",
            "occlusion_size_category": "unannotated",
            "registration_difficulty": "unannotated",
            "evidence": "",
            "annotator_method": "manual visual review; truth label unchanged",
        }
        for group in groups
        if group["review_decision"] == "accepted"
    ]
    write_csv(output_dir / "accepted_group_occlusion_subtypes.csv", subtype_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": report["csv_sha256"],
        "split_version": split_version,
        "counts": report["counts"],
        "split_counts": dict(Counter(item.split for item in splits)),
        "staged_image_count": len(staged_rows),
        "staging_storage_modes": dict(
            Counter(row["storage_mode"] for row in staged_rows)
        ),
        "split_truth_counts": {
            split: dict(
                Counter(
                    item.review_decision for item in splits if item.split == split
                )
            )
            for split in sorted({item.split for item in splits})
        },
    }
    (output_dir / "dataset_preparation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
