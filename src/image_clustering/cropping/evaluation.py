"""Evaluation helpers for comparing crop manifests with curated targets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .image_ops import bbox_iou
from .io_utils import read_json
from .models import BBox


@dataclass(frozen=True)
class EvaluationMetrics:
    """Exact-set and crop-localization metrics for one curated fixture."""

    expected_count: int
    actual_count: int
    matched_count: int
    false_positive_count: int
    missed_count: int
    exact_submission_set: bool
    mean_iou: float
    minimum_iou: float

    def to_dict(self) -> dict[str, int | bool | float]:
        """Convert metrics to a JSON-serializable dictionary."""
        return {
            "expected_count": self.expected_count,
            "actual_count": self.actual_count,
            "matched_count": self.matched_count,
            "false_positive_count": self.false_positive_count,
            "missed_count": self.missed_count,
            "exact_submission_set": self.exact_submission_set,
            "mean_iou": self.mean_iou,
            "minimum_iou": self.minimum_iou,
        }


def _flatten_submissions(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if "submissions" in manifest:
        return list(manifest["submissions"])
    submissions: list[dict[str, Any]] = []
    for cluster in manifest.get("clusters", []):
        if "submissions" in cluster:
            submissions.extend(cluster["submissions"])
        for group in cluster.get("clusters", []):
            submissions.extend(group.get("submissions", []))
    return submissions


def _key(item: dict[str, Any]) -> tuple[str, str, str]:
    source = Path(item.get("source_image") or item["source_path"]).name
    return source, str(item["kind"]), str(item["completeness"])


def evaluate_manifest(
    manifest: dict[str, Any],
    expected: dict[str, Any],
) -> EvaluationMetrics:
    """Compare automatic submissions with curated source/kind/completeness targets.

    Args:
        manifest: Automatic crop manifest or aggregate cropping result.
        expected: Curated fixture containing ``expected_submissions``.

    Returns:
        Exact-set counts and bounding-box intersection-over-union metrics.
    """
    actual = _flatten_submissions(manifest)
    targets = list(expected["expected_submissions"])
    remaining = set(range(len(actual)))
    ious: list[float] = []

    for target in targets:
        candidates = [
            index for index in remaining if _key(actual[index]) == _key(target)
        ]
        if not candidates:
            continue
        target_bbox = tuple(int(value) for value in target["bbox"])
        best_index = max(
            candidates,
            key=lambda index: bbox_iou(
                target_bbox,
                tuple(int(value) for value in actual[index]["bbox"]),
            ),
        )
        best_bbox: BBox = tuple(int(value) for value in actual[best_index]["bbox"])
        ious.append(bbox_iou(target_bbox, best_bbox))
        remaining.remove(best_index)

    matched = len(ious)
    missed = len(targets) - matched
    false_positive = len(actual) - matched
    return EvaluationMetrics(
        expected_count=len(targets),
        actual_count=len(actual),
        matched_count=matched,
        false_positive_count=false_positive,
        missed_count=missed,
        exact_submission_set=missed == 0 and false_positive == 0,
        mean_iou=sum(ious) / max(len(ious), 1),
        minimum_iou=min(ious, default=0.0),
    )


def evaluate_files(manifest_path: Path, expected_path: Path) -> EvaluationMetrics:
    """Read and evaluate two JSON files."""
    manifest = read_json(manifest_path)
    expected = read_json(expected_path)
    if not isinstance(manifest, dict) or not isinstance(expected, dict):
        raise ValueError("Evaluation inputs must be JSON objects")
    return evaluate_manifest(manifest=manifest, expected=expected)
