"""Prediction loading, canonicalization, and safe probability handling."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from image_clustering.evaluation.reviewed_models import SCHEMA_VERSION, load_jsonl


class UnionFind:
    def __init__(self, items: Iterable[str] = ()) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        left, right = self.find(first), self.find(second)
        if left != right:
            self.parent[right] = left


def prediction_id_map(prepared_dir: Path) -> dict[str, str]:
    path = prepared_dir / "evaluation_input_manifest.csv"
    if not path.is_file():
        return {}
    output = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            prediction_id = str(row.get("prediction_image_id") or "").strip()
            image_id = str(row.get("image_id") or "").strip()
            if not prediction_id or not image_id:
                raise ValueError(f"Invalid evaluation input mapping row in {path}")
            if prediction_id in output and output[prediction_id] != image_id:
                raise ValueError(f"Prediction ID maps twice: {prediction_id}")
            output[prediction_id] = image_id
            output[image_id] = image_id
    return output


def _canonicalize_pair(
    row: Mapping[str, Any],
    id_map: Mapping[str, str],
) -> dict[str, Any]:
    output = dict(row)
    for name in ("first_image_id", "image_a", "image_id_a"):
        if name in output:
            output[name] = id_map.get(str(output[name]), str(output[name]))
    for name in ("second_image_id", "image_b", "image_id_b"):
        if name in output:
            output[name] = id_map.get(str(output[name]), str(output[name]))
    return output


def load_prediction_source(
    path: Path,
    *,
    id_map: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    mapping = id_map or {}
    if path.suffix.lower() != ".json":
        return [_canonicalize_pair(row, mapping) for row in load_jsonl(path)], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [_canonicalize_pair(row, mapping) for row in payload], None
    if not isinstance(payload, dict) or "comparisons" not in payload:
        raise ValueError(f"Unsupported prediction JSON structure: {path}")
    components = {}
    for cluster in payload.get("clusters", []):
        cluster_id = str(cluster["cluster_id"])
        for image_id in cluster.get("image_ids", []):
            canonical = mapping.get(str(image_id), str(image_id))
            if canonical in components and components[canonical] != cluster_id:
                raise ValueError(f"Image appears in multiple components: {canonical}")
            components[canonical] = cluster_id
    comparisons = [
        _canonicalize_pair(row, mapping) for row in payload["comparisons"]
    ]
    return comparisons, components


def _coalesce(row: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    return next((row[name] for name in names if name in row), default)


def prediction_key(row: Mapping[str, Any]) -> tuple[str, str]:
    first = str(_coalesce(row, "first_image_id", "image_a", "image_id_a", default=""))
    second = str(
        _coalesce(row, "second_image_id", "image_b", "image_id_b", default="")
    )
    if not first or not second or first == second:
        raise ValueError("pair prediction requires two distinct image IDs")
    return tuple(sorted((first, second)))


def _probability(row: Mapping[str, Any], name: str, default: float) -> float:
    value = float(row.get(name, default))
    if not 0.0 <= value <= 1.0 or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite probability, got {value}")
    return value


def normalize_prediction(row: Mapping[str, Any]) -> dict[str, Any]:
    first, second = prediction_key(row)
    contradiction = bool(row.get("hard_contradiction", False))
    same = bool(
        _coalesce(
            row,
            "same_document",
            "deterministic_same_document",
            default=False,
        )
    )
    eligible = bool(row.get("automatic_link_eligible", same))
    edge = same and eligible and not contradiction
    candidate = bool(
        _coalesce(
            row,
            "occlusion_candidate_flag",
            "candidate_flag",
            default=False,
        )
    )
    p_same = _probability(row, "same_document_probability", float(same))
    q = _probability(row, "occluded_given_same_probability", 0.0)
    p_clean = _probability(row, "same_clean_probability", p_same * (1 - q))
    p_occluded = _probability(row, "same_occluded_probability", p_same * q)
    p_different = _probability(row, "different_document_probability", 1 - p_same)
    if not math.isclose(p_clean + p_occluded, p_same, abs_tol=1e-6):
        raise ValueError(f"incoherent same-state probabilities: {first}, {second}")
    if not math.isclose(p_same + p_different, 1.0, abs_tol=1e-6):
        raise ValueError(f"incoherent same/different probabilities: {first}, {second}")
    return {
        **dict(row),
        "image_a": first,
        "image_b": second,
        "deterministic_same_document": same,
        "automatic_link_eligible": eligible,
        "automatic_edge": edge,
        "candidate_flag": candidate,
        "hard_contradiction": contradiction,
        "same_document_probability": p_same,
        "occluded_given_same_probability": q,
        "same_clean_probability": p_clean,
        "same_occluded_probability": p_occluded,
        "different_document_probability": p_different,
    }


def apply_isotonic(value: float, knots: Sequence[Sequence[float]]) -> float:
    if not knots:
        return value
    points = [(float(x), float(y)) for x, y in knots]
    if value <= points[0][0]:
        return points[0][1]
    if value >= points[-1][0]:
        return points[-1][1]
    for index in range(1, len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        if x0 <= value <= x1:
            if x0 == x1:
                return max(y0, y1)
            weight = (value - x0) / (x1 - x0)
            return y0 + weight * (y1 - y0)
    return points[-1][1]


def load_calibrator(
    path: Path | None,
    *,
    prepared_dir: Path,
) -> dict[str, Any] | None:
    if path is None:
        return None
    calibrator = json.loads(path.read_text(encoding="utf-8"))
    if calibrator.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported calibrator schema: {path}")
    summary = json.loads(
        (prepared_dir / "dataset_preparation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if calibrator.get("prepared_manifest_sha256") != summary["manifest_sha256"]:
        raise ValueError("calibrator was fit on a different prepared manifest")
    if calibrator.get("locked_audit_used_for_fit") is not False:
        raise ValueError("calibrator must not use the locked audit")
    return calibrator


def calibrate_prediction(
    prediction: Mapping[str, Any],
    calibrator: Mapping[str, Any] | None,
) -> dict[str, Any]:
    output = dict(prediction)
    if calibrator is None:
        output["probability_calibration_source"] = output.get(
            "probability_model_version", "uncalibrated"
        )
        return output
    p_same = apply_isotonic(
        float(output["same_document_probability"]),
        calibrator["identity_isotonic_knots"],
    )
    q_knots = calibrator.get("conditional_occlusion_isotonic_knots")
    q = (
        apply_isotonic(float(output["occluded_given_same_probability"]), q_knots)
        if q_knots
        else float(output["occluded_given_same_probability"])
    )
    output.update(
        same_document_probability=p_same,
        occluded_given_same_probability=q,
        same_clean_probability=p_same * (1 - q),
        same_occluded_probability=p_same * q,
        different_document_probability=1 - p_same,
        probability_calibration_source=calibrator["calibration_source"],
    )
    return output
