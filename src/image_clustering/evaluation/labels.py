"""Small JSONL label store for reviewed clusters, non-clusters, and crop targets."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VALID_RELATIONS = {
    "same_document",
    "same_document_near_duplicate",
    "same_document_occlusion",
    "different_document",
}
VALID_COMPLETENESS = {"complete", "partial_best_available", "review_required"}
VALID_CROP_KINDS = {"base_page", "data_bearing_overlay", "page_state"}


def _clean_stem(value: str) -> str:
    stem = Path(value).stem.replace("(1)", "")
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def make_case_id(images: Iterable[str]) -> str:
    """Create a stable readable identifier from ordered image names."""
    values = [_clean_stem(image) for image in images]
    if not values:
        raise ValueError("At least one image is required")
    return "__".join(values)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    values: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            message = f"Invalid JSON on {path}:{line_number}: {error}"
            raise ValueError(message) from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected JSON object on {path}:{line_number}")
        values.append(value)
    return values


def override_path(path: Path) -> Path:
    """Return the optional correction file associated with one JSONL store."""
    return path.with_name(f"{path.stem}.overrides{path.suffix}")


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Read reviewed cases and apply optional upsert/delete corrections.

    The companion ``<stem>.overrides.jsonl`` file is intentionally small. It
    lets a mistaken filename or crop label be corrected without rewriting a
    large append-friendly reviewed-case history. Override records either contain
    a normal case object or ``{"delete_case_id": "..."}``.
    """
    primary = _read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for case in primary:
        case_id = str(case.get("case_id", ""))
        if case_id in by_id:
            raise ValueError(f"Duplicate case_id in {path}: {case_id}")
        by_id[case_id] = case

    corrections = override_path(path)
    for operation in _read_jsonl(corrections):
        delete_case_id = operation.get("delete_case_id")
        if delete_case_id is not None:
            by_id.pop(str(delete_case_id), None)
            continue
        case_id = str(
            operation.get("case_id") or make_case_id(operation.get("images", []))
        )
        by_id[case_id] = {**operation, "case_id": case_id}
    return [by_id[case_id] for case_id in sorted(by_id)]


def write_cases(path: Path, cases: Iterable[dict[str, Any]]) -> None:
    """Write cases in stable case-id order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(cases, key=lambda case: str(case["case_id"]))
    payload = "\n".join(json.dumps(case, sort_keys=True) for case in ordered)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def upsert_case(path: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Insert or replace one case by case_id and validate the complete store."""
    cases = load_cases(path)
    case_id = str(case.get("case_id") or make_case_id(case.get("images", [])))
    normalized = {**case, "case_id": case_id}
    by_id = {str(existing["case_id"]): existing for existing in cases}
    by_id[case_id] = normalized
    validate_cases(list(by_id.values()))
    write_cases(path, by_id.values())
    return normalized


def append_crop(path: Path, case_id: str, crop: dict[str, Any]) -> dict[str, Any]:
    """Append a crop target to an existing reviewed case."""
    cases = load_cases(path)
    for case in cases:
        if case.get("case_id") != case_id:
            continue
        crops = list(case.get("expected_submissions", []))
        crops.append(crop)
        case["expected_submissions"] = crops
        validate_cases(cases)
        write_cases(path, cases)
        return case
    raise KeyError(f"Unknown case_id: {case_id}")


def validate_cases(cases: list[dict[str, Any]]) -> None:
    """Validate labels without requiring the private image files."""
    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id:
            raise ValueError("Every case requires case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id: {case_id}")
        seen.add(case_id)
        images = case.get("images")
        if not isinstance(images, list) or len(images) < 2:
            raise ValueError(f"{case_id}: images must contain at least two names")
        if len(set(images)) != len(images):
            raise ValueError(f"{case_id}: image names must be unique")
        relation = case.get("relation")
        if relation not in VALID_RELATIONS:
            raise ValueError(f"{case_id}: invalid relation {relation!r}")
        expected_cluster = case.get("expected_cluster")
        should_cluster = relation != "different_document"
        if expected_cluster is not should_cluster:
            raise ValueError(
                f"{case_id}: expected_cluster disagrees with relation {relation}"
            )
        minimum = int(case.get("expected_min_submissions", 0))
        if should_cluster and minimum < 1:
            raise ValueError(
                f"{case_id}: same-document cases require at least one submission"
            )
        for crop in case.get("expected_submissions", []):
            if crop.get("image") not in images:
                raise ValueError(f"{case_id}: crop image is not a case member")
            bbox = crop.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"{case_id}: crop bbox must have four values")
            coordinates = crop.get("coordinates", "normalized")
            if coordinates not in {"normalized", "pixels"}:
                raise ValueError(f"{case_id}: invalid coordinate type")
            if coordinates == "normalized" and not all(
                0.0 <= float(value) <= 1.0 for value in bbox
            ):
                raise ValueError(f"{case_id}: normalized bbox outside [0, 1]")
            if float(bbox[0]) >= float(bbox[2]) or float(bbox[1]) >= float(bbox[3]):
                raise ValueError(f"{case_id}: invalid bbox ordering")
            if crop.get("kind") not in VALID_CROP_KINDS:
                raise ValueError(f"{case_id}: invalid crop kind")
            if crop.get("completeness") not in VALID_COMPLETENESS:
                raise ValueError(f"{case_id}: invalid completeness")
