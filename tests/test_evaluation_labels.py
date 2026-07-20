from __future__ import annotations

from pathlib import Path

import pytest

from image_clustering.evaluation.cli import run
from image_clustering.evaluation.labels import (
    append_crop,
    load_cases,
    make_case_id,
    upsert_case,
    validate_cases,
)


def test_reviewed_fixture_is_valid() -> None:
    path = Path("examples/evaluation/reviewed_cases.jsonl")
    cases = load_cases(path)
    validate_cases(cases)
    assert len(cases) >= 20
    positive = [case for case in cases if case["expected_cluster"]]
    assert positive
    assert all(case["expected_min_submissions"] >= 1 for case in positive)


def test_upsert_and_append_crop(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    case = upsert_case(
        path,
        {
            "case_id": "a__b",
            "images": ["a.j2k", "b.j2k"],
            "relation": "same_document_near_duplicate",
            "expected_cluster": True,
            "expected_min_submissions": 1,
            "expected_submissions": [],
        },
    )
    assert case["case_id"] == "a__b"
    append_crop(
        path,
        "a__b",
        {
            "image": "a.j2k",
            "bbox": [0.1, 0.1, 0.9, 0.9],
            "coordinates": "normalized",
            "kind": "base_page",
            "completeness": "complete",
            "side": "single",
        },
    )
    restored = load_cases(path)
    assert len(restored[0]["expected_submissions"]) == 1


def test_cli_adds_different_pair(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    assert run(["pair", str(path), "one.j2k", "two.j2k", "--different"]) == 0
    case = load_cases(path)[0]
    assert case["case_id"] == make_case_id(["one.j2k", "two.j2k"])
    assert case["expected_cluster"] is False
    assert case["expected_min_submissions"] == 0


def test_same_document_requires_submission() -> None:
    with pytest.raises(ValueError, match="at least one submission"):
        validate_cases(
            [
                {
                    "case_id": "bad",
                    "images": ["a", "b"],
                    "relation": "same_document",
                    "expected_cluster": True,
                    "expected_min_submissions": 0,
                }
            ]
        )
