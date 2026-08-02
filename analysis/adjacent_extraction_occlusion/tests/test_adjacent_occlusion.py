from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adjacent_occlusion import (
    Config,
    CvEvidence,
    Extraction,
    Record,
    analyze,
    canonical_image_id,
    load_extractions,
    merge_extraction_runs,
    pair_key,
    sequence_parts,
    write_outputs,
)


def extraction(
    image_id: str,
    records: list[Record],
    *,
    occlusion: float = 0.0,
    crop: float = 0.0,
    indexability: float = 0.9,
) -> Extraction:
    sequence_key, sequence_index = sequence_parts(image_id)
    return Extraction(
        image_id=image_id,
        source_filename=f"{image_id}.jpg",
        sequence_key=sequence_key,
        sequence_index=sequence_index,
        records=records,
        quality={
            "occlusion_risk": occlusion,
            "crop_risk": crop,
            "indexability_score": indexability,
        },
    )


def self_record(**fields: str) -> Record:
    return Record(role="self", event_type="Petition", fields=fields)


def test_canonical_image_id_recovers_source_from_batch_stem() -> None:
    value = "0170_63129.IMG.001_db62c8190639_63129_2327225e_0014-00136"
    assert canonical_image_id(value) == "63129_2327225e_0014-00136"
    assert canonical_image_id("i4071659-01096(1).jpg") == "i4071659-01096"


def test_partial_and_complete_views_are_linked_and_roles_are_assigned() -> None:
    partial = extraction(
        "63129_demo_0001-00100",
        [self_record(given_name="Marie", surname="Gagnon", petition_number="814")],
        occlusion=0.7,
        crop=0.3,
        indexability=0.55,
    )
    complete = extraction(
        "63129_demo_0001-00101",
        [
            self_record(
                given_name="Marie",
                surname="Gagnon",
                petition_number="814",
                birth_year="1912",
                birth_city="Sherbrooke",
                birth_country="Canada",
                marriage_year="1934",
            ),
            Record(role="child", event_type="Petition", fields={"given_name": "Luc"}),
        ],
    )
    cv = {
        pair_key(partial.image_id, complete.image_id): CvEvidence(
            same_scene_probability=0.96, occlusion_probability=0.94
        )
    }
    decisions, groups, flags = analyze(
        [partial, complete], config=Config(), cv_evidence=cv
    )
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision["link_tier"] == "automatic"
    assert decision["possible_occlusion_event"]
    assert decision["likely_occluded_image_id"] == partial.image_id
    assert decision["likely_more_complete_image_id"] == complete.image_id
    assert len(groups) == 1
    assert groups[0]["canonical_image_id"] == complete.image_id
    by_id = {row["image_id"]: row for row in flags}
    assert by_id[partial.image_id]["possible_occluded_view"]
    assert by_id[complete.image_id]["possible_more_complete_companion"]


def test_same_person_different_legal_page_is_not_automatically_linked() -> None:
    petition = extraction(
        "63129_demo_0001-00100",
        [
            Record(
                role="self",
                event_type="Petition",
                fields={
                    "given_name": "Marie",
                    "surname": "Gagnon",
                    "petition_number": "814",
                    "birth_year": "1912",
                    "birth_city": "Sherbrooke",
                },
            )
        ],
    )
    oath = extraction(
        "63129_demo_0001-00101",
        [
            Record(
                role="self",
                event_type="Oath",
                fields={
                    "given_name": "Marie",
                    "surname": "Gagnon",
                    "event_day": "12",
                    "event_month": "Jun",
                    "event_year": "1952",
                },
            )
        ],
    )
    decisions, groups, _ = analyze([petition, oath], config=Config())
    assert decisions[0]["link_tier"] != "automatic"
    assert not decisions[0]["possible_occlusion_event"]
    assert groups == []


def test_multiple_occluded_views_form_one_group() -> None:
    base_fields = {
        "given_name": "John",
        "surname": "Smith",
        "petition_number": "42",
    }
    first = extraction(
        "63129_demo_0001-00100",
        [self_record(**base_fields)],
        occlusion=0.7,
        indexability=0.5,
    )
    second = extraction(
        "63129_demo_0001-00101",
        [self_record(**base_fields, birth_year="1901")],
        occlusion=0.45,
        indexability=0.7,
    )
    third = extraction(
        "63129_demo_0001-00102",
        [
            self_record(
                **base_fields,
                birth_year="1901",
                birth_city="Quebec",
                birth_country="Canada",
                arrival_year="1920",
            )
        ],
        occlusion=0.05,
        indexability=0.98,
    )
    cv = {
        pair_key(first.image_id, second.image_id): CvEvidence(0.92, 0.85),
        pair_key(second.image_id, third.image_id): CvEvidence(0.94, 0.88),
    }
    _, groups, _ = analyze([first, second, third], config=Config(), cv_evidence=cv)
    assert len(groups) == 1
    assert groups[0]["image_ids"] == [first.image_id, second.image_id, third.image_id]
    assert groups[0]["canonical_image_id"] == third.image_id
    assert groups[0]["multiple_occlusion_states"]


def test_strong_embedding_can_support_sparse_extractions() -> None:
    first = extraction(
        "63129_demo_0001-00100",
        [self_record(given_name="Alice", surname="Rediker")],
        occlusion=0.6,
    )
    second = extraction(
        "63129_demo_0001-00101",
        [self_record(given_name="Alice", surname="Rediker", birth_year="1920")],
    )
    embeddings = {
        first.image_id: np.array([1.0, 0.0, 0.0], dtype=np.float32),
        second.image_id: np.array([0.999, 0.02, 0.0], dtype=np.float32),
    }
    decisions, _, _ = analyze(
        [first, second], config=Config(), embeddings=embeddings
    )
    assert decisions[0]["possible_same_document"]
    assert "strong_embedding_similarity" in decisions[0]["reasons"]


def test_multiple_runs_are_consensus_merged(tmp_path: Path) -> None:
    first = tmp_path / "run_a.jsonl"
    second = tmp_path / "run_b.jsonl"
    first.write_text(
        json.dumps(
            {
                "image_stem": "0001_x_63129_demo_0001-00100",
                "parsed_response": {
                    "records": [
                        {
                            "role": "self",
                            "event_type": "Petition",
                            "fields": {"given_name": "Alice Mae", "surname": "Rediker"},
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "source_filename": "63129_demo_0001-00100.jpg",
                "records": [
                    {
                        "role": "self",
                        "event_type": "Petition",
                        "fields": {
                            "given_name": "Alice Mae",
                            "surname": "Rediker",
                            "birth_year": "1920",
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    merged = load_extractions([first, second])
    assert len(merged) == 1
    assert merged[0].run_count == 2
    assert merged[0].records[0].fields["birth_year"] == "1920"


def test_outputs_include_review_and_link_artifacts(tmp_path: Path) -> None:
    first = extraction(
        "63129_demo_0001-00100",
        [self_record(given_name="Alice", surname="Rediker")],
        occlusion=0.7,
    )
    second = extraction(
        "63129_demo_0001-00101",
        [self_record(given_name="Alice", surname="Rediker", birth_year="1920")],
    )
    decisions, groups, flags = analyze(
        [first, second],
        config=Config(),
        cv_evidence={
            pair_key(first.image_id, second.image_id): CvEvidence(0.95, 0.9)
        },
    )
    summary = write_outputs(
        tmp_path, [first, second], decisions, groups, flags, Config()
    )
    assert summary["image_count"] == 2
    for name in (
        "adjacent_pair_scores.jsonl",
        "occlusion_groups.jsonl",
        "image_flags.jsonl",
        "review_queue.jsonl",
        "review.html",
        "summary.json",
    ):
        assert (tmp_path / name).exists()


def test_merge_extraction_runs_preserves_unmatched_child() -> None:
    base = extraction(
        "63129_demo_0001-00100",
        [self_record(given_name="A", surname="B")],
    )
    other = extraction(
        "63129_demo_0001-00100",
        [
            self_record(given_name="A", surname="B"),
            Record(role="child", event_type="Petition", fields={"given_name": "C"}),
        ],
    )
    merged = merge_extraction_runs([base, other])
    assert len(merged.records) == 2
