from __future__ import annotations

import csv
import json
from pathlib import Path

FIELDS = [
    "index",
    "image_id",
    "source_path",
    "source_project",
    "sequence_id",
    "sequence_index",
    "review_decision",
    "cluster_status",
    "assignment_id",
    "assignment_type",
    "original_cluster_id",
    "included_in_original_cluster",
    "reviewed_at",
    "package_relative_path",
]


def build_package(root: Path) -> tuple[Path, Path]:
    rows = []
    index = 0
    cluster_number = 0
    specs = [
        ("accepted", 2, 126),
        ("accepted", 3, 7),
        ("accepted", 9, 1),
        ("rejected", 2, 59),
        ("rejected", 3, 6),
        ("rejected", 4, 1),
    ]
    for decision, size, count in specs:
        for _ in range(count):
            cluster_id = f"cluster_{cluster_number:05d}"
            sequence_id = f"project/sequence_{cluster_number // 2:05d}"
            for offset in range(size):
                image_id = (
                    f"project/sequence_{cluster_number // 2:05d}/"
                    f"image_{cluster_number:05d}_{offset:02d}.jpg"
                )
                assignment_id = (
                    cluster_id
                    if decision == "accepted"
                    else f"{cluster_id}__separated_{offset + 1:05d}"
                )
                relative = f"images/{decision}/{assignment_id}/{image_id}"
                rows.append(
                    {
                        "index": index,
                        "image_id": image_id,
                        "source_path": f"D:/source/{image_id}",
                        "source_project": "project",
                        "sequence_id": sequence_id,
                        "sequence_index": cluster_number * 20 + offset,
                        "review_decision": decision,
                        "cluster_status": (
                            "approved" if decision == "accepted" else "dissolved"
                        ),
                        "assignment_id": assignment_id,
                        "assignment_type": (
                            "accepted_cluster"
                            if decision == "accepted"
                            else "rejected_singleton"
                        ),
                        "original_cluster_id": cluster_id,
                        "included_in_original_cluster": decision == "accepted",
                        "reviewed_at": "2026-08-03T00:00:00+00:00",
                        "package_relative_path": relative,
                    }
                )
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
                index += 1
            cluster_number += 1
    csv_path = root / "assignments.csv"
    jsonl_path = root / "assignments.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return csv_path, jsonl_path


def write_predictions(
    prepared: Path,
    path: Path,
    *,
    false_edge: bool = False,
) -> None:
    truths = []
    for name in ("positive_pairs.jsonl", "negative_pairs.jsonl"):
        truths.extend(
            json.loads(line)
            for line in (prepared / name).read_text().splitlines()
            if line
        )
    rows = []
    false_edge_added = False
    for truth in truths:
        accepted = truth["truth"] == "accepted"
        edge = accepted
        if false_edge and not accepted and not false_edge_added:
            edge = True
            false_edge_added = True
        p_same = 0.9 if accepted else 0.1
        q = 0.7 if accepted else 0.2
        rows.append(
            {
                "first_image_id": truth["image_a"],
                "second_image_id": truth["image_b"],
                "same_document": edge,
                "automatic_link_eligible": edge,
                "hard_contradiction": False,
                "occlusion_candidate_flag": accepted,
                "same_document_probability": p_same,
                "occluded_given_same_probability": q,
                "same_clean_probability": p_same * (1 - q),
                "same_occluded_probability": p_same * q,
                "different_document_probability": 1 - p_same,
                "registration_model": "homography",
                "registration_fallback_used": False,
                "feature_overlap": 0.5,
                "reason": "fixture",
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
