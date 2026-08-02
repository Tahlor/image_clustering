from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from loaders import (
    canonical_image_id,
    load_cv,
    load_embeddings,
    load_extractions,
    load_quality,
    merge_extraction_runs,
    merge_runs,
    pair_key,
    sequence_parts,
)
from models import Config, CvEvidence, Extraction, Record
from scoring import candidate_pairs, completeness, decide

SCHEMA_VERSION = "1.0"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        root_first, root_second = self.find(first), self.find(second)
        if root_first != root_second:
            self.parent[root_second] = root_first


def analyze(
    items: Sequence[Extraction],
    *,
    config: Config,
    embeddings: Mapping[str, np.ndarray] | None = None,
    cv_evidence: Mapping[tuple[str, str], CvEvidence] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    embeddings = embeddings or {}
    cv_evidence = cv_evidence or {}
    decisions = [
        decide(first, second, gap, config, embeddings, cv_evidence)
        for first, second, gap in candidate_pairs(items, config.max_gap)
    ]
    by_id = {item.image_id: item for item in items}
    union_find = UnionFind()
    for image_id in by_id:
        union_find.find(image_id)
    for decision in decisions:
        if decision["link_tier"] == "automatic":
            union_find.union(decision["image_a"], decision["image_b"])
    components: dict[str, list[str]] = defaultdict(list)
    for image_id in by_id:
        components[union_find.find(image_id)].append(image_id)

    groups = []
    group_by_image: dict[str, list[str]] = defaultdict(list)
    for image_ids in components.values():
        if len(image_ids) < 2:
            continue
        image_ids.sort(key=lambda image_id: by_id[image_id].sequence_index)
        edges = [
            decision
            for decision in decisions
            if decision["link_tier"] == "automatic"
            and decision["image_a"] in image_ids
            and decision["image_b"] in image_ids
        ]
        if not edges:
            continue
        group_id = "occ-" + hashlib.sha1("|".join(image_ids).encode()).hexdigest()[:10]
        canonical = max(
            image_ids,
            key=lambda image_id: (
                completeness(by_id[image_id]),
                by_id[image_id].field_mass,
                -by_id[image_id].sequence_index,
            ),
        )
        occluded = sorted(
            {
                decision["likely_occluded_image_id"]
                for decision in edges
                if decision["possible_occlusion_event"]
                and decision["likely_occluded_image_id"]
            }
        )
        more_complete = sorted(
            {
                decision["likely_more_complete_image_id"]
                for decision in edges
                if decision["possible_occlusion_event"]
                and decision["likely_more_complete_image_id"]
            }
        )
        group = {
            "schema_version": SCHEMA_VERSION,
            "group_id": group_id,
            "sequence_key": by_id[image_ids[0]].sequence_key,
            "image_ids": image_ids,
            "canonical_image_id": canonical,
            "possible_occluded_view_ids": occluded,
            "possible_more_complete_companion_ids": more_complete,
            "multiple_occlusion_states": len(occluded) > 1,
            "possible_occlusion_event": any(
                decision["possible_occlusion_event"] for decision in edges
            ),
            "group_confidence": min(
                decision["same_document_score"] for decision in edges
            ),
            "automatic_edge_count": len(edges),
            "edges": [
                pair_key(decision["image_a"], decision["image_b"])
                for decision in edges
            ],
        }
        groups.append(group)
        for image_id in image_ids:
            group_by_image[image_id].append(group_id)

    candidate_links: dict[str, set[str]] = defaultdict(set)
    automatic_links: dict[str, set[str]] = defaultdict(set)
    occluded_roles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    complete_roles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_images: set[str] = set()
    for decision in decisions:
        if not decision["possible_same_document"]:
            continue
        first, second = decision["image_a"], decision["image_b"]
        candidate_links[first].add(second)
        candidate_links[second].add(first)
        if decision["link_tier"] == "automatic":
            automatic_links[first].add(second)
            automatic_links[second].add(first)
        if decision["possible_occlusion_event"]:
            event_images.update((first, second))
        if decision["likely_occluded_image_id"]:
            occluded_roles[decision["likely_occluded_image_id"]].append(decision)
        if decision["likely_more_complete_image_id"]:
            complete_roles[decision["likely_more_complete_image_id"]].append(decision)

    group_lookup = {group["group_id"]: group for group in groups}
    flags = []
    for item in items:
        group_ids = sorted(group_by_image[item.image_id])
        role_edges = occluded_roles[item.image_id] + complete_roles[item.image_id]
        flags.append(
            {
                "schema_version": SCHEMA_VERSION,
                "image_id": item.image_id,
                "source_filename": item.source_filename,
                "sequence_key": item.sequence_key,
                "sequence_index": item.sequence_index,
                "possible_occlusion_event": item.image_id in event_images,
                "possible_occluded_view": bool(occluded_roles[item.image_id]),
                "possible_more_complete_companion": bool(complete_roles[item.image_id]),
                "role_confidence": max(
                    (decision["role_confidence"] for decision in role_edges),
                    default=0.0,
                ),
                "linked_image_ids": sorted(candidate_links[item.image_id]),
                "automatic_linked_image_ids": sorted(automatic_links[item.image_id]),
                "occlusion_group_ids": group_ids,
                "canonical_for_group_ids": [
                    group_id
                    for group_id in group_ids
                    if group_lookup[group_id]["canonical_image_id"] == item.image_id
                ],
                "run_count": item.run_count,
                "field_mass": item.field_mass,
                "record_count": len(item.records),
                "quality": item.quality,
            }
        )
    return (
        decisions,
        sorted(groups, key=lambda group: (group["sequence_key"], group["image_ids"])),
        sorted(flags, key=lambda row: (row["sequence_key"], row["sequence_index"])),
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def write_outputs(
    output_dir: Path,
    items: Sequence[Extraction],
    decisions: Sequence[dict[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    flags: Sequence[Mapping[str, Any]],
    config: Config,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rounded = json.loads(json.dumps(decisions))
    for decision in rounded:
        decision["same_document_score"] = round(decision["same_document_score"], 6)
        decision["role_confidence"] = round(decision["role_confidence"], 6)
        for key, value in decision["signals"].items():
            if isinstance(value, float):
                decision["signals"][key] = round(value, 6)
    write_jsonl(output_dir / "adjacent_pair_scores.jsonl", rounded)
    write_jsonl(output_dir / "occlusion_groups.jsonl", groups)
    write_jsonl(output_dir / "image_flags.jsonl", flags)
    write_jsonl(
        output_dir / "review_queue.jsonl",
        (decision for decision in rounded if decision["link_tier"] == "review"),
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(decision['link_tier'])}</td>"
        f"<td>{decision['same_document_score']:.3f}</td>"
        f"<td>{html.escape(decision['image_a'])}</td>"
        f"<td>{html.escape(decision['image_b'])}</td>"
        f"<td>{'yes' if decision['possible_occlusion_event'] else 'no'}</td>"
        f"<td>{html.escape(', '.join(decision['reasons']))}</td>"
        f"<td><pre>{html.escape(json.dumps(decision['signals'], indent=2))}</pre></td>"
        "</tr>"
        for decision in sorted(
            decisions,
            key=lambda row: row["same_document_score"],
            reverse=True,
        )
        if decision["link_tier"] != "rejected"
    )
    (output_dir / "review.html").write_text(
        "<!doctype html><meta charset=utf-8>"
        "<title>Adjacent occlusion review</title>"
        "<style>body{font-family:system-ui}table{border-collapse:collapse}"
        "th,td{border:1px solid #aaa;padding:.4rem;vertical-align:top}"
        "pre{white-space:pre-wrap}</style>"
        "<h1>Adjacent extraction occlusion review</h1>"
        "<table><tr><th>tier</th><th>score</th><th>A</th><th>B</th>"
        "<th>occlusion</th><th>reasons</th><th>signals</th></tr>"
        f"{table_rows}</table>",
        encoding="utf-8",
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "image_count": len(items),
        "candidate_pair_count": len(decisions),
        "automatic_link_count": sum(
            decision["link_tier"] == "automatic" for decision in decisions
        ),
        "review_link_count": sum(
            decision["link_tier"] == "review" for decision in decisions
        ),
        "possible_occlusion_pair_count": sum(
            decision["possible_occlusion_event"] for decision in decisions
        ),
        "occlusion_group_count": len(groups),
        "possible_occluded_image_count": sum(
            bool(row["possible_occluded_view"]) for row in flags
        ),
        "possible_companion_image_count": sum(
            bool(row["possible_more_complete_companion"]) for row in flags
        ),
        "outputs": {
            "pair_scores": "adjacent_pair_scores.jsonl",
            "groups": "occlusion_groups.jsonl",
            "image_flags": "image_flags.jsonl",
            "review_queue": "review_queue.jsonl",
            "review_html": "review.html",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Link adjacent extractions that may show one physical document "
            "under different occlusions."
        )
    )
    parser.add_argument("--extractions", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--cv-evidence", type=Path)
    parser.add_argument("--quality-metadata", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = Config.from_mapping(
        json.loads(args.config.read_text()) if args.config else None
    )
    items = load_extractions(args.extractions)
    metadata = load_quality(args.quality_metadata)
    for item in items:
        item.quality.update(metadata.get(item.image_id, {}))
    decisions, groups, flags = analyze(
        items,
        config=config,
        embeddings=load_embeddings(args.embeddings),
        cv_evidence=load_cv(args.cv_evidence),
    )
    print(
        json.dumps(
            write_outputs(output_dir=args.output_dir, items=items, decisions=decisions, groups=groups, flags=flags, config=config),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "Config",
    "CvEvidence",
    "Extraction",
    "Record",
    "analyze",
    "canonical_image_id",
    "load_extractions",
    "merge_extraction_runs",
    "merge_runs",
    "pair_key",
    "sequence_parts",
    "write_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
