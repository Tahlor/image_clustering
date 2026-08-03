"""Replay and tune reviewed real-data candidate scores without image recomputation.

The replay path consumes the canonical reviewed manifest plus the scored candidate
CSV emitted by the production occlusion pipeline. Probability thresholds may
rank review candidates and may act as an *additional* automatic-edge gate, but
never create an edge by themselves: deterministic identity, automatic-link
eligibility, and contradiction safeguards remain mandatory.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_clustering.evaluation.reviewed_models import (
    ManifestRow,
    load_csv_manifest,
    load_jsonl,
    parse_bool,
)
from image_clustering.evaluation.reviewed_predictions import UnionFind
from image_clustering.evaluation.reviewed_validate import write_csv, write_jsonl

AUTO_SCORE_FIELDS = (
    "same_occluded_probability",
    "same_document_probability",
    "occluded_given_same_probability",
)


@dataclass(frozen=True)
class ReplayConfig:
    score_field: str
    automatic_threshold: float
    candidate_threshold: float
    max_index_gap: int | None = None
    min_alignment_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "score_field": self.score_field,
            "automatic_threshold": self.automatic_threshold,
            "candidate_threshold": self.candidate_threshold,
            "max_index_gap": self.max_index_gap,
            "min_alignment_score": self.min_alignment_score,
            "edge_policy": (
                "deterministic_same_document AND automatic_link_eligible AND "
                "NOT hard_contradiction AND tuned supplemental gates"
            ),
        }


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    output = float(value)
    if not math.isfinite(output):
        raise ValueError(f"non-finite numeric value: {value!r}")
    return output


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _path_identity(value: str) -> tuple[str, str]:
    text = str(value).strip().replace("\\", "/")
    parts = [part for part in text.split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"image identity lacks sequence and filename: {value!r}")
    sequence = parts[-2].lower()
    filename = parts[-1]
    stem = filename.rsplit(".", 1)[0].lower()
    return sequence, stem


def manifest_identity_map(rows: Sequence[ManifestRow]) -> dict[tuple[str, str], str]:
    output: dict[tuple[str, str], str] = {}
    for row in rows:
        key = _path_identity(row.image_id)
        previous = output.get(key)
        if previous is not None and previous != row.image_id:
            raise ValueError(
                f"ambiguous reviewed image identity {key}: "
                f"{previous}, {row.image_id}"
            )
        output[key] = row.image_id
    return output


def _parse_candidate_row(
    row: Mapping[str, Any],
    identity_map: Mapping[tuple[str, str], str],
) -> dict[str, Any] | None:
    first_raw = str(row.get("first_image_id") or "").strip()
    second_raw = str(row.get("second_image_id") or "").strip()
    if not first_raw or not second_raw:
        raise ValueError("candidate row requires first_image_id and second_image_id")
    first = identity_map.get(_path_identity(first_raw))
    second = identity_map.get(_path_identity(second_raw))
    if first is None or second is None:
        return None
    if first == second:
        raise ValueError(f"candidate row maps both sides to {first}")
    output = dict(row)
    output.update(
        first_image_id=first,
        second_image_id=second,
        deterministic_same_document=parse_bool(
            row.get("deterministic_same_document", False)
        ),
        automatic_link_eligible=parse_bool(
            row.get("automatic_link_eligible", False)
        ),
        hard_contradiction=parse_bool(row.get("hard_contradiction", False)),
        raw_hard_contradiction=parse_bool(
            row.get("raw_hard_contradiction", False)
        ),
        index_gap=_optional_int(row.get("index_gap")),
        registration_alignment_score=_optional_float(
            row.get("registration_alignment_score")
        ),
    )
    for field in AUTO_SCORE_FIELDS:
        value = _optional_float(row.get(field))
        if value is None:
            raise ValueError(f"candidate row is missing {field}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be a probability: {value}")
        output[field] = value
    return output


def load_reviewed_candidate_csv(
    assignments_csv: Path,
    candidates_csv: Path,
) -> tuple[list[ManifestRow], list[dict[str, Any]], dict[str, Any]]:
    rows = load_csv_manifest(assignments_csv)
    identity_map = manifest_identity_map(rows)
    cluster_by_image = {row.image_id: row.original_cluster_id for row in rows}
    matched: list[dict[str, Any]] = []
    unmatched = 0
    cross_group = 0
    duplicates = 0
    seen: set[tuple[str, str]] = set()
    with candidates_csv.open(newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            parsed = _parse_candidate_row(raw, identity_map)
            if parsed is None:
                unmatched += 1
                continue
            first = parsed["first_image_id"]
            second = parsed["second_image_id"]
            if cluster_by_image[first] != cluster_by_image[second]:
                # Cross-proposal truth was not reviewed; do not invent a label.
                cross_group += 1
                continue
            key = tuple(sorted((first, second)))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            parsed["original_cluster_id"] = cluster_by_image[first]
            matched.append(parsed)
    summary = {
        "reviewed_manifest_rows": len(rows),
        "matched_within_reviewed_group_candidates": len(matched),
        "unmatched_candidate_rows": unmatched,
        "excluded_cross_group_rows_unlabeled": cross_group,
        "duplicate_candidate_rows_ignored": duplicates,
    }
    return rows, matched, summary


def _score_values(
    candidates: Sequence[Mapping[str, Any]], field: str
) -> list[float]:
    values = sorted({float(row[field]) for row in candidates})
    if not values:
        return [1.000000001]
    return [0.0, *values, 1.000000001]


def _compact_values(values: Iterable[float], *, limit: int = 24) -> list[float]:
    ordered = sorted(set(values))
    if len(ordered) <= limit:
        return ordered
    positions = {
        round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)
    }
    return [ordered[index] for index in sorted(positions)]


def _candidate_threshold(
    candidates: Sequence[Mapping[str, Any]],
    truth: Mapping[tuple[str, str], int],
    field: str,
    allowed_clusters: set[str],
    max_review_fraction: float,
) -> float:
    relevant = [
        row for row in candidates if row["original_cluster_id"] in allowed_clusters
    ]
    if not relevant:
        return 1.000000001
    best: tuple[tuple[float, float, float], float] | None = None
    for threshold in _score_values(relevant, field):
        selected = [row for row in relevant if float(row[field]) >= threshold]
        review_fraction = len(selected) / len(relevant)
        if review_fraction > max_review_fraction + 1e-12:
            continue
        tp = sum(
            truth[tuple(sorted((row["first_image_id"], row["second_image_id"])))]
            for row in selected
        )
        positives = sum(
            truth[tuple(sorted((row["first_image_id"], row["second_image_id"])))]
            for row in relevant
        )
        precision = tp / len(selected) if selected else 1.0
        recall = tp / positives if positives else 1.0
        objective = (recall, precision, -review_fraction)
        candidate = (objective, threshold)
        if best is None or candidate > best:
            best = candidate
    return best[1] if best is not None else 1.000000001


def _automatic_edge(row: Mapping[str, Any], config: ReplayConfig) -> bool:
    if not bool(row["deterministic_same_document"]):
        return False
    if not bool(row["automatic_link_eligible"]):
        return False
    if bool(row["hard_contradiction"]):
        return False
    if float(row[config.score_field]) < config.automatic_threshold:
        return False
    gap = row.get("index_gap")
    if config.max_index_gap is not None and gap is not None:
        if int(gap) > config.max_index_gap:
            return False
    alignment = row.get("registration_alignment_score")
    if config.min_alignment_score is not None:
        if alignment is None or float(alignment) < config.min_alignment_score:
            return False
    return True


def _truth_maps(
    prepared_dir: Path,
) -> tuple[
    dict[tuple[str, str], int],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    truth: dict[tuple[str, str], int] = {}
    for filename, value in (("positive_pairs.jsonl", 1), ("negative_pairs.jsonl", 0)):
        for row in load_jsonl(prepared_dir / filename):
            truth[tuple(sorted((row["image_a"], row["image_b"])))] = value
    groups = {
        row["original_cluster_id"]: row
        for row in load_jsonl(prepared_dir / "canonical_reviewed_groups.jsonl")
    }
    splits = {
        row["original_cluster_id"]: row["split"]
        for row in load_jsonl(prepared_dir / "split_manifest.jsonl")
    }
    return truth, groups, splits


def replay_metrics(
    candidates: Sequence[Mapping[str, Any]],
    prepared_dir: Path,
    config: ReplayConfig,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    truth, groups, splits = _truth_maps(prepared_dir)
    allowed_clusters = {
        cluster_id
        for cluster_id in groups
        if split is None or splits[cluster_id] == split
    }
    image_ids = [
        image_id
        for cluster_id in allowed_clusters
        for image_id in groups[cluster_id]["image_ids"]
    ]
    union = UnionFind(image_ids)
    auto_tp = 0
    auto_fp = 0
    candidate_tp = 0
    candidate_fp = 0
    compared_positive = 0
    compared_negative = 0
    for row in candidates:
        cluster_id = str(row["original_cluster_id"])
        if cluster_id not in allowed_clusters:
            continue
        key = tuple(sorted((row["first_image_id"], row["second_image_id"])))
        label = truth[key]
        if label:
            compared_positive += 1
        else:
            compared_negative += 1
        if float(row[config.score_field]) >= config.candidate_threshold:
            if label:
                candidate_tp += 1
            else:
                candidate_fp += 1
        if _automatic_edge(row, config):
            union.union(row["first_image_id"], row["second_image_id"])
            if label:
                auto_tp += 1
            else:
                auto_fp += 1
    positive_pairs = [key for key, label in truth.items() if label]
    negative_pairs = [key for key, label in truth.items() if not label]
    positive_pairs = [
        key
        for key in positive_pairs
        if next(
            cluster_id
            for cluster_id, group in groups.items()
            if key[0] in group["image_ids"]
        )
        in allowed_clusters
    ]
    negative_pairs = [
        key
        for key in negative_pairs
        if next(
            cluster_id
            for cluster_id, group in groups.items()
            if key[0] in group["image_ids"]
        )
        in allowed_clusters
    ]
    connected_positive = sum(union.find(a) == union.find(b) for a, b in positive_pairs)
    connected_negative = sum(union.find(a) == union.find(b) for a, b in negative_pairs)
    accepted_groups = [
        groups[key]
        for key in allowed_clusters
        if groups[key]["review_decision"] == "accepted"
    ]
    rejected_groups = [
        groups[key]
        for key in allowed_clusters
        if groups[key]["review_decision"] == "rejected"
    ]
    recovered = sum(
        len({union.find(image_id) for image_id in group["image_ids"]}) == 1
        for group in accepted_groups
    )
    separated = sum(
        len({union.find(image_id) for image_id in group["image_ids"]})
        == len(group["image_ids"])
        for group in rejected_groups
    )
    return {
        "split": split or "all",
        "accepted_groups": len(accepted_groups),
        "accepted_groups_recovered": recovered,
        "rejected_groups": len(rejected_groups),
        "rejected_groups_separated": separated,
        "positive_pairs": len(positive_pairs),
        "positive_pairs_connected": connected_positive,
        "negative_pairs": len(negative_pairs),
        "negative_pairs_connected": connected_negative,
        "automatic_edge_tp": auto_tp,
        "automatic_edge_fp": auto_fp,
        "candidate_tp": candidate_tp,
        "candidate_fp": candidate_fp,
        "compared_positive_pairs": compared_positive,
        "compared_negative_pairs": compared_negative,
        "perfect_group_fit": (
            recovered == len(accepted_groups) and separated == len(rejected_groups)
        ),
        "zero_false_links": auto_fp == 0 and connected_negative == 0,
    }


def _search_configs(
    candidates: Sequence[Mapping[str, Any]],
    prepared_dir: Path,
    *,
    fit_split: str | None,
    max_review_fraction: float,
) -> tuple[ReplayConfig, dict[str, Any]]:
    truth, groups, splits = _truth_maps(prepared_dir)
    allowed_clusters = {
        cluster_id
        for cluster_id in groups
        if fit_split is None or splits[cluster_id] == fit_split
    }
    relevant = [
        row for row in candidates if row["original_cluster_id"] in allowed_clusters
    ]
    if not relevant:
        raise ValueError(f"no candidate rows matched fit split {fit_split or 'all'}")
    gap_values: list[int | None] = [None]
    observed_gaps = sorted(
        {int(row["index_gap"]) for row in relevant if row.get("index_gap") is not None}
    )
    gap_values.extend(_compact_values(observed_gaps, limit=12))
    alignment_values: list[float | None] = [None]
    observed_alignment = [
        float(row["registration_alignment_score"])
        for row in relevant
        if row.get("registration_alignment_score") is not None
    ]
    alignment_values.extend(_compact_values(observed_alignment, limit=12))
    best: tuple[tuple[int, ...], ReplayConfig, dict[str, Any]] | None = None
    for field in AUTO_SCORE_FIELDS:
        candidate_threshold = _candidate_threshold(
            relevant,
            truth,
            field,
            allowed_clusters,
            max_review_fraction,
        )
        for auto_threshold in _score_values(relevant, field):
            for max_gap in gap_values:
                for min_alignment in alignment_values:
                    config = ReplayConfig(
                        score_field=field,
                        automatic_threshold=auto_threshold,
                        candidate_threshold=candidate_threshold,
                        max_index_gap=max_gap,
                        min_alignment_score=min_alignment,
                    )
                    metrics = replay_metrics(
                        candidates,
                        prepared_dir,
                        config,
                        split=fit_split,
                    )
                    safe = int(metrics["zero_false_links"])
                    objective = (
                        safe,
                        (
                            metrics["accepted_groups_recovered"]
                            if safe
                            else -metrics["automatic_edge_fp"]
                        ),
                        (
                            metrics["positive_pairs_connected"]
                            if safe
                            else -metrics["negative_pairs_connected"]
                        ),
                        metrics["automatic_edge_tp"] if safe else 0,
                        -metrics["automatic_edge_fp"],
                        -metrics["negative_pairs_connected"],
                        int(max_gap is None),
                        int(min_alignment is None),
                        int(auto_threshold * 1_000_000_000),
                    )
                    candidate = (objective, config, metrics)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
    assert best is not None
    return best[1], best[2]


def replay_predictions(
    candidates: Sequence[Mapping[str, Any]],
    config: ReplayConfig,
) -> list[dict[str, Any]]:
    output = []
    for row in candidates:
        p_same = float(row["same_document_probability"])
        q = float(row["occluded_given_same_probability"])
        output.append(
            {
                **dict(row),
                "same_document": bool(row["deterministic_same_document"]),
                "automatic_link_eligible": (
                    bool(row["automatic_link_eligible"])
                    and float(row[config.score_field]) >= config.automatic_threshold
                    and (
                        config.max_index_gap is None
                        or row.get("index_gap") is None
                        or int(row["index_gap"]) <= config.max_index_gap
                    )
                    and (
                        config.min_alignment_score is None
                        or (
                            row.get("registration_alignment_score") is not None
                            and float(row["registration_alignment_score"])
                            >= config.min_alignment_score
                        )
                    )
                ),
                "occlusion_candidate_flag": (
                    float(row[config.score_field]) >= config.candidate_threshold
                ),
                "same_clean_probability": p_same * (1 - q),
                "same_occluded_probability": p_same * q,
                "different_document_probability": 1 - p_same,
                "probability_model_version": "reviewed-candidate-score-replay-v1",
                "replay_score_field": config.score_field,
                "replay_automatic_threshold": config.automatic_threshold,
                "replay_candidate_threshold": config.candidate_threshold,
                "reason": row.get("decision_reason", "candidate_score_replay"),
            }
        )
    return output


def tune_candidate_replay(
    assignments_csv: Path,
    candidates_csv: Path,
    prepared_dir: Path,
    output_dir: Path,
    *,
    max_review_fraction: float = 1.0,
) -> dict[str, Any]:
    if not 0 < max_review_fraction <= 1:
        raise ValueError("max_review_fraction must be in (0, 1]")
    _, candidates, ingestion = load_reviewed_candidate_csv(
        assignments_csv, candidates_csv
    )
    selection_config, selection_fit = _search_configs(
        candidates,
        prepared_dir,
        fit_split="selection",
        max_review_fraction=max_review_fraction,
    )
    overfit_config, overfit_fit = _search_configs(
        candidates,
        prepared_dir,
        fit_split=None,
        max_review_fraction=max_review_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "matched_reviewed_candidates.csv", candidates)
    selection_predictions = replay_predictions(candidates, selection_config)
    overfit_predictions = replay_predictions(candidates, overfit_config)
    write_jsonl(output_dir / "selection_tuned_predictions.jsonl", selection_predictions)
    write_jsonl(
        output_dir / "full_dataset_overfit_predictions.jsonl",
        overfit_predictions,
    )
    summary = {
        "schema_version": "1.0",
        "ingestion": ingestion,
        "selection_tuned": {
            "fit_scope": "selection split only",
            "config": selection_config.as_dict(),
            "fit_metrics": selection_fit,
            "all_metrics": replay_metrics(
                candidates, prepared_dir, selection_config, split=None
            ),
            "locked_audit_metrics": replay_metrics(
                candidates, prepared_dir, selection_config, split="locked_audit"
            ),
            "promotion_eligible": True,
        },
        "full_dataset_overfit_diagnostic": {
            "fit_scope": "all 200 reviewed groups",
            "config": overfit_config.as_dict(),
            "fit_metrics": overfit_fit,
            "promotion_eligible": False,
            "warning": (
                "Diagnostic upper bound only; every reviewed label was used to "
                "select these parameters."
            ),
        },
    }
    (output_dir / "candidate_replay_tuning.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


__all__ = [
    "ReplayConfig",
    "load_reviewed_candidate_csv",
    "manifest_identity_map",
    "replay_metrics",
    "replay_predictions",
    "tune_candidate_replay",
]
