from __future__ import annotations

import math
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from loaders import norm, pair_key
from models import Config, CvEvidence, Extraction, FIELD_WEIGHTS, Record

NUMERIC_FIELDS = {
    key for key in FIELD_WEIGHTS if key.endswith(("_day", "_year"))
} | {"petition_number"}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return min(high, max(low, value))


def similarity(field: str, first: str, second: str) -> float:
    left, right = norm(first), norm(second)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    if field in NUMERIC_FIELDS:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def record_identity(first: Record, second: Record) -> float:
    if first.role != second.role:
        return 0.0
    values = []
    for key, weight in (
        ("given_name", 1.1),
        ("surname", 1.8),
        ("petition_number", 2.2),
        ("birth_year", 0.8),
    ):
        left, right = first.fields.get(key), second.fields.get(key)
        if left and right:
            values.append((weight, similarity(key, left, right)))
    if not values:
        return 0.35 if first.role == second.role else 0.0
    return sum(weight * score for weight, score in values) / sum(
        weight for weight, _ in values
    )


def align_records(
    first: Sequence[Record],
    second: Sequence[Record],
) -> list[tuple[int, int, float]]:
    candidates = sorted(
        (
            (record_identity(left, right), left_index, right_index)
            for left_index, left in enumerate(first)
            for right_index, right in enumerate(second)
        ),
        reverse=True,
    )
    output = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for score, left_index, right_index in candidates:
        if score < 0.40 or left_index in used_left or right_index in used_right:
            continue
        output.append((left_index, right_index, score))
        used_left.add(left_index)
        used_right.add(right_index)
    return output


def pair_features(first: Extraction, second: Extraction) -> dict[str, float | int]:
    aligned = align_records(first.records, second.records)
    identity_scores = []
    anchor_scores = []
    event_scores = []
    matched_first = matched_second = comparable = conflicts = shared_non_name = 0.0
    for first_index, second_index, identity in aligned:
        left, right = first.records[first_index], second.records[second_index]
        identity_scores.append(identity)
        petition_left = norm(left.fields.get("petition_number"))
        petition_right = norm(right.fields.get("petition_number"))
        name_left = norm(
            f"{left.fields.get('given_name', '')} {left.fields.get('surname', '')}"
        )
        name_right = norm(
            f"{right.fields.get('given_name', '')} {right.fields.get('surname', '')}"
        )
        surname_left = norm(left.fields.get("surname"))
        surname_right = norm(right.fields.get("surname"))
        birth_left = norm(left.fields.get("birth_year"))
        birth_right = norm(right.fields.get("birth_year"))
        if petition_left and petition_left == petition_right:
            anchor_scores.append(1.0)
        elif name_left and name_left == name_right:
            anchor_scores.append(0.82)
        elif surname_left and surname_left == surname_right and birth_left == birth_right:
            anchor_scores.append(0.72)
        elif surname_left and surname_left == surname_right:
            anchor_scores.append(0.45)
        else:
            anchor_scores.append(0.0)
        if left.event_type and right.event_type:
            event_scores.append(float(left.event_type == right.event_type))
        for key in set(left.fields) | set(right.fields):
            value_left, value_right = left.fields.get(key), right.fields.get(key)
            if not value_left or not value_right:
                continue
            weight = FIELD_WEIGHTS.get(key, 0.25)
            score = similarity(key, value_left, value_right)
            comparable += weight
            if score >= 0.86:
                matched_first += weight * score
                matched_second += weight * score
                if key not in {"given_name", "surname"} and not key.startswith("event_"):
                    shared_non_name += weight * score
            elif score <= 0.48:
                conflicts += weight
    mass_first, mass_second = first.field_mass, second.field_mass
    containment_first = matched_first / mass_first if mass_first else 0.0
    containment_second = matched_second / mass_second if mass_second else 0.0
    identity = max(identity_scores, default=0.0)
    anchor = max(anchor_scores, default=0.0)
    extraction_score = clamp(
        0.36 * max(containment_first, containment_second)
        + 0.19 * min(containment_first, containment_second)
        + 0.28 * identity
        + 0.17 * anchor
    )
    return {
        "extraction_score": extraction_score,
        "identity_similarity": identity,
        "anchor_score": anchor,
        "containment_a_in_b": containment_first,
        "containment_b_in_a": containment_second,
        "conflict_rate": conflicts / comparable if comparable else 0.0,
        "aligned_records": len(aligned),
        "field_mass_a": mass_first,
        "field_mass_b": mass_second,
        "record_count_a": len(first.records),
        "record_count_b": len(second.records),
        "event_type_agreement": sum(event_scores) / len(event_scores) if event_scores else 0.5,
        "shared_non_name_weight": shared_non_name,
    }


def quality_value(item: Extraction, key: str) -> float:
    return clamp(float(item.quality.get(key, 0.0)))


def completeness(item: Extraction) -> float:
    mass = 1 - math.exp(-item.field_mass / 12)
    records = 1 - math.exp(-len(item.records) / 3)
    indexability = item.quality.get(
        "indexability_score",
        item.quality.get("capture_quality_score", 0.5),
    )
    penalty = (
        0.45 * quality_value(item, "occlusion_risk")
        + 0.32 * quality_value(item, "crop_risk")
        + 0.18 * quality_value(item, "page_boundary_risk")
    )
    return clamp(0.58 * mass + 0.17 * records + 0.25 * clamp(float(indexability)) - penalty)


def cosine(first: np.ndarray | None, second: np.ndarray | None) -> float | None:
    if first is None or second is None:
        return None
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return float(np.dot(first, second) / denominator) if denominator else None


def candidate_pairs(
    items: Sequence[Extraction],
    max_gap: int,
) -> Iterator[tuple[Extraction, Extraction, int]]:
    grouped: dict[str, list[Extraction]] = defaultdict(list)
    for item in items:
        grouped[item.sequence_key].append(item)
    for group in grouped.values():
        group.sort(key=lambda item: item.sequence_index)
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                gap = right.sequence_index - left.sequence_index
                if gap > max_gap:
                    break
                yield left, right, gap


def decide(
    first: Extraction,
    second: Extraction,
    gap: int,
    config: Config,
    embeddings: Mapping[str, np.ndarray],
    cv_rows: Mapping[tuple[str, str], CvEvidence],
) -> dict[str, Any]:
    features = pair_features(first, second)
    embedding = cosine(embeddings.get(first.image_id), embeddings.get(second.image_id))
    cv = cv_rows.get(pair_key(first.image_id, second.image_id), CvEvidence())
    available = [(config.extraction_weight, float(features["extraction_score"]))]
    if embedding is not None:
        available.append((config.embedding_weight, max(0.0, embedding)))
    if cv.same_scene_probability is not None:
        available.append((config.cv_weight, clamp(cv.same_scene_probability)))
    score = sum(weight * value for weight, value in available) / sum(
        weight for weight, _ in available
    )
    score = clamp(
        score
        + 0.04 * (1 - (gap - 1) / max(1, config.max_gap))
        + 0.08 * float(features["anchor_score"])
        - config.conflict_penalty * float(features["conflict_rate"])
    )
    containment = max(
        float(features["containment_a_in_b"]),
        float(features["containment_b_in_a"]),
    )
    extraction_gate = (
        containment >= config.min_extraction_containment
        and float(features["identity_similarity"]) >= config.min_identity_similarity
        and float(features["conflict_rate"]) <= config.max_conflict_rate
    )
    auxiliary = bool(
        (
            embedding is not None
            and embedding >= config.strong_embedding_similarity
            and float(features["identity_similarity"]) >= 0.30
        )
        or (
            cv.same_scene_probability is not None
            and cv.same_scene_probability >= config.strong_cv_same_scene
        )
    )
    possible_same = (
        score >= config.review_link_threshold
        and float(features["conflict_rate"]) <= config.max_conflict_rate
        and (extraction_gate or auxiliary)
    )
    document_support = (
        float(features["event_type_agreement"]) >= 0.5
        or float(features["shared_non_name_weight"]) >= 0.5
        or auxiliary
    )
    tier = (
        "automatic"
        if possible_same and score >= config.automatic_link_threshold and document_support
        else "review"
        if possible_same
        else "rejected"
    )
    complete_first, complete_second = completeness(first), completeness(second)
    mass_asymmetry = abs(first.field_mass - second.field_mass) / max(
        first.field_mass, second.field_mass, 1.0
    )
    record_asymmetry = abs(len(first.records) - len(second.records)) / max(
        len(first.records), len(second.records), 1
    )
    quality_asymmetry = max(
        abs(quality_value(first, "occlusion_risk") - quality_value(second, "occlusion_risk")),
        abs(quality_value(first, "crop_risk") - quality_value(second, "crop_risk")),
    )
    asymmetry = clamp(
        max(
            mass_asymmetry,
            record_asymmetry,
            quality_asymmetry,
            abs(complete_first - complete_second),
            cv.occlusion_probability or 0.0,
        )
    )
    possible_occlusion = (
        possible_same
        and document_support
        and (
            asymmetry >= config.min_occlusion_asymmetry
            or (cv.occlusion_probability or 0.0) >= 0.5
            or norm(cv.relation) == "occlusion"
        )
    )
    occluded = more_complete = None
    role_confidence = 0.0
    if possible_occlusion and abs(complete_first - complete_second) >= config.min_role_margin:
        occluded, more_complete = (
            (first.image_id, second.image_id)
            if complete_first < complete_second
            else (second.image_id, first.image_id)
        )
        role_confidence = clamp(abs(complete_first - complete_second) / 0.35)
    elif possible_occlusion:
        risk_first = quality_value(first, "occlusion_risk") + quality_value(first, "crop_risk")
        risk_second = quality_value(second, "occlusion_risk") + quality_value(second, "crop_risk")
        if abs(risk_first - risk_second) >= config.min_role_margin:
            occluded, more_complete = (
                (first.image_id, second.image_id)
                if risk_first > risk_second
                else (second.image_id, first.image_id)
            )
            role_confidence = clamp(abs(risk_first - risk_second) / 0.5)
    reasons = []
    if containment >= config.min_extraction_containment:
        reasons.append("extraction_containment")
    if float(features["anchor_score"]) >= 0.7:
        reasons.append("strong_identity_anchor")
    if embedding is not None and embedding >= config.strong_embedding_similarity:
        reasons.append("strong_embedding_similarity")
    if (cv.same_scene_probability or 0.0) >= config.strong_cv_same_scene:
        reasons.append("cv_same_scene_support")
    if (cv.occlusion_probability or 0.0) >= 0.5:
        reasons.append("cv_occlusion_support")
    if mass_asymmetry >= config.min_occlusion_asymmetry:
        reasons.append("extraction_completeness_asymmetry")
    if quality_asymmetry >= config.min_occlusion_asymmetry:
        reasons.append("quality_risk_asymmetry")
    features.update(
        embedding_similarity=embedding,
        cv_same_scene_probability=cv.same_scene_probability,
        cv_occlusion_probability=cv.occlusion_probability,
        quality_occlusion_a=quality_value(first, "occlusion_risk"),
        quality_occlusion_b=quality_value(second, "occlusion_risk"),
        quality_crop_a=quality_value(first, "crop_risk"),
        quality_crop_b=quality_value(second, "crop_risk"),
        completeness_score_a=complete_first,
        completeness_score_b=complete_second,
        asymmetry=asymmetry,
    )
    return {
        "schema_version": "1.0",
        "image_a": first.image_id,
        "image_b": second.image_id,
        "source_filename_a": first.source_filename,
        "source_filename_b": second.source_filename,
        "sequence_key": first.sequence_key,
        "gap": gap,
        "signals": features,
        "same_document_score": score,
        "link_tier": tier,
        "possible_same_document": possible_same,
        "possible_occlusion_event": possible_occlusion,
        "likely_occluded_image_id": occluded,
        "likely_more_complete_image_id": more_complete,
        "role_confidence": role_confidence,
        "reasons": reasons or ["weak_or_insufficient_evidence"],
    }
