"""Reusable calibration, review-budget, and split metric helpers."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return sum(rows) / len(rows) if rows else None


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def safe_log_loss(truth: int, probability: float) -> float:
    value = min(1 - 1e-15, max(1e-15, probability))
    return -(truth * math.log(value) + (1 - truth) * math.log(1 - value))


def calibration_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
    truth_key: str,
    bins: int = 10,
) -> list[dict[str, Any]]:
    output = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [
            row
            for row in rows
            if low <= float(row[probability_key]) < high
            or (index == bins - 1 and float(row[probability_key]) == 1.0)
        ]
        if not members:
            continue
        predicted = mean(float(row[probability_key]) for row in members)
        observed = mean(int(row[truth_key]) for row in members)
        output.append(
            {
                "bin": index,
                "low": low,
                "high": high,
                "count": len(members),
                "mean_probability": predicted,
                "observed_rate": observed,
                "absolute_gap": abs(float(predicted) - float(observed)),
            }
        )
    return output


def ece(table: Sequence[Mapping[str, Any]]) -> float:
    total = sum(int(row["count"]) for row in table)
    return (
        sum(int(row["count"]) * float(row["absolute_gap"]) for row in table)
        / total
        if total
        else 0.0
    )


def binary_probability_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    probability_key: str,
    truth_key: str,
) -> dict[str, Any] | None:
    if not rows:
        return None
    reliability = calibration_table(
        rows,
        probability_key=probability_key,
        truth_key=truth_key,
    )
    return {
        "count": len(rows),
        "log_loss": mean(
            safe_log_loss(int(row[truth_key]), float(row[probability_key]))
            for row in rows
        ),
        "brier_score": mean(
            (float(row[probability_key]) - int(row[truth_key])) ** 2
            for row in rows
        ),
        "expected_calibration_error": ece(reliability),
        "reliability_table": reliability,
    }


def review_budget_curve(
    pair_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ranked = sorted(
        pair_rows,
        key=lambda row: (
            float(row["same_occluded_probability"]),
            float(row["same_document_probability"]),
            row["pair_id"],
        ),
        reverse=True,
    )
    positive_total = sum(int(row["truth_same_document"]) for row in ranked)
    output = []
    for share in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.25, 0.50, 1.0):
        count = min(len(ranked), math.ceil(len(ranked) * share))
        selected = ranked[:count]
        caught = sum(int(row["truth_same_document"]) for row in selected)
        output.append(
            {
                "review_fraction": share,
                "review_pairs": count,
                "positive_pairs_caught": caught,
                "same_document_recall": fraction(caught, positive_total),
                "precision": fraction(caught, count),
            }
        )
    return output


def split_metrics(
    pair_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    positive = [row for row in pair_rows if row["truth_same_document"]]
    negative = [row for row in pair_rows if not row["truth_same_document"]]
    accepted = [row for row in group_rows if row["review_decision"] == "accepted"]
    rejected = [row for row in group_rows if row["review_decision"] == "rejected"]
    contaminated = [row for row in rejected if row["group_status"] == "contaminated"]
    candidate_tp = sum(bool(row["candidate_flag"]) for row in positive)
    candidate_fp = sum(bool(row["candidate_flag"]) for row in negative)
    auto_tp = sum(bool(row["automatic_edge"]) for row in positive)
    auto_fp = sum(bool(row["automatic_edge"]) for row in negative)
    return {
        "pair_count": len(pair_rows),
        "group_count": len(group_rows),
        "accepted_group_recovery": fraction(
            sum(row["group_status"] == "recovered" for row in accepted),
            len(accepted),
        ),
        "rejected_group_separation": fraction(
            sum(row["group_status"] == "separated" for row in rejected),
            len(rejected),
        ),
        "contaminated_component_count": len(contaminated),
        "same_document_recall_connected": fraction(
            sum(bool(row["same_component"]) for row in positive), len(positive)
        ),
        "candidate_recall": fraction(candidate_tp, len(positive)),
        "candidate_precision": fraction(candidate_tp, candidate_tp + candidate_fp),
        "automatic_link_recall": fraction(auto_tp, len(positive)),
        "automatic_link_precision": fraction(auto_tp, auto_tp + auto_fp),
        "negative_false_link_rate": fraction(auto_fp, len(negative)),
        "probability": binary_probability_metrics(
            pair_rows,
            probability_key="same_document_probability",
            truth_key="truth_same_document",
        ),
        "promotion_gates": {
            "zero_reviewed_negative_automatic_edges": auto_fp == 0,
            "zero_contaminated_reviewed_components": not contaminated,
            "passed": auto_fp == 0 and not contaminated,
        },
    }
