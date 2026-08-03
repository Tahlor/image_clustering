"""Leakage-safe grouped splits for the reviewed real dataset."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Mapping, Sequence

from image_clustering.evaluation.reviewed_models import (
    SPLIT_VERSION,
    ManifestRow,
    SplitAssignment,
    group_rows,
    stable_hash,
)


def make_grouped_splits(
    rows: Sequence[ManifestRow],
    *,
    split_version: str = SPLIT_VERSION,
    targets: Mapping[str, float] | None = None,
) -> list[SplitAssignment]:
    targets = dict(
        targets
        or {"development": 0.60, "selection": 0.20, "locked_audit": 0.20}
    )
    if not math.isclose(sum(targets.values()), 1.0):
        raise ValueError("split target fractions must sum to one")
    grouped = group_rows(rows)
    families: dict[str, list[list[ManifestRow]]] = defaultdict(list)
    for group in grouped.values():
        sequences = {row.sequence_id for row in group}
        if len(sequences) != 1:
            raise ValueError(f"{group[0].original_cluster_id} crosses sequences")
        families[group[0].sequence_id].append(group)

    splits = tuple(targets)
    truth_totals = Counter(group[0].review_decision for group in grouped.values())
    desired_total = {split: len(grouped) * targets[split] for split in splits}
    desired_truth = {
        split: {truth: truth_totals[truth] * targets[split] for truth in truth_totals}
        for split in splits
    }
    assigned_total = Counter()
    assigned_truth: dict[str, Counter[str]] = {split: Counter() for split in splits}
    output: list[SplitAssignment] = []
    ordered = sorted(
        families.items(),
        key=lambda item: (-len(item[1]), stable_hash(f"{split_version}:{item[0]}")),
    )
    for sequence_id, groups in ordered:
        family_truth = Counter(group[0].review_decision for group in groups)

        def cost(split: str) -> tuple[float, int]:
            total_after = assigned_total[split] + len(groups)
            total_error = abs(total_after - desired_total[split]) / max(
                desired_total[split], 1
            )
            truth_error = sum(
                abs(assigned_truth[split][truth] + count - desired_truth[split][truth])
                / max(desired_truth[split][truth], 1)
                for truth, count in family_truth.items()
            )
            return (
                total_error + truth_error,
                stable_hash(f"{sequence_id}:{split}"),
            )

        split = min(splits, key=cost)
        assigned_total[split] += len(groups)
        assigned_truth[split].update(family_truth)
        output.extend(
            SplitAssignment(
                original_cluster_id=group[0].original_cluster_id,
                split=split,
                review_decision=group[0].review_decision,
                cluster_size=len(group),
                sequence_id=sequence_id,
                split_version=split_version,
            )
            for group in groups
        )
    output.sort(key=lambda item: item.original_cluster_id)
    validate_split_leakage(rows, output)
    return output


def validate_split_leakage(
    rows: Sequence[ManifestRow],
    assignments: Sequence[SplitAssignment],
) -> None:
    by_cluster = {
        assignment.original_cluster_id: assignment.split
        for assignment in assignments
    }
    if set(by_cluster) != set(group_rows(rows)):
        raise ValueError("split does not cover every original cluster exactly once")
    sequence_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        sequence_splits[row.sequence_id].add(by_cluster[row.original_cluster_id])
    leaking = {key: value for key, value in sequence_splits.items() if len(value) > 1}
    if leaking:
        raise ValueError(f"sequence families leak across splits: {leaking}")
