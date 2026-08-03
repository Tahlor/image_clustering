"""Truth models and manifest readers for the reviewed real dataset."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
SPLIT_VERSION = "reviewed-real-v1"
ALLOWED_SUBTYPES = {
    "same_clean",
    "same_occluded",
    "mixed_or_multi_state",
    "uncertain_occlusion_subtype",
}


@dataclass(frozen=True)
class DatasetContract:
    rows: int = 422
    original_clusters: int = 200
    accepted_clusters: int = 134
    rejected_clusters: int = 66
    accepted_images: int = 282
    rejected_images: int = 140
    positive_pairs: int = 183
    negative_pairs: int = 83
    accepted_size_distribution: tuple[tuple[int, int], ...] = (
        (2, 126),
        (3, 7),
        (9, 1),
    )
    rejected_size_distribution: tuple[tuple[int, int], ...] = (
        (2, 59),
        (3, 6),
        (4, 1),
    )


CONTRACT = DatasetContract()


@dataclass(frozen=True)
class ManifestRow:
    index: int
    image_id: str
    source_path: str
    source_project: str
    sequence_id: str
    sequence_index: int
    review_decision: str
    cluster_status: str
    assignment_id: str
    assignment_type: str
    original_cluster_id: str
    included_in_original_cluster: bool
    reviewed_at: str
    package_relative_path: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> ManifestRow:
        missing = [name for name in cls.__dataclass_fields__ if name not in row]
        if missing:
            raise ValueError(f"Manifest row is missing columns: {', '.join(missing)}")
        return cls(
            index=int(row["index"]),
            image_id=str(row["image_id"]).strip(),
            source_path=str(row["source_path"]),
            source_project=str(row["source_project"]).strip(),
            sequence_id=str(row["sequence_id"]).strip(),
            sequence_index=int(row["sequence_index"]),
            review_decision=str(row["review_decision"]).strip().lower(),
            cluster_status=str(row["cluster_status"]).strip().lower(),
            assignment_id=str(row["assignment_id"]).strip(),
            assignment_type=str(row["assignment_type"]).strip().lower(),
            original_cluster_id=str(row["original_cluster_id"]).strip(),
            included_in_original_cluster=parse_bool(
                row["included_in_original_cluster"]
            ),
            reviewed_at=str(row["reviewed_at"]).strip(),
            package_relative_path=str(row["package_relative_path"]).strip(),
        )

    def normalized(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairTruth:
    pair_id: str
    original_cluster_id: str
    truth: str
    image_a: str
    image_b: str
    assignment_id_a: str
    assignment_id_b: str
    sequence_id: str
    sequence_index_a: int
    sequence_index_b: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SplitAssignment:
    original_cluster_id: str
    split: str
    review_decision: str
    cluster_size: int
    sequence_id: str
    split_version: str = SPLIT_VERSION

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def pair_id(first: str, second: str) -> str:
    left, right = sorted((first, second))
    return hashlib.sha256(f"{left}\0{right}".encode()).hexdigest()[:20]


def load_csv_manifest(path: Path) -> list[ManifestRow]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [ManifestRow.from_mapping(row) for row in csv.DictReader(handle)]


def load_jsonl_manifest(path: Path) -> list[ManifestRow]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(ManifestRow.from_mapping(json.loads(line)))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL line {number} in {path}") from error
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL line {number} in {path}") from error
    return rows


def group_rows(rows: Sequence[ManifestRow]) -> dict[str, list[ManifestRow]]:
    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[row.original_cluster_id].append(row)
    return {
        key: sorted(value, key=lambda row: (row.sequence_index, row.image_id))
        for key, value in grouped.items()
    }


def size_distribution(groups: Iterable[Sequence[ManifestRow]]) -> dict[int, int]:
    return dict(sorted(Counter(len(group) for group in groups).items()))


def expand_pairs(
    rows: Sequence[ManifestRow],
) -> tuple[list[PairTruth], list[PairTruth]]:
    positive: list[PairTruth] = []
    negative: list[PairTruth] = []
    for cluster_id, group in group_rows(rows).items():
        target = positive if group[0].review_decision == "accepted" else negative
        for first, second in itertools.combinations(group, 2):
            target.append(
                PairTruth(
                    pair_id=pair_id(first.image_id, second.image_id),
                    original_cluster_id=cluster_id,
                    truth=first.review_decision,
                    image_a=first.image_id,
                    image_b=second.image_id,
                    assignment_id_a=first.assignment_id,
                    assignment_id_b=second.assignment_id,
                    sequence_id=first.sequence_id,
                    sequence_index_a=first.sequence_index,
                    sequence_index_b=second.sequence_index,
                )
            )
    return positive, negative
