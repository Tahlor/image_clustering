from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ImageRecord:
    index: int
    path: Path
    folder: Path
    name: str


@dataclass
class WorkImage:
    record: ImageRecord
    gray: np.ndarray
    scale: float
    fingerprint: np.ndarray


@dataclass
class FeatureSet:
    keypoints: list[Any]
    descriptors: np.ndarray | None


@dataclass
class RegistrationResult:
    accepted: bool
    model: str | None = None
    matrix: np.ndarray | None = None
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    good_match_count: int = 0
    overlap_fraction: float = 0.0
    stable_residual: float = 1.0
    fingerprint_correlation: float = -1.0
    reason: str | None = None


@dataclass
class PairEdge:
    a: int
    b: int
    registration_b_to_a: RegistrationResult
    residual_tiles: np.ndarray | None = None
    valid_tiles: np.ndarray | None = None
    tile_boxes: list[BBox] = field(default_factory=list)


@dataclass
class ChangeCandidate:
    pair: tuple[int, int]
    side: str
    bbox_anchor: BBox
    area_fraction: float
    inside_score: float
    outside_score: float
    contrast: float
    support_fraction: float
    confidence: float


@dataclass
class Submission:
    submission_id: str
    group_id: str
    image_index: int
    source_path: Path
    kind: str
    side: str
    bbox: BBox
    completeness: str
    confidence: float
    content_score: float
    occlusion_fraction: float
    reason: str
    crop_path: Path | None = None
