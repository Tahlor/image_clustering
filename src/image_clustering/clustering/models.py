"""Public and internal models for document-view clustering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True)
class ImageItem:
    """One ordered source image in an independent sequence."""

    image_id: str
    path: Path
    sequence_id: str
    sequence_index: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the image to a JSON-serializable dictionary."""
        return {
            "image_id": self.image_id,
            "path": str(self.path),
            "sequence_id": self.sequence_id,
            "sequence_index": self.sequence_index,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ImageItem":
        """Construct an image item from serialized data."""
        return cls(
            image_id=value["image_id"],
            path=Path(value["path"]),
            sequence_id=value["sequence_id"],
            sequence_index=int(value["sequence_index"]),
        )


@dataclass(frozen=True)
class PairComparison:
    """Decision and diagnostics for one nearby image pair.

    `transform`, when present, is a source-pixel 3×3 transform mapping points
    in `second_image_id` into the coordinate system of `first_image_id`.
    """

    first_image_id: str
    second_image_id: str
    sequence_id: str
    index_gap: int
    same_document: bool
    confidence: float
    reason: str
    registration_model: str | None = None
    transform: Matrix3x3 | None = None
    good_match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    feature_overlap: float = 0.0
    median_reprojection_error: float = 0.0
    valid_fraction: float = 0.0
    changed_fraction: float = 1.0
    stable_fraction: float = 0.0
    tiles_changed_fraction: float = 1.0
    largest_change_share: float = 0.0
    branch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the comparison to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PairComparison":
        """Construct a pair comparison from serialized data."""
        transform = value.get("transform")
        normalized_transform = None
        if transform is not None:
            normalized_transform = tuple(
                tuple(float(v) for v in row) for row in transform
            )
        return cls(**{**value, "transform": normalized_transform})


@dataclass(frozen=True)
class ImageCluster:
    """A complete conservative component of repeated document captures."""

    cluster_id: str
    sequence_id: str
    image_ids: tuple[str, ...]
    representative_image_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert the cluster to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ImageCluster":
        """Construct a cluster from serialized data."""
        return cls(
            cluster_id=value["cluster_id"],
            sequence_id=value["sequence_id"],
            image_ids=tuple(value["image_ids"]),
            representative_image_id=value["representative_image_id"],
        )


@dataclass(frozen=True)
class ClusteringResult:
    """Complete clustering output and downstream handoff contract."""

    config_fingerprint: str
    images: tuple[ImageItem, ...]
    clusters: tuple[ImageCluster, ...]
    comparisons: tuple[PairComparison, ...]
    input_root: Path | None = None

    def image(self, image_id: str) -> ImageItem:
        """Return one image by stable image identifier."""
        for image in self.images:
            if image.image_id == image_id:
                return image
        raise KeyError(f"Unknown image_id: {image_id}")

    def cluster(self, cluster_id: str) -> ImageCluster:
        """Return one cluster by identifier."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        raise KeyError(f"Unknown cluster_id: {cluster_id}")

    def images_for(self, cluster_id: str) -> tuple[ImageItem, ...]:
        """Return source images for a cluster in sequence order."""
        cluster = self.cluster(cluster_id=cluster_id)
        return tuple(self.image(image_id) for image_id in cluster.image_ids)

    def accepted_comparisons(self, cluster_id: str) -> tuple[PairComparison, ...]:
        """Return accepted direct registrations internal to a cluster."""
        cluster = self.cluster(cluster_id=cluster_id)
        image_ids = set(cluster.image_ids)
        return tuple(
            comparison
            for comparison in self.comparisons
            if comparison.same_document
            and comparison.first_image_id in image_ids
            and comparison.second_image_id in image_ids
        )

    def comparison(
        self,
        first_image_id: str,
        second_image_id: str,
    ) -> PairComparison:
        """Return a direct candidate comparison in either requested order."""
        requested = {first_image_id, second_image_id}
        for comparison in self.comparisons:
            if {
                comparison.first_image_id,
                comparison.second_image_id,
            } == requested:
                return comparison
        raise KeyError(
            f"No direct comparison for {first_image_id!r} and {second_image_id!r}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete result to a JSON-serializable dictionary."""
        return {
            "schema_version": 1,
            "config_fingerprint": self.config_fingerprint,
            "input_root": str(self.input_root) if self.input_root is not None else None,
            "images": [image.to_dict() for image in self.images],
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "comparisons": [
                comparison.to_dict() for comparison in self.comparisons
            ],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ClusteringResult":
        """Construct a clustering result from serialized data."""
        if value.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported schema_version: {value.get('schema_version')}"
            )
        input_root = value.get("input_root")
        return cls(
            config_fingerprint=value["config_fingerprint"],
            input_root=Path(input_root) if input_root is not None else None,
            images=tuple(ImageItem.from_dict(item) for item in value["images"]),
            clusters=tuple(
                ImageCluster.from_dict(item) for item in value["clusters"]
            ),
            comparisons=tuple(
                PairComparison.from_dict(item) for item in value["comparisons"]
            ),
        )


@dataclass
class ImageFeatures:
    """Working image and local features for one source image."""

    image: ImageItem
    gray: np.ndarray
    scale: float
    keypoints_xy: np.ndarray
    descriptors: np.ndarray


@dataclass
class Registration:
    """Estimated mapping from the current working image into the previous one."""

    accepted: bool
    model: str | None = None
    matrix: np.ndarray | None = None
    good_match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    feature_overlap: float = 0.0
    median_reprojection_error: float = 0.0
    occupied_grid_cells: int = 0
    x_span: float = 0.0
    y_span: float = 0.0
    reason: str | None = None
