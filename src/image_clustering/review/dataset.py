"""Read a completed evaluation directory into a reviewable dataset model.

Nothing here writes to the canonical clustering, cropping, or report artifacts.
Clusters are ordered weakest-confidence first so the review queue starts with
the groupings most likely to be wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_clustering.clustering.models import ClusteringResult
from image_clustering.clustering.serialization import load_result


@dataclass(frozen=True)
class ReviewBox:
    """One automatic crop box in original source pixel coordinates."""

    submission_id: str
    bbox: tuple[int, int, int, int]
    kind: str
    completeness: str
    confidence: float
    crop_path: Path | None
    review_required: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert the box to a JSON-serializable dictionary."""
        return {
            "box_id": self.submission_id,
            "submission_id": self.submission_id,
            "bbox": list(self.bbox),
            "kind": self.kind,
            "completeness": self.completeness,
            "confidence": self.confidence,
            "crop_path": str(self.crop_path) if self.crop_path else None,
            "review_required": self.review_required,
            "origin": "automatic",
        }


@dataclass(frozen=True)
class ReviewImage:
    """One cluster member capture with its automatic boxes."""

    image_id: str
    filename: str
    source_path: Path
    sequence_index: int
    width: int | None
    height: int | None
    boxes: tuple[ReviewBox, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the image to a JSON-serializable dictionary."""
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "source_path": str(self.source_path),
            "sequence_index": self.sequence_index,
            "width": self.width,
            "height": self.height,
            "boxes": [box.to_dict() for box in self.boxes],
        }


@dataclass(frozen=True)
class ReviewCluster:
    """One automatic cluster presented as a single review unit."""

    cluster_id: str
    source_folder: str
    images: tuple[ReviewImage, ...]
    minimum_confidence: float | None
    mean_confidence: float | None
    largest_gap: int
    review_reasons: tuple[str, ...]

    @property
    def image_count(self) -> int:
        """Return the number of member captures."""
        return len(self.images)

    def to_dict(self) -> dict[str, Any]:
        """Convert the cluster to a JSON-serializable dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "source_folder": self.source_folder,
            "image_count": len(self.images),
            "minimum_confidence": self.minimum_confidence,
            "mean_confidence": self.mean_confidence,
            "largest_gap": self.largest_gap,
            "review_reasons": list(self.review_reasons),
            "box_count": sum(len(image.boxes) for image in self.images),
            "images": [image.to_dict() for image in self.images],
        }


@dataclass(frozen=True)
class ReviewDataset:
    """All clusters available for review plus run provenance."""

    output_root: Path
    input_root: Path | None
    config_fingerprint: str
    clusters: tuple[ReviewCluster, ...]

    def cluster(self, cluster_id: str) -> ReviewCluster:
        """Return one cluster by identifier."""
        for cluster in self.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        raise KeyError(f"Unknown cluster_id: {cluster_id}")

    def image(self, cluster_id: str, image_id: str) -> ReviewImage:
        """Return one member image of a cluster."""
        for image in self.cluster(cluster_id).images:
            if image.image_id == image_id:
                return image
        raise KeyError(f"Unknown image_id {image_id!r} in {cluster_id!r}")

    @property
    def provenance(self) -> dict[str, Any]:
        """Return provenance recorded alongside reviewer decisions."""
        return {
            "output_root": str(self.output_root),
            "input_root": str(self.input_root) if self.input_root else None,
            "clustering_config_fingerprint": self.config_fingerprint,
            "cluster_count": len(self.clusters),
            "image_count": sum(cluster.image_count for cluster in self.clusters),
            "review_tool_revision": "review-tool-v1",
        }

    @property
    def source_paths(self) -> tuple[Path, ...]:
        """Return every member source path in the dataset."""
        return tuple(
            image.source_path
            for cluster in self.clusters
            for image in cluster.images
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_dimensions(output_root: Path) -> dict[str, tuple[int | None, int | None]]:
    inventory = output_root / "inventory" / "dataset_inventory.json"
    if not inventory.is_file():
        return {}
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    dimensions: dict[str, tuple[int | None, int | None]] = {}
    for row in payload.get("images", []):
        key = str(Path(row["absolute_source_path"]).resolve())
        dimensions[key] = (row.get("width"), row.get("height"))
    return dimensions


def _boxes_by_source(output_root: Path) -> dict[str, list[ReviewBox]]:
    boxes: dict[str, list[ReviewBox]] = {}
    rows = _load_jsonl(output_root / "reports" / "crops_for_recognizer.jsonl")
    for row in rows:
        key = str(Path(row["source_image_path"]).resolve())
        crop_path = row.get("crop_path")
        boxes.setdefault(key, []).append(
            ReviewBox(
                submission_id=str(row["submission_id"]),
                bbox=tuple(int(value) for value in row["bbox"]),
                kind=str(row.get("kind") or "base_page"),
                completeness=str(row.get("completeness") or "unknown"),
                confidence=float(row.get("confidence") or 0.0),
                crop_path=Path(crop_path) if crop_path else None,
                review_required=bool(row.get("review_required")),
            )
        )
    return boxes


def _cluster_review_reasons(
    clustering: ClusteringResult,
    cluster_id: str,
    accepted_confidences: list[float],
    member_count: int,
    largest_gap: int,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if member_count == 1:
        reasons.append("singleton has no registration evidence")
    if accepted_confidences and min(accepted_confidences) < 0.55:
        reasons.append("low-confidence accepted registration")
    if largest_gap > 1:
        reasons.append("nonadjacent relationship")
    cluster = clustering.cluster(cluster_id)
    member_ids = set(cluster.image_ids)
    if any(
        not comparison.same_document
        and comparison.first_image_id in member_ids
        and comparison.second_image_id in member_ids
        for comparison in clustering.comparisons
    ):
        reasons.append("a direct pair failed but the component is connected")
    return tuple(reasons)


def build_review_dataset(
    output_root: Path,
    clustering: ClusteringResult | None = None,
) -> ReviewDataset:
    """Build the review model from a completed evaluation output directory."""
    output_root = Path(output_root).resolve()
    if clustering is None:
        clustering_json = output_root / "clustering" / "clustering.json"
        if not clustering_json.is_file():
            raise FileNotFoundError(
                f"No clustering output to review at {clustering_json}"
            )
        clustering = load_result(clustering_json)
    dimensions = _load_dimensions(output_root)
    boxes_by_source = _boxes_by_source(output_root)
    clusters: list[ReviewCluster] = []
    for cluster in clustering.clusters:
        members = clustering.images_for(cluster.cluster_id)
        accepted = clustering.accepted_comparisons(cluster.cluster_id)
        confidences = [comparison.confidence for comparison in accepted]
        largest_gap = max((comparison.index_gap for comparison in accepted), default=0)
        images: list[ReviewImage] = []
        for member in members:
            key = str(member.path.resolve())
            width, height = dimensions.get(key, (None, None))
            images.append(
                ReviewImage(
                    image_id=member.image_id,
                    filename=member.path.name,
                    source_path=member.path.resolve(),
                    sequence_index=member.sequence_index,
                    width=width,
                    height=height,
                    boxes=tuple(boxes_by_source.get(key, ())),
                )
            )
        clusters.append(
            ReviewCluster(
                cluster_id=cluster.cluster_id,
                source_folder=cluster.sequence_id,
                images=tuple(images),
                minimum_confidence=min(confidences) if confidences else None,
                mean_confidence=(
                    sum(confidences) / len(confidences) if confidences else None
                ),
                largest_gap=largest_gap,
                review_reasons=_cluster_review_reasons(
                    clustering,
                    cluster.cluster_id,
                    confidences,
                    len(members),
                    largest_gap,
                ),
            )
        )
    clusters.sort(key=review_sort_key)
    return ReviewDataset(
        output_root=output_root,
        input_root=clustering.input_root,
        config_fingerprint=clustering.config_fingerprint,
        clusters=tuple(clusters),
    )


def review_sort_key(cluster: ReviewCluster) -> tuple[float, str]:
    """Order clusters weakest-confidence first, then by identifier."""
    confidence = (
        cluster.minimum_confidence if cluster.minimum_confidence is not None else 2.0
    )
    return (confidence, cluster.cluster_id)


def dataset_payload(dataset: ReviewDataset) -> dict[str, Any]:
    """Return the JSON payload the review client consumes."""
    return {
        "schema_version": 1,
        "provenance": dataset.provenance,
        "defaults": {
            "minimum_cluster_size": 2,
            "sort": "confidence-asc",
        },
        "clusters": [cluster.to_dict() for cluster in dataset.clusters],
    }
