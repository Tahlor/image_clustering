"""Reviewer decision state: membership, dissolution, and bbox approval.

State is a plain JSON-serializable dictionary so it can be persisted and
inspected without this package. Every mutation returns the affected cluster
state so callers can echo it back to a client.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from image_clustering.review.dataset import ReviewDataset

SCHEMA_VERSION = 1

STATUS_UNREVIEWED = "unreviewed"
STATUS_APPROVED = "approved"
STATUS_EDITED = "edited"
STATUS_DISSOLVED = "dissolved"
STATUS_IRREGULAR = "irregular"

CLUSTER_STATUSES = (
    STATUS_UNREVIEWED,
    STATUS_APPROVED,
    STATUS_EDITED,
    STATUS_DISSOLVED,
    STATUS_IRREGULAR,
)

BBOX_STATUS_UNREVIEWED = "unreviewed"
BBOX_STATUS_APPROVED = "approved"
BBOX_STATUS_EDITED = "edited"

MINIMUM_CLUSTER_SIZE = 2


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def empty_state(provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an empty reviewer decision document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": dict(provenance or {}),
        "clusters": {},
    }


def cluster_state(state: dict[str, Any], cluster_id: str) -> dict[str, Any]:
    """Return the mutable decision record for one cluster, creating it if absent."""
    clusters = state.setdefault("clusters", {})
    record = clusters.get(cluster_id)
    if not isinstance(record, dict):
        record = {
            "cluster_id": cluster_id,
            "status": STATUS_UNREVIEWED,
            "dissolved": False,
            "excluded_image_ids": [],
            "images": {},
            "updated_at": None,
        }
        clusters[cluster_id] = record
    record.setdefault("cluster_id", cluster_id)
    record.setdefault("status", STATUS_UNREVIEWED)
    record.setdefault("dissolved", False)
    record.setdefault("excluded_image_ids", [])
    if not isinstance(record.get("images"), dict):
        record["images"] = {}
    return record


def image_state(
    state: dict[str, Any],
    cluster_id: str,
    image_id: str,
) -> dict[str, Any]:
    """Return the mutable decision record for one cluster member image."""
    cluster = cluster_state(state, cluster_id)
    record = cluster["images"].get(image_id)
    if not isinstance(record, dict):
        record = {
            "image_id": image_id,
            "included": True,
            "bbox_status": BBOX_STATUS_UNREVIEWED,
            "boxes": None,
        }
        cluster["images"][image_id] = record
    record.setdefault("image_id", image_id)
    record.setdefault("included", True)
    record.setdefault("bbox_status", BBOX_STATUS_UNREVIEWED)
    record.setdefault("boxes", None)
    return record


def included_image_ids(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
) -> list[str]:
    """Return member image ids the reviewer still considers part of the cluster."""
    cluster = cluster_state(state, cluster_id)
    excluded = set(cluster.get("excluded_image_ids") or ())
    return [
        image.image_id
        for image in dataset.cluster(cluster_id).images
        if image.image_id not in excluded
    ]


def _refresh_membership(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
) -> dict[str, Any]:
    """Recompute exclusions, dissolution, and reviewed status for one cluster."""
    cluster = cluster_state(state, cluster_id)
    members = dataset.cluster(cluster_id).images
    excluded = [
        image.image_id
        for image in members
        if cluster["images"].get(image.image_id, {}).get("included") is False
    ]
    cluster["excluded_image_ids"] = excluded
    remaining = len(members) - len(excluded)
    # A pair cannot survive losing a member: one remaining capture is not a
    # cluster, so the whole grouping dissolves instead of leaving a singleton.
    dissolved = (
        len(members) >= MINIMUM_CLUSTER_SIZE and remaining < MINIMUM_CLUSTER_SIZE
    )
    cluster["dissolved"] = dissolved
    if dissolved:
        cluster["status"] = STATUS_DISSOLVED
    elif excluded:
        cluster["status"] = STATUS_EDITED
    elif cluster.get("status") in {STATUS_DISSOLVED, STATUS_EDITED}:
        cluster["status"] = STATUS_EDITED
    cluster["updated_at"] = utc_now()
    return cluster


def set_membership(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
    image_id: str,
    included: bool,
) -> dict[str, Any]:
    """Include or exclude one image, marking the cluster reviewed."""
    dataset.image(cluster_id, image_id)
    record = image_state(state, cluster_id, image_id)
    record["included"] = bool(included)
    cluster = _refresh_membership(state, dataset, cluster_id)
    if cluster["status"] == STATUS_UNREVIEWED:
        cluster["status"] = STATUS_EDITED
    return cluster


def dissolve_cluster(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
) -> dict[str, Any]:
    """Reject an entire grouping; every member becomes its own capture."""
    cluster = cluster_state(state, cluster_id)
    for image in dataset.cluster(cluster_id).images:
        image_state(state, cluster_id, image.image_id)["included"] = False
    cluster["excluded_image_ids"] = [
        image.image_id for image in dataset.cluster(cluster_id).images
    ]
    cluster["dissolved"] = True
    cluster["status"] = STATUS_DISSOLVED
    cluster["updated_at"] = utc_now()
    return cluster


def mark_irregular_cluster(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
) -> dict[str, Any]:
    """Classify a grouping as irregular and exclude all its images from exports."""
    dataset.cluster(cluster_id)
    cluster = cluster_state(state, cluster_id)
    cluster["dissolved"] = False
    cluster["status"] = STATUS_IRREGULAR
    cluster["updated_at"] = utc_now()
    return cluster


def approve_cluster(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
) -> dict[str, Any]:
    """Confirm the grouping exactly as the algorithm produced it."""
    cluster = cluster_state(state, cluster_id)
    for image in dataset.cluster(cluster_id).images:
        image_state(state, cluster_id, image.image_id)["included"] = True
    cluster["excluded_image_ids"] = []
    cluster["dissolved"] = False
    cluster["status"] = STATUS_APPROVED
    cluster["updated_at"] = utc_now()
    return cluster


def reopen_cluster(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
) -> dict[str, Any]:
    """Return a cluster to unreviewed with original membership."""
    cluster = cluster_state(state, cluster_id)
    for image in dataset.cluster(cluster_id).images:
        record = image_state(state, cluster_id, image.image_id)
        record["included"] = True
    cluster["excluded_image_ids"] = []
    cluster["dissolved"] = False
    cluster["status"] = STATUS_UNREVIEWED
    cluster["updated_at"] = utc_now()
    return cluster


def restore_cluster(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Replace one cluster decision wholesale, for undo.

    The incoming record is rebuilt field by field so an undo cannot introduce
    unknown images, invalid boxes, or a status the tool does not understand.
    """
    cluster = dataset.cluster(cluster_id)
    if not isinstance(record, dict):
        raise ValueError("cluster decision must be an object")
    status = str(record.get("status") or STATUS_UNREVIEWED)
    if status not in CLUSTER_STATUSES:
        raise ValueError(f"unknown cluster status: {status}")
    known_ids = {image.image_id for image in cluster.images}
    incoming_images = record.get("images") or {}
    if not isinstance(incoming_images, dict):
        raise ValueError("cluster images must be an object")
    restored_images: dict[str, Any] = {}
    for image in cluster.images:
        incoming = incoming_images.get(image.image_id) or {}
        boxes = incoming.get("boxes")
        normalized_boxes = None
        if boxes is not None:
            normalized_boxes = [
                {
                    "box_id": str(box.get("box_id") or f"box_{index + 1:03d}"),
                    "bbox": list(
                        validate_box(box.get("bbox"), image.width, image.height)
                    ),
                    "kind": str(box.get("kind") or "base_page"),
                    "submission_id": box.get("submission_id"),
                    "origin": str(box.get("origin") or "reviewer"),
                }
                for index, box in enumerate(boxes)
            ]
        restored_images[image.image_id] = {
            "image_id": image.image_id,
            "included": bool(incoming.get("included", True)),
            "bbox_status": str(incoming.get("bbox_status") or BBOX_STATUS_UNREVIEWED),
            "boxes": normalized_boxes,
        }
    excluded = [
        image_id
        for image_id in (record.get("excluded_image_ids") or [])
        if image_id in known_ids
    ]
    state.setdefault("clusters", {})[cluster_id] = {
        "cluster_id": cluster_id,
        "status": status,
        "dissolved": bool(record.get("dissolved")),
        "excluded_image_ids": excluded,
        "images": restored_images,
        "updated_at": utc_now(),
    }
    return cluster_state(state, cluster_id)


def validate_box(
    box: Any,
    width: int | None,
    height: int | None,
) -> tuple[int, int, int, int]:
    """Return an integer bbox validated against original source dimensions."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("bbox must have exactly four values")
    try:
        values = [int(round(float(value))) for value in box]
    except (TypeError, ValueError) as error:
        raise ValueError("bbox values must be numeric") from error
    x_min, y_min, x_max, y_max = values
    if x_min < 0 or y_min < 0:
        raise ValueError("bbox must start inside the image")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bbox must have positive width and height")
    if width is not None and x_max > int(width):
        raise ValueError("bbox exceeds source width")
    if height is not None and y_max > int(height):
        raise ValueError("bbox exceeds source height")
    return x_min, y_min, x_max, y_max


def set_boxes(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
    image_id: str,
    boxes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Replace the reviewer bbox set for one image.

    Bbox edits are deliberately independent of cluster membership: editing boxes
    marks only the bbox state, never the grouping decision.
    """
    image = dataset.image(cluster_id, image_id)
    normalized: list[dict[str, Any]] = []
    for index, box in enumerate(boxes or []):
        payload = box if isinstance(box, dict) else {"bbox": box}
        bbox = validate_box(payload.get("bbox"), image.width, image.height)
        normalized.append(
            {
                "box_id": str(payload.get("box_id") or f"box_{index + 1:03d}"),
                "bbox": list(bbox),
                "kind": str(payload.get("kind") or "base_page"),
                "submission_id": payload.get("submission_id"),
                "origin": str(payload.get("origin") or "reviewer"),
            }
        )
    record = image_state(state, cluster_id, image_id)
    record["boxes"] = normalized
    record["bbox_status"] = BBOX_STATUS_EDITED
    cluster_state(state, cluster_id)["updated_at"] = utc_now()
    return record


def approve_bboxes(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
    image_id: str,
    approved: bool = True,
) -> dict[str, Any]:
    """Approve or unapprove the bbox set for one image."""
    dataset.image(cluster_id, image_id)
    record = image_state(state, cluster_id, image_id)
    if approved:
        record["bbox_status"] = BBOX_STATUS_APPROVED
    else:
        record["bbox_status"] = (
            BBOX_STATUS_EDITED
            if record.get("boxes") is not None
            else BBOX_STATUS_UNREVIEWED
        )
    cluster_state(state, cluster_id)["updated_at"] = utc_now()
    return record


def mark_remaining_clusters_ok(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_ids: list[str] | None = None,
) -> list[str]:
    """Approve every still-unreviewed cluster and return the ids changed."""
    candidates = cluster_ids if cluster_ids is not None else [
        cluster.cluster_id for cluster in dataset.clusters
    ]
    changed: list[str] = []
    for cluster_id in candidates:
        if cluster_state(state, cluster_id).get("status") == STATUS_UNREVIEWED:
            approve_cluster(state, dataset, cluster_id)
            changed.append(cluster_id)
    return changed


def mark_remaining_bboxes_ok(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_ids: list[str] | None = None,
) -> list[str]:
    """Approve every unreviewed bbox set and return the image ids changed."""
    candidates = cluster_ids if cluster_ids is not None else [
        cluster.cluster_id for cluster in dataset.clusters
    ]
    changed: list[str] = []
    for cluster_id in candidates:
        for image in dataset.cluster(cluster_id).images:
            record = image_state(state, cluster_id, image.image_id)
            if record.get("bbox_status") == BBOX_STATUS_UNREVIEWED:
                record["bbox_status"] = BBOX_STATUS_APPROVED
                changed.append(image.image_id)
        cluster_state(state, cluster_id)["updated_at"] = utc_now()
    return changed


@dataclass(frozen=True)
class ClusterProgress:
    """Counts a reviewer needs to know how much work remains."""

    cluster_count: int
    reviewed_cluster_count: int
    approved_cluster_count: int
    edited_cluster_count: int
    dissolved_cluster_count: int
    irregular_cluster_count: int
    unreviewed_cluster_count: int
    excluded_image_count: int
    image_count: int
    bbox_approved_image_count: int
    bbox_edited_image_count: int
    bbox_unreviewed_image_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert progress counts to a JSON-serializable dictionary."""
        return asdict(self)


def progress(state: dict[str, Any], dataset: ReviewDataset) -> ClusterProgress:
    """Summarize reviewer progress across the dataset."""
    counts = dict.fromkeys(CLUSTER_STATUSES, 0)
    excluded = 0
    bbox_counts = {
        BBOX_STATUS_APPROVED: 0,
        BBOX_STATUS_EDITED: 0,
        BBOX_STATUS_UNREVIEWED: 0,
    }
    image_total = 0
    for cluster in dataset.clusters:
        record = cluster_state(state, cluster.cluster_id)
        counts[record.get("status", STATUS_UNREVIEWED)] = (
            counts.get(record.get("status", STATUS_UNREVIEWED), 0) + 1
        )
        excluded += len(record.get("excluded_image_ids") or ())
        for image in cluster.images:
            image_total += 1
            status = image_state(state, cluster.cluster_id, image.image_id).get(
                "bbox_status", BBOX_STATUS_UNREVIEWED
            )
            bbox_counts[status] = bbox_counts.get(status, 0) + 1
    reviewed = (
        counts[STATUS_APPROVED]
        + counts[STATUS_EDITED]
        + counts[STATUS_DISSOLVED]
        + counts[STATUS_IRREGULAR]
    )
    return ClusterProgress(
        cluster_count=len(dataset.clusters),
        reviewed_cluster_count=reviewed,
        approved_cluster_count=counts[STATUS_APPROVED],
        edited_cluster_count=counts[STATUS_EDITED],
        dissolved_cluster_count=counts[STATUS_DISSOLVED],
        irregular_cluster_count=counts[STATUS_IRREGULAR],
        unreviewed_cluster_count=counts[STATUS_UNREVIEWED],
        excluded_image_count=excluded,
        image_count=image_total,
        bbox_approved_image_count=bbox_counts[BBOX_STATUS_APPROVED],
        bbox_edited_image_count=bbox_counts[BBOX_STATUS_EDITED],
        bbox_unreviewed_image_count=bbox_counts[BBOX_STATUS_UNREVIEWED],
    )
