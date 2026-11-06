"""Reviewer decision state: membership, dissolution, relation subtype, and bbox approval.

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

# Relation subtype, kept separate from membership status (see REVIEW_TOOL.md).
# same_document: literally the same visible document (a re-capture, or the same
#   sheet with and without a covering occluder).
# same_background_document: the same underlying base page, but the foreground
#   sheet differs between captures, so the visible documents are not the same.
RELATION_SAME_DOCUMENT = "same_document"
RELATION_SAME_BACKGROUND = "same_background_document"
RELATION_DIFFERENT = "different_document"
RELATION_UNSURE = "unsure"

RELATIONS = (
    RELATION_SAME_DOCUMENT,
    RELATION_SAME_BACKGROUND,
    RELATION_DIFFERENT,
    RELATION_UNSURE,
)

# The membership status each relation implies; any other status clears the relation.
RELATION_STATUS = {
    RELATION_SAME_DOCUMENT: STATUS_APPROVED,
    RELATION_SAME_BACKGROUND: STATUS_APPROVED,
    RELATION_DIFFERENT: STATUS_DISSOLVED,
    RELATION_UNSURE: STATUS_IRREGULAR,
}

# primary_view records which capture is the primary unoccluded view of the
# (background) document: a member image id, PRIMARY_VIEW_NONE when no capture
# is unoccluded, or None when not yet reviewed.
PRIMARY_VIEW_NONE = "none"

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
            "relation": None,
            "primary_view": None,
            "optional": False,
            "updated_at": None,
        }
        clusters[cluster_id] = record
    record.setdefault("cluster_id", cluster_id)
    record.setdefault("relation", None)
    record.setdefault("primary_view", None)
    record.setdefault("optional", False)
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


def _reconcile_relation(cluster: dict[str, Any]) -> dict[str, Any]:
    """Drop a relation subtype the current membership status contradicts.

    An edited cluster keeps a same-* relation: it is still a confirmed grouping.
    """
    relation = cluster.get("relation")
    if relation is None:
        return cluster
    implied = RELATION_STATUS.get(relation)
    status = cluster.get("status")
    if status == implied or (implied == STATUS_APPROVED and status == STATUS_EDITED):
        return cluster
    cluster["relation"] = None
    return cluster


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
    return _reconcile_relation(cluster)


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
    return _reconcile_relation(cluster)


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
    return _reconcile_relation(cluster)


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
    return _reconcile_relation(cluster)


def approve_cluster(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
) -> dict[str, Any]:
    """Confirm the grouping exactly as the algorithm produced it.

    Approving a cluster also vouches for the boxes the reviewer was looking at: each
    capture whose box set is still unreviewed is approved, and its proposals are
    copied into the decision so that regenerating proposals later cannot change
    what was approved. Edited box sets are left as they are. A dissolved cluster's
    boxes stay unreviewed.
    """
    cluster = cluster_state(state, cluster_id)
    for image in dataset.cluster(cluster_id).images:
        record = image_state(state, cluster_id, image.image_id)
        record["included"] = True
        if record.get("bbox_status", BBOX_STATUS_UNREVIEWED) == BBOX_STATUS_UNREVIEWED:
            _pin_proposals(record, image)
    cluster["excluded_image_ids"] = []
    cluster["dissolved"] = False
    cluster["status"] = STATUS_APPROVED
    cluster["updated_at"] = utc_now()
    return _reconcile_relation(cluster)


def _pin_proposals(record: dict[str, Any], image: Any) -> None:
    """Approve an image's box set, copying the automatic boxes in if none were saved."""
    if record.get("boxes") is None:
        try:
            record["boxes"] = [
                normalize_box(box.to_dict(), index, image.width, image.height)
                for index, box in enumerate(image.boxes)
            ]
        except ValueError:
            return  # an unusable proposal stays unreviewed rather than approved
        record["pinned"] = True
    record["bbox_status"] = BBOX_STATUS_APPROVED


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
    cluster["relation"] = None
    cluster["primary_view"] = None
    cluster["optional"] = False
    cluster["updated_at"] = utc_now()
    return cluster


def set_optional(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
    optional: bool,
) -> dict[str, Any]:
    """Flag a cluster as optional: technically right but a poor example (say a blank
    sheet on a blank page). A model gets credit for finding it and no penalty for a
    miss. Independent of status and relation.
    """
    dataset.cluster(cluster_id)
    cluster = cluster_state(state, cluster_id)
    cluster["optional"] = bool(optional)
    cluster["updated_at"] = utc_now()
    return cluster


def set_relation(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
    relation: str,
) -> dict[str, Any]:
    """Record the relation subtype and the membership status it implies.

    Same-document and same-background relations approve the full grouping,
    different-document dissolves it, and unsure marks it irregular.
    """
    if relation not in RELATIONS:
        raise ValueError(f"unknown relation: {relation}")
    status = RELATION_STATUS[relation]
    if status == STATUS_APPROVED:
        cluster = approve_cluster(state, dataset, cluster_id)
    elif status == STATUS_DISSOLVED:
        cluster = dissolve_cluster(state, dataset, cluster_id)
    else:
        cluster = mark_irregular_cluster(state, dataset, cluster_id)
    cluster["relation"] = relation
    return cluster


def validate_primary_view(value: Any, image_ids: set[str]) -> str | None:
    """Return a primary_view that is None, PRIMARY_VIEW_NONE, or a member image id."""
    if value is None or value == PRIMARY_VIEW_NONE:
        return value
    if isinstance(value, str) and value in image_ids:
        return value
    raise ValueError(
        f"primary_view must be a member image id or {PRIMARY_VIEW_NONE!r}"
    )


def set_primary_view(
    state: dict[str, Any],
    dataset: ReviewDataset,
    cluster_id: str,
    primary_view: str | None,
) -> dict[str, Any]:
    """Mark which capture is the primary unoccluded view; independent of status."""
    member_ids = {image.image_id for image in dataset.cluster(cluster_id).images}
    cluster = cluster_state(state, cluster_id)
    cluster["primary_view"] = validate_primary_view(primary_view, member_ids)
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
                normalize_box(box, index, image.width, image.height)
                for index, box in enumerate(boxes)
            ]
        restored_images[image.image_id] = {
            "image_id": image.image_id,
            "included": bool(incoming.get("included", True)),
            "bbox_status": str(incoming.get("bbox_status") or BBOX_STATUS_UNREVIEWED),
            "boxes": normalized_boxes,
        }
        if incoming.get("pinned") and normalized_boxes is not None:
            restored_images[image.image_id]["pinned"] = True
    excluded = [
        image_id
        for image_id in (record.get("excluded_image_ids") or [])
        if image_id in known_ids
    ]
    relation = record.get("relation")
    if relation is not None and relation not in RELATIONS:
        raise ValueError(f"unknown relation: {relation}")
    restored = {
        "cluster_id": cluster_id,
        "status": status,
        "dissolved": bool(record.get("dissolved")),
        "excluded_image_ids": excluded,
        "images": restored_images,
        "relation": relation,
        "primary_view": validate_primary_view(record.get("primary_view"), known_ids),
        "optional": bool(record.get("optional")),
        "updated_at": utc_now(),
    }
    state.setdefault("clusters", {})[cluster_id] = _reconcile_relation(restored)
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


def validate_polygon(
    polygon: Any,
    width: int | None,
    height: int | None,
) -> list[list[int]] | None:
    """Return an integer outline of at least three points inside the image, or None."""
    if polygon is None:
        return None
    if not isinstance(polygon, (list, tuple)) or len(polygon) < 3:
        raise ValueError("polygon must have at least three points")
    points = []
    for point in polygon:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("polygon points must be [x, y]")
        try:
            x, y = (int(round(float(value))) for value in point)
        except (TypeError, ValueError) as error:
            raise ValueError("polygon values must be numeric") from error
        if x < 0 or y < 0:
            raise ValueError("polygon must lie inside the image")
        if (width is not None and x > int(width)) or (height is not None and y > int(height)):
            raise ValueError("polygon exceeds the source image")
        points.append([x, y])
    return points


def normalize_box(
    payload: dict[str, Any],
    index: int,
    width: int | None,
    height: int | None,
) -> dict[str, Any]:
    """Validate one reviewer box.

    A box may carry a polygon: a 4-corner quad the reviewer edits (a trapezoid for a
    page photographed at an angle) or a segmentation outline. The bbox is then the
    polygon's bounds, so the two can never disagree.
    """
    polygon = validate_polygon(payload.get("polygon"), width, height)
    if polygon is not None:
        xs, ys = [p[0] for p in polygon], [p[1] for p in polygon]
        bbox = validate_box([min(xs), min(ys), max(xs), max(ys)], width, height)
    else:
        bbox = validate_box(payload.get("bbox"), width, height)
    box = {
        "box_id": str(payload.get("box_id") or f"box_{index + 1:03d}"),
        "bbox": list(bbox),
        "kind": str(payload.get("kind") or "base_page"),
        "submission_id": payload.get("submission_id"),
        "origin": str(payload.get("origin") or "reviewer"),
    }
    if polygon is not None:
        box["polygon"] = polygon
    # Optional: a real box that is a poor example (blank on blank); not penalized when missed.
    if payload.get("optional"):
        box["optional"] = True
    for key in ("completeness", "confidence"):
        if payload.get(key) is not None:
            box[key] = payload[key]
    return box


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
        normalized.append(normalize_box(payload, index, image.width, image.height))
    record = image_state(state, cluster_id, image_id)
    record["boxes"] = normalized
    record.pop("pinned", None)
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
    """Approve or unapprove the bbox set for one image.

    Approving pins the proposals on screen (see _pin_proposals); unapproving a pinned
    set returns it to the live proposals, unreviewed.
    """
    image = dataset.image(cluster_id, image_id)
    record = image_state(state, cluster_id, image_id)
    if approved:
        _pin_proposals(record, image)
        record["bbox_status"] = BBOX_STATUS_APPROVED
    else:
        if record.pop("pinned", False):
            record["boxes"] = None
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
    relation_counts: dict[str, int]
    primary_view_set_count: int
    optional_cluster_count: int = 0

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
    relation_counts = dict.fromkeys(RELATIONS, 0)
    primary_view_set = 0
    optional = 0
    for cluster in dataset.clusters:
        record = cluster_state(state, cluster.cluster_id)
        if record.get("relation") in relation_counts:
            relation_counts[record["relation"]] += 1
        if record.get("primary_view") is not None:
            primary_view_set += 1
        optional += bool(record.get("optional"))
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
        relation_counts=relation_counts,
        primary_view_set_count=primary_view_set,
        optional_cluster_count=optional,
    )
