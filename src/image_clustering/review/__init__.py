"""Reviewer-facing labeling tool for clustering and crop corrections.

The modules here never mutate canonical clustering or cropping output. They read
a completed evaluation directory, keep reviewer decisions in their own
``review_labels`` namespace, and export corrected manifests on request.
"""

from __future__ import annotations

from image_clustering.review.app import review_app_html
from image_clustering.review.dataset import (
    ReviewBox,
    ReviewCluster,
    ReviewDataset,
    ReviewImage,
    build_review_dataset,
    dataset_payload,
)
from image_clustering.review.decisions import (
    CLUSTER_STATUSES,
    ClusterProgress,
    approve_bboxes,
    approve_cluster,
    cluster_state,
    empty_state,
    image_state,
    mark_remaining_bboxes_ok,
    mark_remaining_clusters_ok,
    progress,
    restore_cluster,
    set_boxes,
    set_membership,
    validate_box,
)
from image_clustering.review.exports import write_review_exports
from image_clustering.review.previews import ensure_preview
from image_clustering.review.store import DecisionStore

__all__ = [
    "CLUSTER_STATUSES",
    "ClusterProgress",
    "DecisionStore",
    "ReviewBox",
    "ReviewCluster",
    "ReviewDataset",
    "ReviewImage",
    "approve_bboxes",
    "approve_cluster",
    "build_review_dataset",
    "cluster_state",
    "dataset_payload",
    "empty_state",
    "ensure_preview",
    "image_state",
    "mark_remaining_bboxes_ok",
    "mark_remaining_clusters_ok",
    "progress",
    "restore_cluster",
    "review_app_html",
    "set_boxes",
    "set_membership",
    "validate_box",
    "write_review_exports",
]
