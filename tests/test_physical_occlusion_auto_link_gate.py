from pathlib import Path

import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content import ContentMetrics
from image_clustering.clustering.models import ImageFeatures, ImageItem, Registration
from image_clustering.clustering.scoring import _automatic_link_safety_reason


def _features(name: str, index: int) -> ImageFeatures:
    return ImageFeatures(
        image=ImageItem(
            image_id=name,
            path=Path(name),
            sequence_id="sequence",
            sequence_index=index,
        ),
        gray=np.zeros((16, 16), dtype=np.uint8),
        scale=1.0,
        keypoints_xy=np.empty((0, 2), dtype=np.float32),
        descriptors=np.empty((0, 128), dtype=np.float32),
    )


def _content() -> ContentMetrics:
    return ContentMetrics(
        unmatched_ink_fraction=0.01,
        unmatched_ink_union_fraction=0.10,
        ink_mismatch_tiles_fraction=0.20,
        coherent_ink_component_count=1,
        largest_ink_component_fraction=0.10,
        residual_tiles_changed_fraction=0.30,
        occlusion_candidate_count=1,
        occlusion_area_fraction=0.50,
        occlusion_residual_capture=0.90,
        occlusion_rectangularity=0.80,
        occlusion_boundary_score=1.0,
        occlusion_material_fraction=0.50,
        occlusion_material_median=0.04,
        outside_unmatched_ink_fraction=0.0,
        outside_unmatched_ink_union_fraction=0.0,
        outside_ink_mismatch_tiles_fraction=0.0,
        full_page_occlusion_count=0,
        shallow_occlusion_count=0,
        page_count=1,
        inside_unmatched_ink_union_fraction=0.30,
        occlusion_ink_mismatch_capture=0.90,
        occlusion_localization_contrast=0.30,
    )


def test_physical_occlusion_requires_strong_feature_overlap_for_auto_link() -> None:
    config = ClusterConfig()
    reason = _automatic_link_safety_reason(
        previous=_features("i4071658-00306.jpg", 0),
        current=_features("i4071658-00307.jpg", 1),
        registration=Registration(
            accepted=True,
            model="affine",
            matrix=np.eye(3),
            feature_overlap=(
                config.automatic_link_min_physical_occlusion_feature_overlap - 0.001
            ),
            alignment_score=0.95,
        ),
        content=_content(),
        branch="physical_occlusion",
        config=config,
    )
    assert reason == (
        "physical-occlusion match lacks strong document-specific feature overlap"
    )


def test_physical_occlusion_gate_allows_threshold_overlap() -> None:
    config = ClusterConfig()
    reason = _automatic_link_safety_reason(
        previous=_features("i4071658-00306.jpg", 0),
        current=_features("i4071658-00307.jpg", 1),
        registration=Registration(
            accepted=True,
            model="affine",
            matrix=np.eye(3),
            feature_overlap=(
                config.automatic_link_min_physical_occlusion_feature_overlap
            ),
            alignment_score=0.95,
        ),
        content=_content(),
        branch="physical_occlusion",
        config=config,
    )
    assert reason is None
