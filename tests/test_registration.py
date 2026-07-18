"""Tests for source-coordinate registration handoff."""

import numpy as np

from image_clustering.clustering.models import Registration
from image_clustering.clustering.registration import source_pixel_transform


def test_source_pixel_transform_accounts_for_resize_scales() -> None:
    registration = Registration(
        accepted=True,
        model="affine",
        matrix=np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]]),
    )
    transform = source_pixel_transform(
        registration=registration,
        previous_scale=0.5,
        current_scale=0.25,
    )
    assert transform == (
        (0.5, 0.0, 20.0),
        (0.0, 0.5, 40.0),
        (0.0, 0.0, 1.0),
    )
