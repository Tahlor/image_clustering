from __future__ import annotations

import numpy as np

from image_clustering.cropping.config import Config
from image_clustering.cropping.residuals import ResidualGrid, component_rectangles


def test_low_mass_component_tails_are_trimmed() -> None:
    scores = np.zeros((7, 3), dtype=np.float32)
    changed = np.zeros((7, 3), dtype=bool)
    changed[:, 1] = True
    scores[:, 1] = [0.11, 0.12, 0.70, 0.90, 0.75, 0.12, 0.11]
    boxes = [
        (col * 10, row * 10, (col + 1) * 10, (row + 1) * 10)
        for row in range(7)
        for col in range(3)
    ]
    grid = ResidualGrid(
        scores=scores,
        zscores=scores,
        changed=changed,
        valid=np.ones_like(changed),
        boxes=boxes,
        tile_rows=7,
        tile_cols=3,
        baseline=0.10,
        scale=0.01,
    )
    config = Config(
        {
            "residual": {
                "min_component_tiles": 2,
                "support_mass_trim_fraction_x": 0.0,
                "support_mass_trim_fraction_y": 0.12,
            }
        }
    )
    rectangles = component_rectangles(grid, config)
    assert len(rectangles) == 1
    bbox, _ = rectangles[0]
    assert bbox[1] >= 20
    assert bbox[3] <= 50
