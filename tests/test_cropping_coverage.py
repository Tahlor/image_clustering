from __future__ import annotations

from pathlib import Path

import numpy as np

from image_clustering.cropping.config import Config
from image_clustering.cropping.coverage import (
    fallback_page_submissions,
    pad_tall_sheet,
)
from image_clustering.cropping.models import ImageRecord, WorkImage


def _config() -> Config:
    return Config(
        {
            "pages": {
                "detect_gutter": False,
                "outer_margin_fraction": 0.02,
            },
            "selection": {
                "min_page_frontness": 0.58,
                "fallback_min_content_score": 0.01,
                "fallback_min_detail_score": 0.001,
                "fallback_min_page_frontness": 0.0,
            },
            "sheet_expansion": {
                "tall_sheet_min_width_fraction": 0.68,
                "tall_sheet_min_height_fraction": 0.45,
                "tall_sheet_top_padding_fraction": 0.12,
                "tall_sheet_bottom_padding_fraction": 0.08,
            },
        }
    )


def test_tall_sheet_padding_expands_active_text_band() -> None:
    assert pad_tall_sheet((55, 60, 97, 160), (50, 2, 100, 198), _config()) == (
        55,
        36,
        97,
        176,
    )


def test_crop_empty_cluster_gets_recognizer_submission(monkeypatch) -> None:
    monkeypatch.setattr(
        "image_clustering.cropping.coverage.content_score", lambda image, bbox: 0.3
    )
    monkeypatch.setattr(
        "image_clustering.cropping.coverage.absolute_detail_score",
        lambda image, bbox: 0.2,
    )
    monkeypatch.setattr(
        "image_clustering.cropping.coverage.page_frontness_score",
        lambda image, bbox: 0.9,
    )
    image = np.full((100, 80), 220, dtype=np.uint8)
    record = ImageRecord(0, Path("duplicate.j2k"), Path("."), "duplicate.j2k")
    work = {
        0: WorkImage(
            record=record,
            gray=image,
            scale=1.0,
            fingerprint=np.zeros(4, dtype=np.float32),
        )
    }
    submissions = fallback_page_submissions(
        group_id="duplicate",
        anchor=0,
        transforms={0: np.eye(3)},
        work=work,
        original_shapes={0: image.shape},
        records_by_index={0: record},
        config=_config(),
    )
    assert len(submissions) == 1
    assert submissions[0].kind == "base_page"
    assert submissions[0].completeness == "complete"
