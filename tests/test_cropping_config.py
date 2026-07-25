from __future__ import annotations

from image_clustering.cropping import load_default_config


def test_packaged_crop_config_loads() -> None:
    config = load_default_config()
    assert config.get("pair_search.window") == 3
    assert config.get("selection.duplicate_correlation_threshold") == 0.985
