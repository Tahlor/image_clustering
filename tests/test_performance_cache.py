"""Regression tests for exact feature and pair-result caches."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from image_clustering.clustering import api
from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.features import _legacy_cache_key, extract_features
from image_clustering.clustering.models import (
    ImageFeatures,
    ImageItem,
    PairComparison,
)


def _write_test_image(path: Path) -> None:
    image = np.full((240, 320), 245, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (300, 220), 20, 3)
    cv2.putText(
        image,
        "VERMONT 1924",
        (35, 125),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        15,
        2,
        cv2.LINE_AA,
    )
    assert cv2.imwrite(str(path), image)


def test_feature_cache_reuses_exact_working_image(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "image.png"
    _write_test_image(image_path)
    item = ImageItem("image.png", image_path, ".", 0)
    config = ClusterConfig(
        workers=1,
        cache_working_images=True,
        feature_cache_compressed=True,
    )
    cache_dir = tmp_path / "cache"

    first = extract_features(item, config, cache_dir)

    def fail_decode(*_args, **_kwargs):
        raise AssertionError("source image was decoded despite a complete cache hit")

    monkeypatch.setattr(
        "image_clustering.clustering.features._read_gray",
        fail_decode,
    )
    second = extract_features(item, config, cache_dir)

    assert np.array_equal(second.gray, first.gray)
    assert second.scale == first.scale
    assert np.array_equal(second.keypoints_xy, first.keypoints_xy)
    assert np.array_equal(second.descriptors, first.descriptors)


def test_uncompressed_feature_cache_is_exact(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _write_test_image(image_path)
    item = ImageItem("image.png", image_path, ".", 0)
    config = ClusterConfig(
        workers=1,
        cache_working_images=True,
        feature_cache_compressed=False,
    )
    cache_dir = tmp_path / "cache"

    first = extract_features(item, config, cache_dir)
    second = extract_features(item, config, cache_dir)

    assert np.array_equal(second.gray, first.gray)
    assert np.array_equal(second.descriptors, first.descriptors)


def _features(path: Path, image_id: str, index: int) -> ImageFeatures:
    path.write_bytes(image_id.encode("utf-8"))
    return ImageFeatures(
        image=ImageItem(image_id, path, "sequence", index),
        gray=np.full((32, 32), 200, dtype=np.uint8),
        scale=1.0,
        keypoints_xy=np.empty((0, 2), dtype=np.float32),
        descriptors=np.empty((0, 128), dtype=np.float32),
    )


def test_pair_cache_skips_repeated_scoring(tmp_path: Path, monkeypatch) -> None:
    previous = _features(tmp_path / "a.bin", "a", 0)
    current = _features(tmp_path / "b.bin", "b", 1)
    config = ClusterConfig(max_gap=1, workers=1, cache_pairs=True)
    calls = 0

    def fake_score_pair(**kwargs) -> PairComparison:
        nonlocal calls
        calls += 1
        return PairComparison(
            first_image_id=kwargs["previous"].image.image_id,
            second_image_id=kwargs["current"].image.image_id,
            sequence_id="sequence",
            index_gap=kwargs["index_gap"],
            same_document=False,
            confidence=0.1,
            reason="cached test",
        )

    monkeypatch.setattr(api, "score_pair", fake_score_pair)
    cache_dir = tmp_path / "cache"
    first = api._score_sequence(
        [previous, current],
        config,
        cache_dir,
        workers=1,
        show_progress=False,
    )
    second = api._score_sequence(
        [previous, current],
        config,
        cache_dir,
        workers=1,
        show_progress=False,
    )

    assert calls == 1
    assert second == first


def test_pair_cache_invalidates_when_source_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    previous = _features(tmp_path / "a.bin", "a", 0)
    current = _features(tmp_path / "b.bin", "b", 1)
    config = ClusterConfig(max_gap=1, workers=1, cache_pairs=True)
    calls = 0

    def fake_score_pair(**kwargs) -> PairComparison:
        nonlocal calls
        calls += 1
        return PairComparison(
            first_image_id="a",
            second_image_id="b",
            sequence_id="sequence",
            index_gap=1,
            same_document=False,
            confidence=0.1,
            reason=f"call {calls}",
        )

    monkeypatch.setattr(api, "score_pair", fake_score_pair)
    cache_dir = tmp_path / "cache"
    api._score_sequence(
        [previous, current],
        config,
        cache_dir,
        workers=1,
        show_progress=False,
    )
    current.image.path.write_bytes(b"changed source contents")
    api._score_sequence(
        [previous, current],
        config,
        cache_dir,
        workers=1,
        show_progress=False,
    )

    assert calls == 2


def test_legacy_feature_cache_is_upgraded_without_sift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "image.png"
    _write_test_image(image_path)
    item = ImageItem("image.png", image_path, ".", 0)
    config = ClusterConfig(workers=1)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    legacy_path = cache_dir / f"{_legacy_cache_key(item, config)}.npz"
    expected_keypoints = np.asarray([[10.0, 20.0]], dtype=np.float32)
    expected_descriptors = np.ones((1, 128), dtype=np.float32)
    np.savez_compressed(
        legacy_path,
        keypoints_xy=expected_keypoints,
        descriptors=expected_descriptors,
    )

    def fail_sift(*_args, **_kwargs):
        raise AssertionError("legacy cache upgrade recomputed SIFT")

    monkeypatch.setattr(cv2, "SIFT_create", fail_sift)
    features = extract_features(item, config, cache_dir)

    assert np.array_equal(features.keypoints_xy, expected_keypoints)
    assert np.array_equal(features.descriptors, expected_descriptors)
    assert len(list(cache_dir.glob("*.npz"))) == 2
