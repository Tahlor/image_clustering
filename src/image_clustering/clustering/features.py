"""Image decoding and local-feature extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.models import ImageFeatures, ImageItem


def _read_gray(path: Path, max_dimension: int) -> tuple[np.ndarray, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    scale = min(1.0, max_dimension / float(max(image.shape)))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image, scale


def _cache_key(image: ImageItem, config: ClusterConfig) -> str:
    stat = image.path.stat()
    payload = (
        f"{image.path}|{stat.st_size}|{stat.st_mtime_ns}|"
        f"{config.max_working_dimension}|{config.max_features}|"
        f"{config.sift_contrast_threshold}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_features(
    image: ImageItem,
    config: ClusterConfig,
    cache_dir: Path | None = None,
) -> ImageFeatures:
    """Load or compute SIFT features for one source image."""
    gray, scale = _read_gray(
        path=image.path,
        max_dimension=config.max_working_dimension,
    )
    cache_path = None
    if config.cache_features and cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{_cache_key(image=image, config=config)}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            return ImageFeatures(
                image=image,
                gray=gray,
                scale=scale,
                keypoints_xy=cached["keypoints_xy"],
                descriptors=cached["descriptors"],
            )

    detector = cv2.SIFT_create(
        nfeatures=config.max_features,
        contrastThreshold=config.sift_contrast_threshold,
    )
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    keypoints_xy = np.float32([keypoint.pt for keypoint in keypoints])
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            keypoints_xy=keypoints_xy,
            descriptors=descriptors,
        )
    return ImageFeatures(
        image=image,
        gray=gray,
        scale=scale,
        keypoints_xy=keypoints_xy,
        descriptors=descriptors,
    )
