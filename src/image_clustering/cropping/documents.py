from __future__ import annotations

import cv2
import numpy as np

from .config import Config
from .image_ops import bbox_area
from .models import BBox


def detect_gutter(image_gray: np.ndarray, config: Config) -> int | None:
    """Detect the central binding of a two-page spread."""
    if not bool(config.get("pages.detect_gutter", True)):
        return None
    height, width = image_gray.shape
    if width < 1.15 * height:
        return None
    normalized = cv2.GaussianBlur(
        image_gray,
        (0, 0),
        sigmaX=max(2.0, width / 450),
    )
    gradient = np.abs(cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3))
    darkness = 1.0 - normalized.astype(np.float32).mean(axis=0) / 255.0
    edge_profile = gradient.mean(axis=0)
    edge_profile /= max(float(np.percentile(edge_profile, 95)), 1e-6)
    profile = 0.55 * darkness + 0.45 * edge_profile
    low = round(float(config.get("pages.gutter_search_min_fraction", 0.34)) * width)
    high = round(float(config.get("pages.gutter_search_max_fraction", 0.66)) * width)
    search = profile[low:high]
    if search.size == 0:
        return None
    index = low + int(np.argmax(search))
    prominence = float(profile[index] - np.median(search))
    if prominence < float(config.get("pages.gutter_min_prominence", 0.18)):
        return None
    return index


def page_boxes(
    shape: tuple[int, int],
    gutter: int | None,
    config: Config,
) -> list[tuple[str, BBox]]:
    """Return conservative page boxes in one reference frame."""
    height, width = shape
    outer = round(
        float(config.get("pages.outer_margin_fraction", 0.018)) * min(height, width)
    )
    if gutter is None:
        return [("single", (outer, outer, width - outer, height - outer))]
    gutter_margin = round(
        float(config.get("pages.gutter_margin_fraction", 0.010)) * width
    )
    return [
        (
            "left",
            (
                outer,
                outer,
                max(outer + 1, gutter - gutter_margin),
                height - outer,
            ),
        ),
        (
            "right",
            (
                min(width - outer - 1, gutter + gutter_margin),
                outer,
                width - outer,
                height - outer,
            ),
        ),
    ]


def _smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    window = max(3, window | 1)
    return cv2.GaussianBlur(
        profile.reshape(1, -1).astype(np.float32),
        (window, 1),
        0,
    ).ravel()


def expand_to_sheet_boundary(
    image_gray: np.ndarray,
    support_bbox: BBox,
    page_bbox: BBox,
    config: Config,
) -> BBox:
    """Expand residual support to nearby axis-aligned paper boundaries."""
    x0, y0, x1, y1 = support_bbox
    px0, py0, px1, py1 = page_bbox
    page_width = px1 - px0
    page_height = py1 - py0
    search_fraction = float(config.get("sheet_expansion.search_padding_fraction", 0.20))
    max_fraction = float(config.get("sheet_expansion.max_expansion_fraction", 0.38))
    search = (
        max(px0, x0 - round(search_fraction * page_width)),
        max(py0, y0 - round(search_fraction * page_height)),
        min(px1, x1 + round(search_fraction * page_width)),
        min(py1, y1 + round(search_fraction * page_height)),
    )
    sx0, sy0, sx1, sy1 = search
    roi = image_gray[sy0:sy1, sx0:sx1]
    if roi.size == 0:
        return support_bbox
    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    gradient_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    gradient_y = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    smooth_fraction = float(
        config.get("sheet_expansion.projection_smoothing_fraction", 0.012)
    )
    gradient_x = _smooth_profile(
        gradient_x,
        round(smooth_fraction * len(gradient_x)),
    )
    gradient_y = _smooth_profile(
        gradient_y,
        round(smooth_fraction * len(gradient_y)),
    )

    local_x0 = x0 - sx0
    local_x1 = x1 - sx0
    local_y0 = y0 - sy0
    local_y1 = y1 - sy0
    max_dx = round(max_fraction * page_width)
    max_dy = round(max_fraction * page_height)
    prominence = float(config.get("sheet_expansion.minimum_edge_prominence", 1.30))

    def choose(
        profile: np.ndarray,
        start: int,
        stop: int,
        prefer_last: bool,
    ) -> int | None:
        start = max(0, start)
        stop = min(len(profile), stop)
        if stop <= start:
            return None
        values = profile[start:stop]
        threshold = prominence * max(float(np.median(values)), 1e-5)
        candidates = np.where(values >= threshold)[0] + start
        if len(candidates) == 0:
            return None
        return int(candidates[-1] if prefer_last else candidates[0])

    left = choose(gradient_x, local_x0 - max_dx, local_x0 + 1, True)
    right = choose(gradient_x, local_x1, local_x1 + max_dx + 1, False)
    top = choose(gradient_y, local_y0 - max_dy, local_y0 + 1, True)
    bottom = choose(gradient_y, local_y1, local_y1 + max_dy + 1, False)
    fallback_x = round(
        float(config.get("sheet_expansion.fallback_padding_fraction", 0.035))
        * page_width
    )
    fallback_y = round(
        float(config.get("sheet_expansion.fallback_padding_fraction", 0.035))
        * page_height
    )
    result = (
        max(px0, sx0 + (left if left is not None else max(0, local_x0 - fallback_x))),
        max(py0, sy0 + (top if top is not None else max(0, local_y0 - fallback_y))),
        min(
            px1,
            sx0
            + (
                right if right is not None else min(roi.shape[1], local_x1 + fallback_x)
            ),
        ),
        min(
            py1,
            sy0
            + (
                bottom
                if bottom is not None
                else min(roi.shape[0], local_y1 + fallback_y)
            ),
        ),
    )
    return support_bbox if bbox_area(result) < bbox_area(support_bbox) else result


def boundary_score(image_gray: np.ndarray, bbox: BBox) -> float:
    """Measure whether at least three sides resemble a sheet boundary."""
    x0, y0, x1, y1 = bbox
    height, width = image_gray.shape
    box_width = x1 - x0
    box_height = y1 - y0
    if box_width < 20 or box_height < 20:
        return 0.0
    blurred = cv2.GaussianBlur(image_gray, (3, 3), 0)
    gradient_x = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    gradient_y = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3))
    band = max(1, round(min(box_width, box_height) * 0.006))
    search_x = max(8, round(box_width * 0.10))
    search_y = max(8, round(box_height * 0.10))

    def vertical_energy(x: int) -> float:
        xa = max(0, x - band)
        xb = min(width, x + band + 1)
        return float(np.mean(gradient_x[max(0, y0) : min(height, y1), xa:xb]))

    def horizontal_energy(y: int) -> float:
        ya = max(0, y - band)
        yb = min(height, y + band + 1)
        return float(np.mean(gradient_y[ya:yb, max(0, x0) : min(width, x1)]))

    def prominence(target: float, nearby: list[float]) -> float:
        return target / max(float(np.median(nearby)), 1.0) if nearby else 0.0

    x_offsets = range(3, search_x + 1, max(3, band * 2))
    y_offsets = range(3, search_y + 1, max(3, band * 2))
    ratios = np.asarray(
        [
            prominence(
                vertical_energy(x0),
                [
                    vertical_energy(x0 + offset)
                    for offset in x_offsets
                    if x0 + offset < x1
                ],
            ),
            prominence(
                vertical_energy(x1 - 1),
                [
                    vertical_energy(x1 - 1 - offset)
                    for offset in x_offsets
                    if x1 - 1 - offset > x0
                ],
            ),
            prominence(
                horizontal_energy(y0),
                [
                    horizontal_energy(y0 + offset)
                    for offset in y_offsets
                    if y0 + offset < y1
                ],
            ),
            prominence(
                horizontal_energy(y1 - 1),
                [
                    horizontal_energy(y1 - 1 - offset)
                    for offset in y_offsets
                    if y1 - 1 - offset > y0
                ],
            ),
        ],
        dtype=np.float32,
    )
    return float(np.sort(ratios)[1])


def absolute_detail_score(image_gray: np.ndarray, bbox: BBox) -> float:
    """Measure absolute local detail without contrast stretching."""
    x0, y0, x1, y1 = bbox
    roi = image_gray[y0:y1, x0:x1]
    if roi.size < 100:
        return 0.0
    values = roi.astype(np.float32) / 255.0
    low_frequency = cv2.GaussianBlur(
        values,
        (0, 0),
        sigmaX=max(2.0, min(roi.shape) / 65),
    )
    highpass = np.abs(values - low_frequency)
    gradients = cv2.magnitude(
        cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3),
    )
    return 0.55 * float(np.percentile(highpass, 85)) + 0.45 * float(
        np.percentile(gradients, 80)
    )


def content_score(image_gray: np.ndarray, bbox: BBox) -> float:
    """Measure nonsemantic ink and structural content."""
    x0, y0, x1, y1 = bbox
    roi = image_gray[y0:y1, x0:x1]
    if roi.size < 100:
        return 0.0
    normalized = cv2.normalize(
        roi,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    highpass = normalized - cv2.GaussianBlur(
        normalized,
        (0, 0),
        sigmaX=max(2.0, min(roi.shape) / 70),
    )
    gradients = cv2.magnitude(
        cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3),
    )
    dark_fraction = float((normalized < 0.62).mean())
    return (
        0.35 * dark_fraction
        + 0.35 * float(np.percentile(np.abs(highpass), 85))
        + 0.30 * float(np.percentile(gradients, 80))
    )


def page_frontness_score(image_gray: np.ndarray, bbox: BBox) -> float:
    """Estimate whether a crop is a front-facing data-bearing sheet."""
    x0, y0, x1, y1 = bbox
    roi = image_gray[y0:y1, x0:x1]
    if roi.size < 400:
        return 0.0
    values = roi.astype(np.float32) / 255.0
    background = cv2.GaussianBlur(
        values,
        (0, 0),
        sigmaX=max(2.0, min(roi.shape) / 45),
    )
    dark_residual = np.maximum(background - values, 0.0)
    light_residual = np.maximum(values - background, 0.0)
    dark_p95 = float(np.percentile(dark_residual, 95))
    light_p95 = float(np.percentile(light_residual, 95))
    asymmetry = dark_p95 / max(light_p95, 1e-4)
    asymmetry_score = float(np.clip((asymmetry - 1.35) / 0.85, 0.0, 1.0))
    gradients = cv2.magnitude(
        cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3),
    )
    sharpness = float(np.clip(np.percentile(gradients, 85) / 1.10, 0.0, 1.0))
    return 0.80 * asymmetry_score + 0.20 * sharpness
