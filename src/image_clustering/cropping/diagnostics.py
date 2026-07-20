from __future__ import annotations

import cv2
import numpy as np

from .models import Submission


def annotate(image: np.ndarray, submissions: list[Submission]) -> np.ndarray:
    output = image.copy()
    for submission in submissions:
        x0, y0, x1, y1 = submission.bbox
        if submission.completeness == "complete":
            color = (0, 190, 0)
        elif submission.completeness == "review_required":
            color = (0, 0, 220)
        else:
            color = (0, 140, 255)
        thickness = max(3, round(min(image.shape[:2]) / 800))
        cv2.rectangle(output, (x0, y0), (x1, y1), color, thickness)
        label = (
            f"{submission.submission_id} {submission.completeness} {submission.kind}"
        )
        cv2.putText(
            output,
            label,
            (x0 + 8, max(30, y0 + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.6, min(image.shape[:2]) / 2200),
            color,
            max(2, thickness // 2),
            cv2.LINE_AA,
        )
    return output


def residual_heatmap(
    scores: np.ndarray, valid: np.ndarray, output_shape: tuple[int, int]
) -> np.ndarray:
    values = scores.copy()
    finite = valid & np.isfinite(values)
    normalized = np.zeros_like(values, dtype=np.uint8)
    if finite.any():
        low, high = np.percentile(values[finite], [10, 95])
        normalized[finite] = np.clip(
            (values[finite] - low) * 255.0 / max(high - low, 1e-6), 0, 255
        )
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    height, width = output_shape
    return cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_NEAREST)
