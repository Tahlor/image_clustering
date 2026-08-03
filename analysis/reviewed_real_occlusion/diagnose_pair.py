"""Render text-localization diagnostics for one registered image pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from image_clustering.clustering.config import ClusterConfig
from image_clustering.clustering.content_geometry import _tile_bounds
from image_clustering.clustering.content_grid import compute_content_grid
from image_clustering.clustering.content_pages import select_page_candidates
from image_clustering.clustering.features import extract_features
from image_clustering.clustering.models import ImageItem
from image_clustering.clustering.registration import register_pair, warp_current
from image_clustering.clustering.scoring import _normalize_brightness, score_pair


def _item(path: Path, index: int) -> ImageItem:
    return ImageItem(
        image_id=path.name,
        path=path,
        sequence_id="diagnostic_pair",
        sequence_index=index,
    )


def _label(image: np.ndarray, text: str) -> np.ndarray:
    output = image.copy()
    cv2.putText(
        output,
        text,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def _diagnostic_panel(
    reference: np.ndarray,
    aligned: np.ndarray,
    valid_mask: np.ndarray,
    config: ClusterConfig,
) -> np.ndarray | None:
    grid = compute_content_grid(reference, aligned, valid_mask, config)
    if grid is None:
        return None
    selected, _, _ = select_page_candidates(
        reference=reference,
        aligned=aligned,
        grid=grid,
        config=config,
    )
    candidate = np.zeros_like(grid.core)
    rows, columns = grid.valid_tiles.shape
    for item in selected:
        support = item["support"]
        assert isinstance(support, np.ndarray)
        for row, column in zip(*np.where(support), strict=True):
            x0, y0, x1, y1 = _tile_bounds(
                row,
                column,
                reference.shape,
                rows,
                columns,
            )
            candidate[y0:y1, x0:x1] = True

    reference_rgb = _label(
        cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR),
        "reference",
    )
    aligned_rgb = _label(
        cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR),
        "registered second view",
    )
    mismatch = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    mismatch[grid.mismatch] = (0, 0, 255)
    mismatch[candidate & ~grid.mismatch] = (0, 255, 255)
    mismatch = _label(mismatch, "red mismatch; yellow candidate")

    localization = cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
    localization[grid.mismatch & candidate] = (0, 255, 0)
    localization[grid.mismatch & ~candidate] = (255, 0, 255)
    localization = _label(
        localization,
        "green inside mismatch; magenta outside",
    )
    return np.vstack(
        [
            np.hstack([reference_rgb, aligned_rgb]),
            np.hstack([mismatch, localization]),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score a pair and render the text-channel evidence that supports or "
            "rejects a physical-occlusion explanation."
        )
    )
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    config = ClusterConfig.from_json(args.config)
    first = args.first.resolve()
    second = args.second.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    first_features = extract_features(_item(first, 0), config)
    second_features = extract_features(_item(second, 1), config)
    comparison = score_pair(
        previous=first_features,
        current=second_features,
        index_gap=1,
        config=config,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(comparison.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    registration = register_pair(first_features, second_features, config)
    if not registration.accepted:
        return 0
    aligned, valid_mask = warp_current(
        current_gray=second_features.gray,
        registration=registration,
        previous_shape=first_features.gray.shape,
    )
    aligned = _normalize_brightness(
        reference=first_features.gray,
        aligned=aligned,
        valid_mask=valid_mask,
    )
    panel = _diagnostic_panel(
        reference=first_features.gray,
        aligned=aligned,
        valid_mask=valid_mask,
        config=config,
    )
    if panel is not None:
        cv2.imwrite(str(output_dir / "text_localization.png"), panel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
