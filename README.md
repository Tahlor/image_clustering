# image_clustering

A Python package for conservatively grouping ordered document images that show the **same physical document scene** under changing occlusions, then recovering each unique front-facing page or inserted sheet once.

It is deliberately not a form-template clusterer. Two filled copies of the same printed form remain separate even when their layouts are almost identical.

## Package boundaries

`image_clustering.clustering`:

- discovers independent filename-ordered sequences;
- compares nearby images within each sequence;
- estimates pairwise registration;
- decides whether captures show the same physical scene;
- creates conservative graph components.

`image_clustering.cropping`:

- consumes `ClusteringResult` rather than reclustering globally;
- performs stricter page-level registration within each cluster;
- detects coherent changed regions on a tile grid;
- trims low-mass residual tails and expands toward paper boundaries;
- chooses the best front-facing observation of each persistent page;
- emits distinct data-bearing overlays once;
- suppresses reverse sheets and blank/translucent occluders;
- marks a page `partial_best_available` when a broad occlusion persists in every view.

Images in different immediate parent folders are never compared. Nearby comparisons are bounded by `max_gap` for clustering and `pair_search.window` for crop-localization refinement.

## Install

```bash
python -m pip install -e .
```

## End-to-end CLI

```bash
image-crop \
  --input_dir /path/to/images \
  --output_dir /path/to/results \
  --cluster_config configs/default.json \
  --crop_config configs/cropping_default.yaml \
  --cache_dir /path/to/results/.feature_cache \
  --show_progress
```

The cropper writes:

- `clustering/clustering.json`: upstream cluster handoff;
- `cropping.json`: aggregate crop manifest;
- `cluster_manifests/`: resumable per-cluster results;
- `submissions/`: complete and explicitly partial submissions;
- `review_queue/`: uncertain outputs when enabled by future policies;
- `annotated/`: source images with submission boxes;
- `diagnostics/`: optional tile residual heatmaps.

## Python API

```python
from pathlib import Path

from image_clustering import ClusterConfig, crop_directory, load_crop_config

manifest = crop_directory(
    input_dir=Path("/path/to/images"),
    output_dir=Path("/path/to/results"),
    cluster_config=ClusterConfig(max_gap=3),
    crop_config=load_crop_config(),
    cache_dir=Path("/path/to/results/.feature_cache"),
    show_progress=True,
)
```

Or crop an existing clustering result:

```python
from image_clustering import crop_clustering_result, load_result

clustering = load_result(Path("results/clustering.json"))
cropping = crop_clustering_result(
    clustering,
    output_dir=Path("results"),
)
```

## Pilot calibration

The cropper was calibrated against the repository's archival occlusion pilot: eight related groups and twenty curated submission states, including a three-image sequence with two distinct overlays and a nonadjacent direct relationship. The regression target is the **submission set**, not semantic document classification; the implementation uses no OCR.

Treat this as a pilot-calibrated automatic route. Before an 80K-image production run, audit a representative sample and measure false-complete pages, missed overlays, duplicates, crop containment, and cluster precision/recall.
