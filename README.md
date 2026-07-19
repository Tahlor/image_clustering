# image_clustering

A Python package for conservatively grouping ordered document images that show the **same physical document scene** under changing occlusions, then recovering unique page and overlay crops for downstream recognition.

It is deliberately not a form-template clusterer. Two filled copies of the same printed form remain separate even when their layouts are almost identical.

## Package boundaries

### `image_clustering.clustering`

- discovers independent filename-ordered sequences;
- compares nearby images within each sequence;
- estimates pairwise registration;
- verifies document-specific ink agreement;
- accepts either near-exact duplicate views or changes explained by a coherent physical occlusion;
- blocks graph bridges when registered images show distributed handwriting/content disagreement;
- forms conservative graph components;
- exposes accepted registrations and diagnostics.

### `image_clustering.cropping`

- consumes `ClusteringResult`;
- performs stricter page-level registration within each cluster;
- identifies persistent pages and data-bearing foreground sheets;
- suppresses reverse sheets, blank/translucent occluders, and duplicate page states;
- emits complete and `partial_best_available` recognizer crops;
- writes resumable manifests, annotations, and optional diagnostics.

A cluster is never split merely because a downstream model accepts only four images. Batching is a downstream concern.

## Install

```bash
python -m pip install -e .
```

## Python API

```python
from pathlib import Path

from image_clustering import (
    ClusterConfig,
    cluster_directory,
    crop_clustering_result,
)

result = cluster_directory(
    input_dir=Path("/path/to/images"),
    config=ClusterConfig(max_gap=3),
    cache_dir=Path("/path/to/output/.feature_cache"),
)

crop_manifest = crop_clustering_result(
    clustering=result,
    output_dir=Path("/path/to/output/cropping"),
)
```

`PairComparison.transform` is a 3×3 source-pixel transform mapping the second image into the first image. Pair diagnostics now include unmatched-ink, residual-tile, occlusion-geometry, material-change, outside-occlusion agreement, decision-branch, and hard-contradiction fields.

## CLIs

Clustering:

```bash
image-cluster \
  --input_dir /path/to/images \
  --output_dir /path/to/results \
  --config configs/default.json
```

Cropping from a saved clustering result:

```bash
image-crop \
  --clustering_json /path/to/results/clustering.json \
  --output_dir /path/to/results/cropping \
  --crop_config configs/cropping_default.yaml
```

The clustering CLI writes:

- `clustering.json`: canonical, reloadable clustering result;
- `pair_scores.csv`: flat pairwise diagnostics;
- `run.json`: run summary and configuration.

Images in different parent folders are never compared. Images within each folder are sorted by filename. Candidate comparisons are limited to the next `max_gap` images, so runtime is linear in the sequence length for fixed `max_gap`.

## Safety against same-template merges

SIFT support establishes that two views can be registered; it does not prove that the filled physical document is the same. Pair acceptance has two explicit branches:

1. **Near duplicate:** document-specific ink and gradients agree almost exactly after registration.
2. **Physical occlusion:** a coherent page-local sheet or overlay explains the disagreement, and meaningful ink agrees outside that region.

Distributed handwriting, signature, name, or date disagreement is a hard negative signal. A well-registered hard contradiction blocks a transitive graph bridge. Registration failure alone does not block a bridge because heavily occluded views may share little direct visible content.
