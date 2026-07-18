# image_clustering

A Python package for conservatively grouping ordered document images that show the **same physical document scene** under changing occlusions.

It is deliberately not a form-template clusterer. Two filled copies of the same printed form remain separate even when their layouts are almost identical.

## Package boundary

This repository owns only document-view clustering:

- discover independent filename-ordered sequences;
- compare nearby images within a sequence;
- estimate pairwise registration;
- decide whether two captures show the same physical document scene;
- form conservative graph components;
- expose accepted registrations and diagnostics to downstream modules.

It does **not** identify unique crops, choose model submissions, or extract records. A future crop module should consume `ClusteringResult`, inspect each `ImageCluster`, and use the accepted pair transforms to locate content that appears uniquely across views.

A cluster is never split merely because a downstream model accepts only four images. Batching is a downstream concern.

## Install

```bash
python -m pip install -e .
```

## Python API

```python
from pathlib import Path

from image_clustering import ClusterConfig, cluster_directory

result = cluster_directory(
    input_dir=Path("/path/to/images"),
    config=ClusterConfig(max_gap=3),
    cache_dir=Path("/path/to/output/.feature_cache"),
)

for cluster in result.clusters:
    images = result.images_for(cluster.cluster_id)
    registrations = result.accepted_comparisons(cluster.cluster_id)
```

`PairComparison.transform` is a 3×3 source-pixel transform mapping the second image into the first image. This is the intended handoff to crop-localization code.

## CLI

```bash
image-cluster \
  --input_dir /path/to/images \
  --output_dir /path/to/results \
  --config configs/default.json
```

The CLI writes:

- `clustering.json`: canonical, reloadable clustering result;
- `pair_scores.csv`: flat pairwise diagnostics;
- `run.json`: run summary and configuration.

Images in different parent folders are never compared. Images within each folder are sorted by filename. Candidate comparisons are limited to the next `max_gap` images, so runtime is linear in the sequence length for fixed `max_gap`.

## Safety against same-template merges

A pair must retain document-specific local feature support after geometric registration and must not exhibit page-wide disagreement. During graph construction, a well-registered negative pair with page-wide disagreement blocks a transitive bridge merge. Registration failure alone does not block a bridge because heavily occluded views may share no direct visible region.
