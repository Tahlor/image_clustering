# image_clustering

A Python package for conservatively grouping ordered document images that show the **same physical document scene** under changing occlusions, then recovering the unique pages and data-bearing foreground sheets that should be submitted to recognition.

It is deliberately not a form-template clusterer. Two filled copies of the same printed form remain separate even when their layouts are almost identical.

## Package boundaries

### `image_clustering.clustering`

- discover independent filename-ordered sequences;
- compare nearby images within a sequence;
- estimate pairwise registration;
- distinguish near duplicates, physical occlusion states, and different filled documents;
- treat distributed handwriting disagreement as a hard contradiction;
- form conservative graph components;
- expose accepted registrations and diagnostics downstream.

### `image_clustering.cropping`

- consume `ClusteringResult`;
- align all views within a cluster;
- choose the best observation of each persistent page;
- recover distinct data-bearing foreground sheets;
- suppress reverse sheets, blank occluders, and duplicate states;
- guarantee that a content-bearing accepted cluster yields recognizer input, even when all views are literal duplicates;
- emit `partial_best_available` or review-required pages rather than silently calling an occluded page complete.

### `image_clustering.evaluation`

`examples/evaluation/reviewed_cases.jsonl` is the canonical append-friendly store for user-reviewed clusters, non-clusters, and exact crop targets. The private images are not committed; labels reference their filenames.

```bash
# Same template, different filled documents
image-label pair examples/evaluation/reviewed_cases.jsonl \
  image_a.j2k image_b.j2k --different

# Exact same filled document photographed twice
image-label pair examples/evaluation/reviewed_cases.jsonl \
  image_a.j2k image_b.j2k --near-duplicate

# Same physical scene with an overlay/occlusion
image-label pair examples/evaluation/reviewed_cases.jsonl \
  image_a.j2k image_b.j2k --occlusion

# Add an exact reviewed crop (normalized coordinates by default)
image-label crop examples/evaluation/reviewed_cases.jsonl \
  CASE_ID image_a.j2k 0.02 0.015 0.50 0.985 \
  --kind base_page --completeness complete --side left

image-label validate examples/evaluation/reviewed_cases.jsonl
image-label list examples/evaluation/reviewed_cases.jsonl
```

Every positive reviewed case requires `expected_min_submissions >= 1`. A near-duplicate cluster therefore cannot disappear from crop mode merely because it has no changed region.

## Install

```bash
python -m pip install -e .
```

## Python API

```python
from pathlib import Path

from image_clustering import (
    ClusterConfig,
    crop_clustering_result,
    cluster_directory,
)

result = cluster_directory(
    input_dir=Path("/path/to/images"),
    config=ClusterConfig(max_gap=3),
    cache_dir=Path("/path/to/output/.feature_cache"),
)

for cluster in result.clusters:
    images = result.images_for(cluster.cluster_id)
    registrations = result.accepted_comparisons(cluster.cluster_id)

crop_manifest = crop_clustering_result(
    clustering=result,
    output_dir=Path("/path/to/output/cropping"),
)
```

`PairComparison.transform` is a 3×3 source-pixel transform mapping the second image into the first image.

## CLI

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
  --output_dir /path/to/crop-results \
  --crop_config configs/cropping_default.yaml
```

The clustering CLI writes:

- `clustering.json`;
- `pair_scores.csv`;
- `run.json`.

The cropper writes crops, review-queue items, annotations, per-cluster manifests, and aggregate `cropping.json`.

Images in different parent folders are never compared. Images within each folder are sorted by filename. Candidate comparisons are limited to the next `max_gap` images, so runtime is linear in sequence length for fixed `max_gap`.

## Safety against same-template merges

Feature overlap is used to establish registration, not to prove identity. After registration, the scorer compares locally normalized ink and gradients. A pair is accepted only as either:

- a near duplicate with essentially identical document-specific ink; or
- the same scene with a coherent physical occlusion and near-exact agreement outside it.

Distributed handwriting, names, dates, and signatures that do not match are hard-negative evidence. A registered hard contradiction prevents a transitive graph bridge. Registration failure alone does not block a bridge because heavily occluded views may share little direct visible content.
