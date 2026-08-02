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

Images in different parent folders are never compared. Images within each folder are sorted by filename. Candidate comparisons are limited to the next `max_gap` images, so runtime is linear in the sequence length for fixed `max_gap`.

## Clustering invariants

These are correctness constraints, not optional heuristics:

- **Same form does not mean same record.** Different records can share an almost pixel-identical printed template while only handwritten or typewritten names, dates, places, signatures, or values differ. Those pairs must be rejected and must block an indirect transitive merge.
- **A real occlusion is large and contiguous.** It is normally one large polygon per page, usually rectangular but possibly skewed or mildly warped. It is typically at least about one third of the page and often one half or more.
- **Exterior agreement is decisive.** After registration, most of the unoccluded form should match nearly pixel-for-pixel. A dense noisy region with a clean exterior is occlusion-like; residual or ink disagreement spread across the page is a different record.
- **A bounding box is not support.** Sparse handwriting changes can have a huge bounding rectangle. Candidate scoring must use the actual connected changed support, not every pixel or tile inside that rectangle.
- **Registration is necessary but not sufficient.** SIFT/RANSAC establishes geometric correspondence. It does not prove that document-specific ink belongs to the same physical record.

A pair is accepted only as either a near duplicate with essentially identical document-specific ink, or the same scene with a large coherent physical occlusion and near-exact agreement outside it. A registered hard contradiction prevents a transitive graph bridge. Registration failure alone does not block a bridge because heavily occluded views may share little direct visible content.

Automatic edges additionally require plausible capture-sequence proximity. Distant filename states and full-page ECC-only matches remain continuous-score review candidates rather than entering the graph. The rare ECC fallback runs on a bounded canvas and is serialized, while ordinary pair scoring remains parallel.

The July 2026 calibration, alternatives tested, timing, and threshold rationale are recorded in [`docs/CLUSTERING_CALIBRATION_20260726.md`](docs/CLUSTERING_CALIBRATION_20260726.md).

## Explicit neighbor-group manifests

Folder ordering remains the default. For curated neighbor samples, pass a CSV manifest with `source_sample_row`, `neighbor_of`, `media_item_id`, and `filename`; optional `relative_path` and `sequence_index` columns disambiguate and order images. Manifest groups are independent and may contain any number of images unless `ClusterConfig.max_cluster_size` is explicitly set.

```bash
image-crop \
  --input_dir /path/to/images \
  --triplet_manifest /path/to/neighbor_groups.csv \
  --output_dir /path/to/results \
  --cluster_config configs/default.json \
  --crop_config configs/cropping_default.yaml
```

## Evaluation and review

`scripts/evaluate_dataset.py` creates the inventory, clustering/crop reports, validation output, and fast HTML review for a complete run. The local review application then supports keyboard-first membership decisions, bounding-box editing, undo, irregular exclusions, and corrected exports without modifying canonical clustering or crop artifacts.

```bash
image-review --output_dir /path/to/completed-evaluation
```

Review decisions and corrected exports are written under `<output_dir>/review_labels/`.
