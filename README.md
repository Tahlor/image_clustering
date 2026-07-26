# image-clustering

Conservative clustering for repeated archival document captures, plus sequence-level
unique crop recovery for downstream recognition.

The package treats these as separate contracts:

1. `image_clustering.clustering` decides which nearby images show the same physical
   document scene.
2. `image_clustering.cropping` consumes those clusters and emits the minimum useful set
   of recognizer submissions.

The clustering API does not depend on the cropper. The cropper is allowed to reuse
accepted clustering transforms and to run its own registration/refinement when needed.

## Installation

```bash
pip install -e .
```

Development tools:

```bash
pip install -e ".[dev]"
```

## Clustering

```python
from image_clustering import ClusterConfig, cluster_directory

result = cluster_directory(
    "path/to/ordered-images",
    config=ClusterConfig(max_gap=3),
)

for cluster in result.clusters:
    print(cluster.cluster_id, cluster.image_ids)
```

By default, every immediate parent folder is an independent sequence. Images are sorted
by natural filename order and never clustered across those sequence boundaries.

For sampled datasets where each source row names a center image and its neighbors, pass
that manifest explicitly so unrelated triplets in one folder remain independent:

```python
result = cluster_directory(
    "path/to/sample",
    triplet_manifest="path/to/selected_images_500_with_neighbors_manifest.csv",
)
```

The manifest must contain `sampled_image`, `before_image`, and `after_image`. Paths are
matched against the discovered source paths by exact normalized path, then by the longest
unique suffix. Ambiguous or missing entries fail loudly.

Clustering output includes:

- stable source image IDs and sequence order;
- complete cluster membership;
- accepted and rejected nearby pair comparisons;
- source-pixel transforms for accepted direct registrations;
- registration, content-agreement, occlusion, and hard-contradiction diagnostics;
- the grouping mode and manifest path used for the run.

### Clustering invariants

These are correctness constraints, not optional heuristics:

- **Same form does not mean same record.** Different records can share an almost
  pixel-identical printed template while only handwritten or typewritten names, dates,
  places, signatures, or values differ. Those pairs must be rejected and must block an
  indirect transitive merge.
- **A real occlusion is large and contiguous.** It is normally one large polygon per
  page, usually rectangular but possibly skewed or mildly warped. It is typically at
  least about one third of the page and often one half or more.
- **Exterior agreement is the decisive signal.** After registration, most of the
  unoccluded form should match nearly pixel-for-pixel. A dense noisy region with a clean
  exterior is occlusion-like; residual or ink disagreement spread across the page is a
  different record.
- **A bounding box is not support.** Sparse handwriting changes can have a huge bounding
  rectangle. Candidate scoring must use the actual connected changed support, not every
  pixel or tile inside that rectangle.
- **Registration is necessary but not sufficient.** SIFT/RANSAC establishes geometric
  correspondence. It does not prove that document-specific ink belongs to the same
  physical record.

The July 2026 calibration, alternatives tested, timings, and threshold rationale are
recorded in [`docs/CLUSTERING_CALIBRATION_20260726.md`](docs/CLUSTERING_CALIBRATION_20260726.md).

## Unique crop recovery

```python
from image_clustering.cropping import CropConfig, recover_unique_crops

crop_result = recover_unique_crops(
    clustering=result,
    output_dir="outputs/crops",
    config=CropConfig(),
)
```

Or run the end-to-end command:

```bash
image-crop path/to/ordered-images outputs/crops \
  --cluster-json outputs/clustering.json
```

The cropper writes:

- `crops/<crop-id>.png` recognizer-ready images;
- `annotations/<cluster-id>.png` original views with exact crop polygons;
- `manifests/<cluster-id>.json` per-cluster decisions and diagnostics;
- `crops.json` machine-readable crop metadata;
- `manifest.json` the complete run result;
- `review.html` a static human-review page.

The cropper may emit:

- one base document crop when a single view contains the full readable page;
- one data-bearing overlay crop when an insert introduces unique text;
- one crop per distinct page state when an occluder hides data and no single view is
  complete;
- review-required best-available submissions when the evidence is not strong enough for
  an automatic complete answer.

It will not silently return zero crops for a non-empty cluster.

## Reviewing and correcting output

The static `review.html` is optimized for fast full-resolution inspection. It shows the
full source image with crop overlays, supports cluster-to-cluster navigation, and records
three decisions:

- cluster membership is correct;
- crop set is correct;
- edited crop rectangles in source-image coordinates.

Serve the output directory so the browser can persist labels through the review API:

```bash
image-review outputs/crops --port 8765
```

Then open `http://127.0.0.1:8765/review.html`. Decisions are appended to
`review_labels/clusters_reviewed.jsonl` and `review_labels/crops_reviewed.jsonl`, with the
latest revision for each key treated as authoritative. `review_labels/review_export.json`
is refreshed after every save for the next training or evaluation run.

## Configuration

Clustering configuration can be supplied as JSON:

```bash
image-cluster path/to/images outputs/clustering.json \
  --config configs/default.json
```

Crop configuration can be supplied as YAML or JSON:

```bash
image-crop path/to/images outputs/crops \
  --config configs/cropping_default.yaml
```

Defaults favor false splits over false merges. This is deliberate: merging different
records corrupts recognition context, while a split can be reviewed or rejoined later.
The checked-in clustering default is calibrated for large physical overlays and
same-template/different-record hard negatives; do not loosen it based only on aggregate
cluster count.

## Development

```bash
pytest
ruff check .
ruff format --check .
```

Design and calibration notes live in `docs/`.
