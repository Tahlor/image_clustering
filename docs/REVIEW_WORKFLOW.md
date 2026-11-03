# Physical-document duplicate/occlusion review — operating workflow

This is the detailed operating guide for the standalone **Physical Document Duplicate/Occlusion Review Tool**. For the identity, repository boundary, one-command launch contract, no-duplicate-server rule, and first verification checks, start with the canonical [`REVIEW_TOOL.md`](../REVIEW_TOOL.md) guide.

`image-review` is the purpose-built human-review tool in this repository. It is a local, single-user HTTP application for inspecting automatically formed clusters and proposed crop boxes, recording membership and geometry decisions, and exporting corrected manifests. It is the recommended interactive review surface for a completed `image_clustering` evaluation.

This workflow is independent of Vantage2. Do not substitute a Vantage2 route, a generic image viewer, or an extraction-batch result for the evaluation directory described below.

## Current accepted-cluster subtype gap

The underlying clustering model already separates two same-document pair cases:

- `near_duplicate`: near-exact document-specific ink agreement, corresponding to repeated captures of the same visible document;
- `physical_occlusion`: a large contiguous physical occlusion with near-exact agreement outside the occluded area, allowing complementary views of the same physical document scene.

The continuous candidate model exposes `same_clean_probability` and `same_occluded_probability`, and the evaluation fixture records `same_document_near_duplicate` versus `same_document_occlusion`. This is useful evidence that the model is making the requested distinction, but it is not currently carried through the interactive review decision.

The review UI's **Mark OK** action stores only the generic cluster status `approved` (or `edited` after membership changes). The persisted decision schema and corrected exports have no accepted-cluster relation subtype; the remaining cluster statuses are `unreviewed`, `dissolved`, and `irregular`. Consequently, a current review can confirm that a grouping is acceptable but cannot durably designate it as identical/near-duplicate versus partially occluded. Do not use `approved`, crop completeness, or printed-form similarity as a proxy for that subtype. The current physical-occlusion scorer is also intentionally narrow: it requires contiguous occlusion evidence and strong agreement outside the changed region.

This is a known documentation/design boundary, not an implemented review feature. A future change should keep the subtype separate from membership disposition and specify aggregation for clusters whose accepted pair evidence includes both relation types.

## Status and safety

- **Status:** active local workflow.
- **Entry point:** `image-review`, defined in `pyproject.toml` as `image_clustering.review.cli:main`.
- **Runtime:** Python 3.10 or newer with this package installed; see [`DEVELOPMENT.md`](DEVELOPMENT.md).
- **Network:** the server binds to `127.0.0.1` by default and has no authentication. Keep it on loopback unless exposure has been explicitly security-reviewed. Do not treat it as a hosted service.
- **Persistence:** changes are written to the output directory. Start with a copy or a clearly named evaluation output when the source decisions must remain untouched.
- **Canonical artifacts:** the review tool never rewrites `clustering/clustering.json`, `cropping.json`, or the generated report files. Reviewer state and derived exports live under `review_labels/`.

## Choose the right input

The interactive server expects a completed evaluation output root, not a field-extraction JSONL file and not a directory containing only source images.

### Recommended evaluation root

A full run produced by `scripts/evaluate_dataset.py` normally contains:

```text
<output_root>/
├── clustering/
│   ├── clustering.json                 # required to load clusters
│   └── run.json                         # clustering provenance
├── inventory/
│   └── dataset_inventory.json           # dimensions and source-path inventory
├── reports/
│   └── crops_for_recognizer.jsonl       # automatic crop boxes, when available
├── cropping.json                         # cropper result
├── review/                               # generated static HTML review, if produced
└── review_labels/                        # interactive review state and exports
```

The minimum input needed for the server to start is `clustering/clustering.json`. The other two files are strongly recommended for a useful cluster-and-crop review:

| Input | Required? | Used for |
| --- | --- | --- |
| `clustering/clustering.json` | **Yes** | Cluster membership, stable image IDs, source paths, sequence order, accepted comparisons, and confidence ordering |
| `inventory/dataset_inventory.json` | No, but recommended | Original source width/height. Without it, box dimensions may be unavailable and geometry validation is weaker. |
| `reports/crops_for_recognizer.jsonl` | No, but recommended | Automatic crop boxes and crop metadata. Without it, clusters can still be reviewed but no automatic boxes are shown. |
| Referenced source images | **Yes for usable previews** | The server validates that preview requests refer to images in the loaded run and generates cached JPEG previews. |

`build_review_dataset()` fails clearly when `clustering/clustering.json` is absent. Missing optional inventory or crop reports degrade the review rather than creating those artifacts automatically.

## Start the server

Install the package first from the repository root:

```bash
python -m pip install -e ".[dev]"
```

Launch the default local server and browser:

```bash
image-review --output_dir /path/to/completed-evaluation
```

The command accepts these flags:

```text
--output_dir PATH       completed evaluation root (required)
--host HOST             default: 127.0.0.1
--port PORT             default: 8756; use 0 to request a free port
--no_browser            do not open the browser automatically
```

The server logs the actual URL and the decision path. If the default port is already occupied, stop the old review server or use a free port:

```bash
image-review \
  --output_dir /path/to/completed-evaluation \
  --port 0
```

On Windows PowerShell, use the same command with backticks for continuation or put it on one line. Stop the foreground server with `Ctrl+C`; the process closes its socket without deleting review state.

For compatibility, `python scripts/review_clusters.py ...` forwards to the same entry point. New documentation should use `image-review` directly.

## Review a cluster

The UI starts with the weakest accepted-registration confidence. Use the filters for minimum cluster size, status, source folder, and confidence. The visible progress line separates cluster review from bounding-box approval.

### Cluster and membership decisions

| Action | Button | Keyboard |
| --- | --- | --- |
| Move to next/previous cluster | `Next` / `Prev` | `j` / `k` |
| Confirm the automatic grouping (generic approval; no subtype) | `Mark OK` | `a` |
| Reject the grouping | `Dissolve` | `d` |
| Mark an irregular grouping and exclude it from corrected exports | `Irregular` | `i` |
| Return a decision to the original unreviewed state | `Reopen` | `r` |
| Include/exclude the focused capture | `Toggle capture` | `x`, or its `1`–`9` key |
| Switch side-by-side and focused views | `View` | `v` |
| Move between captures in a cluster | capture arrows | Left/Right arrows |
| Undo the last saved change | `Undo` | `Ctrl+Z` |

Membership changes are independent of crop-box approval. Excluding one member from a two-image cluster dissolves that grouping because a one-image remainder is not a cluster; larger groups can remain as edited clusters.

### Box decisions

1. Press `e` or choose **Edit boxes**.
2. Focus a capture, select a box with `Tab`, and drag a box or one of its eight handles. Drag empty space to create a box.
3. Use arrow keys to nudge a selected box; hold `Shift` to resize it.
4. Press `Enter` or release the pointer to save. Press `Escape` to cancel a pending edit.
5. Use `b` to approve or unapprove the focused capture's box set. Use `Delete` to remove the selected box.

Boxes are validated against the original source dimensions when inventory dimensions are available. Editing boxes does not approve the cluster membership.

**Bulk actions are consequential.** `Mark remaining clusters OK` and `Mark remaining boxes OK` approve every still-unreviewed item in the requested scope. Use them only after the queue has been inspected.

## Persistence, provenance, and exports

The server creates or loads:

```text
<output_root>/review_labels/decisions.json
```

Every decision response saves the JSON document atomically through a temporary file replacement. The document contains schema version, cluster decisions, timestamps, and provenance including:

- output root and input root;
- clustering configuration fingerprint;
- cluster and image counts;
- review-tool revision.

The **Export corrected manifests** action writes these derived artifacts under the same `review_labels` directory:

| Artifact | Meaning |
| --- | --- |
| `clusters_reviewed.jsonl` | Corrected cluster membership records. Excluded members and dissolved clusters become standalone records so source images are not silently dropped. |
| `crops_reviewed.jsonl` | Per-image boxes after reviewer edits, with source dimensions, membership disposition, and box status. |
| `excluded_from_training_validation.jsonl` | Audit of clusters marked `irregular`; these are intentionally omitted from corrected cluster/crop manifests. |
| `review_summary.csv` | One summary row per original cluster, including status, confidence, exclusions, and bbox states. |
| `review_export.json` | Manifest with export timestamp, provenance, progress counts, final counts, and artifact paths. |

The decision document is the source of truth for an in-progress review. The exported manifests are derived state and should be regenerated after changing decisions. A successful export is not a validation of the underlying clustering or crop quality; it is a reproducible record of reviewer disposition.

## Static HTML review is a separate surface

`scripts/evaluate_dataset.py` also generates a static review bundle under `<output_root>/review/`, including:

- `review/index.html` for a filterable cluster browser;
- `review/clusters/` for cluster detail pages;
- `review/crops/` for crop pages;
- `reports/review_decisions.json` as a review template.

That generated browser stores its labels in browser `localStorage` and offers client-side downloads. It does **not** share the interactive server's `review_labels/decisions.json` automatically. Use `image-review` when durable server-side autosave and corrected manifests are required.

## Validation and troubleshooting

### Focused tests

Run the review contract tests from the repository root:

```bash
python -m pytest -q tests/test_review_server.py tests/test_review_decisions.py
```

The tests cover dataset loading, preview isolation, autosave, membership semantics, irregular exclusions, box validation, undo, URL-encoded image IDs, port collisions, and export behavior.

### Common failures

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `No clustering output to review .../clustering/clustering.json` | The path is a source directory, extraction run, or incomplete output root | Run the evaluation workflow first or point `--output_dir` at a completed clustering evaluation. |
| Browser does not open | Browser launch was disabled or unavailable in the environment | Use the logged URL, or pass `--no_browser` and open the URL manually. |
| Port bind error | Another review server owns the configured port | Stop the stale server or pass `--port 0`. Do not assume a second server replaced the first one. |
| Images show no preview | Referenced source image is missing/unreadable, or the path is outside the loaded run | Restore the source image or regenerate the evaluation with valid absolute source paths. |
| Boxes have no dimensions | Inventory file is absent or does not contain the source path | Regenerate the full evaluation inventory; do not hand-edit canonical clustering output. |
| Old decisions appear | The output root already contains review state | Inspect the provenance in `review_labels/decisions.json`, back it up if needed, or use a new output root. |

## Related implementation

- [`src/image_clustering/review/cli.py`](../src/image_clustering/review/cli.py) — flags, server lifecycle, browser launch, and initial autosave;
- [`src/image_clustering/review/dataset.py`](../src/image_clustering/review/dataset.py) — evaluation-directory loading and payload construction;
- [`src/image_clustering/review/app.py`](../src/image_clustering/review/app.py) — browser controls and keyboard behavior;
- [`src/image_clustering/review/server.py`](../src/image_clustering/review/server.py) — loopback HTTP operations and preview isolation;
- [`src/image_clustering/review/decisions.py`](../src/image_clustering/review/decisions.py) — decision/status semantics and bbox validation;
- [`src/image_clustering/review/store.py`](../src/image_clustering/review/store.py) — atomic JSON persistence;
- [`src/image_clustering/review/exports.py`](../src/image_clustering/review/exports.py) — corrected manifest generation;
- [`tests/test_review_server.py`](../tests/test_review_server.py) and [`tests/test_review_decisions.py`](../tests/test_review_decisions.py) — executable contract.
