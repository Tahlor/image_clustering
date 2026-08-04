# Reviewed real occlusion evaluation

This analysis treats the 200 manually reviewed `original_cluster_id` values as
the independent evaluation units. It does not import the 1,001 unreviewed or 26
irregular source proposals. Issue #16 tracks the authoritative mounted-package run
and visual subtype review after this reproducible evaluation path is merged.

## Safety contract

- `assignments.csv` is authoritative and `assignments.jsonl` must match it.
- The exact 422/200/134/66/282/140/183/83 contract is a hard preflight.
- Accepted and dissolved proposal groups are never split across development,
  selection, or locked audit.
- Source sequence families stay in one split.
- Review probabilities never create graph edges.
- Automatic edges still require the production deterministic decision,
  automatic-link eligibility, and no hard contradiction.
- The real-data isotonic calibrator is fit on the selection split only. The
  locked audit is never used for fitting.

## Run

From the repository root:

```bash
python -m pip install -e ".[dev]"
python analysis/reviewed_real_occlusion/run_pipeline.py \
  --package-root /path/to/MY_REVIEWED_GT_DATASET \
  --assignments-csv /path/to/MY_REVIEWED_GT_DATASET/assignments.csv \
  --assignments-jsonl /path/to/MY_REVIEWED_GT_DATASET/assignments.jsonl \
  --output-root /path/to/MY_REVIEWED_GT_DATASET/Results/reviewed_real_v1 \
  --config analysis/reviewed_real_occlusion/baseline_config.json
```

The command is intentionally fail-closed: integrity drift, split leakage, malformed
predictions, negative automatic edges, and contaminated components stop the run.

Complete the generated
`prepared/accepted_group_occlusion_subtypes.csv` by visual inspection, then rerun
with `--subtypes` pointing to that completed sidecar. The source accepted/rejected
labels are never modified.

## Materialize from managed exact-key access

The cache materializer remains the authoritative completeness and SHA gate. When
File Library bytes are not already mounted, first generate exact broker requests
from the authoritative assignments manifest:

```bash
export VERMONT_IMAGE_ACCESS_TOKEN='<temporary token>'
python analysis/reviewed_real_occlusion/managed_download_urls.py \
  --assignments-csv /path/to/MY_REVIEWED_GT_DATASET/assignments.csv \
  --output-root /durable/MY_REVIEWED_GT_DATASET_materialized
```

The command performs no network requests. It validates that `source_project`,
`sequence_id`, `image_id`, and `package_relative_path` agree, fixes the authorized
endpoint and prefix, excludes already materialized destinations, and prints one
JSON request per missing image for the platform-managed downloader. The token is
read from the environment only. Never redirect token-bearing output to a file or
include it in logs, receipts, issue comments, or commits.

Download each printed exact URL to its stated `destination`, then run the strict
cache materializer over the download root. The governed evaluation may start only
when it reports `expected=422`, `materialized=422`, `missing=0`, and `collisions=0`:

```bash
python analysis/reviewed_real_occlusion/materialize_library_cache.py \
  --assignments-csv /path/to/MY_REVIEWED_GT_DATASET/assignments.csv \
  --scan-root /path/to/managed/downloads \
  --output-root /durable/MY_REVIEWED_GT_DATASET_materialized \
  --require-complete
```

The broker planner never lists the bucket, guesses an object key, writes remotely,
or changes the reviewed population.

## Outputs

The run writes the required integrity report, canonical group and pair manifests,
leakage-safe split, subtype sidecar, baseline and final pair/group results,
failure analysis, calibration and reliability tables, risk-coverage curve,
runtime/cold-warm cache report, before/after report, and `FINAL_REPORT.md`.

The package’s rejected images are stored under singleton assignment directories.
The preparer therefore stages each reviewed original proposal as an isolated
evaluation sequence. This reconstructs the reviewed comparisons while preventing
unreviewed cross-proposal pairs from changing reviewed components. A reversible
`prediction_image_id` to canonical `image_id` map is written and enforced.

## Validation

Repository CI runs Ruff and the complete pytest suite on Python 3.10 and 3.12.
The focused reviewed-data tests use the exact package population shape and verify
that review probabilities cannot create automatic graph edges.

## Promotion gates

A final run exits nonzero unless both are true:

1. zero automatic edges among the 83 reviewed negative pairs;
2. zero contaminated reviewed negative components.

Recall and review-volume improvements are considered only after those gates pass.
