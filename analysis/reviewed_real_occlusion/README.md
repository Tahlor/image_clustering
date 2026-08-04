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
- The real-data isotonic calibrator is fit on the selection split only.
- Baseline and calibration phases cannot emit locked-audit labels or metrics.
- The locked audit is available only through a separate frozen-system phase.

## Governed phase workflow

Install and validate the repository first:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m pytest -q
```

### 1. Baseline on development and selection only

The default phase stages the complete authority but runs and evaluates only the
development and selection groups. It does not fit calibration and cannot emit
locked-audit rows.

```bash
python analysis/reviewed_real_occlusion/run_pipeline.py \
  --phase baseline \
  --package-root /path/to/MY_REVIEWED_GT_DATASET \
  --assignments-csv /path/to/MY_REVIEWED_GT_DATASET/assignments.csv \
  --assignments-jsonl /path/to/MY_REVIEWED_GT_DATASET/assignments.jsonl \
  --output-root /path/to/MY_REVIEWED_GT_DATASET/Results/reviewed_real_v1 \
  --config analysis/reviewed_real_occlusion/baseline_config.json
```

Complete the generated
`prepared/accepted_group_occlusion_subtypes.csv` by visual inspection before
freezing calibration. The source accepted/rejected labels are never modified.

### 2. Selection-only calibration and freeze

This phase requires the completed subtype sidecar and the exact 40-character code
commit. The package root must also contain `completeness_receipt.json` and
`SHA256SUMS.tsv`. It fits from selection rows in the baseline pair-results file,
evaluates only development and selection, and writes `frozen_system_receipt.json`
with the source ledger, manifests, prepared split, predictions, calibration, and
configuration hashes.

```bash
python analysis/reviewed_real_occlusion/run_pipeline.py \
  --phase calibrate \
  --package-root /path/to/MY_REVIEWED_GT_DATASET \
  --assignments-csv /path/to/MY_REVIEWED_GT_DATASET/assignments.csv \
  --assignments-jsonl /path/to/MY_REVIEWED_GT_DATASET/assignments.jsonl \
  --output-root /path/to/MY_REVIEWED_GT_DATASET/Results/reviewed_real_v1 \
  --config analysis/reviewed_real_occlusion/baseline_config.json \
  --subtypes /path/to/completed_accepted_group_occlusion_subtypes.csv \
  --code-commit 0123456789abcdef0123456789abcdef01234567
```

### 3. Frozen locked audit

The audit phase verifies every frozen hash and the exact code commit before it
runs. It refuses to proceed if source bytes, calibration, configuration, manifests,
subtype evidence, split assignment, or baseline predictions changed. It writes a
`locked_audit_started_receipt.json` before evaluating the audit and refuses any
silent rerun after either the start or terminal execution receipt exists.

```bash
python analysis/reviewed_real_occlusion/run_pipeline.py \
  --phase locked-audit \
  --package-root /path/to/MY_REVIEWED_GT_DATASET \
  --assignments-csv /path/to/MY_REVIEWED_GT_DATASET/assignments.csv \
  --assignments-jsonl /path/to/MY_REVIEWED_GT_DATASET/assignments.jsonl \
  --output-root /path/to/MY_REVIEWED_GT_DATASET/Results/reviewed_real_v1 \
  --config analysis/reviewed_real_occlusion/baseline_config.json \
  --subtypes /path/to/completed_accepted_group_occlusion_subtypes.csv \
  --code-commit 0123456789abcdef0123456789abcdef01234567
```

The old one-shot baseline/calibration/audit behavior is intentionally unavailable.
Changing `--phase` is required to cross the locked-audit boundary.

The lower-level evaluator also accepts repeatable `--include-split` options:

```bash
image-reviewed-eval evaluate \
  --prepared-dir /path/to/prepared \
  --predictions /path/to/clustering.json \
  --output-dir /path/to/evaluation \
  --include-split development \
  --include-split selection
```

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
when it reports `expected=422`, `materialized=422`, `missing=0`, and
`collisions=0`:

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

The governed phases write the integrity report, canonical group and pair
manifests, leakage-safe split, subtype sidecar, baseline and final pair/group
results, failure analysis, calibration and reliability tables, risk-coverage
curve, runtime/cold-warm cache reports, frozen-system receipt, locked-audit
execution receipt, before/after reports, and `FINAL_REPORT.md`.

The package's rejected images are stored under singleton assignment directories.
The preparer therefore stages each reviewed original proposal as an isolated
evaluation sequence. This reconstructs the reviewed comparisons while preventing
unreviewed cross-proposal pairs from changing reviewed components. A reversible
`prediction_image_id` to canonical `image_id` map is written and enforced.

## Validation

Repository CI runs Ruff and the complete pytest suite on Python 3.10 and 3.12.
The focused reviewed-data tests use the exact package population shape and verify
that review probabilities cannot create automatic graph edges and that tuning
evaluation cannot expose the locked audit.

## Promotion gates

A final run exits nonzero unless both are true:

1. zero automatic edges among the 83 reviewed negative pairs;
2. zero contaminated reviewed negative components.

Recall and review-volume improvements are considered only after those gates pass.
