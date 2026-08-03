# Reviewed real occlusion evaluation

This analysis treats the 200 manually reviewed `original_cluster_id` values as
the independent evaluation units. It does not import the 1,001 unreviewed or 26
irregular source proposals.

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

Complete the generated
`prepared/accepted_group_occlusion_subtypes.csv` by visual inspection, then rerun
with `--subtypes` pointing to that completed sidecar. The source accepted/rejected
labels are never modified.

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

## Promotion gates

A final run exits nonzero unless both are true:

1. zero automatic edges among the 83 reviewed negative pairs;
2. zero contaminated reviewed negative components.

Recall and review-volume improvements are considered only after those gates pass.
