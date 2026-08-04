# Next-session prompt: finish the Vermont reviewed-real occlusion evaluation

You are taking over active engineering work in the public GitHub repository `Tahlor/image_clustering`.

Your job is **not** to add more framework-only code, merely rerun synthetic fixtures, or summarize what should eventually happen. Your job is to obtain the actual reviewed image bytes, run the current detector over the complete reviewed dataset, tune it on the permitted splits, inspect real failures visually, preserve false-link safety, publish reproducible outputs, and push durable work to GitHub as you go.

## GitHub workflow

Work against `master`. Push clean, coherent commits directly to `master` as work is completed. Do not leave successful work only on an agent branch, in a transient runtime, or in chat. Keep documentation and issue comments current.

Read these issues first:

- #16 — parent reviewed-real evaluation and calibration issue;
- #30 — materialize all 422 reviewed images and run the governed baseline;
- #31 — annotate accepted groups and classify every real detector failure;
- #32 — selection-only calibration and untouched locked audit;
- #33 — 500-image safety/runtime stress and final result bundle.

Read these repository files before changing anything:

- `analysis/reviewed_real_occlusion/README.md`
- `analysis/reviewed_real_occlusion/run_pipeline.py`
- `analysis/reviewed_real_occlusion/materialize_library_cache.py`
- `analysis/reviewed_real_occlusion/baseline_config.json`
- `analysis/reviewed_real_occlusion/materialization_receipts/local_cache_20260804.json`
- `analysis/reviewed_real_occlusion/materialization_receipts/vermont_52_cache_check_20260804.json`
- recent detector commits and analyses under `analysis/reviewed_real_occlusion/`.

## Authoritative data

The reviewed package is in ChatGPT Library at:

`/Vermont Naturalization/Images/Occlusion Review/MY_REVIEWED_GT_DATASET`

The truth contract is exact:

- 422 manifest rows;
- 200 independent `original_cluster_id` units;
- 134 accepted groups;
- 282 accepted images;
- 183 positive within-group pairs;
- 66 dissolved groups;
- 140 rejected images;
- 83 reviewed negative within-proposal pairs.

`assignments.csv` is authoritative. `assignments.jsonl` must match it exactly. `reviewed_decisions.json` is provenance only and must not expand the evaluated population. Exclude unreviewed, irregular, edited, generated, and unrelated images.

Split only by `original_cluster_id`, with sequence-family grouping preserved. Never split by image or pair.

## What is already implemented

The evaluation/calibration framework is already merged to `master`. It includes:

- strict 422/200/134/66/282/140/183/83 validation;
- canonical positive and reviewed-negative pair reconstruction;
- grouped development/selection/locked-audit splits;
- isolated staging of dissolved proposals;
- candidate/direct-edge/transitive component evaluation;
- review-only real calibration;
- reliability and risk-coverage reporting;
- zero-negative-edge and zero-contaminated-component promotion gates;
- cold/warm cache replay checks.

Recent detector fixes already merged to `master` include:

- paired handwriting/printed-text masks;
- localized inside/outside text mismatch evidence;
- page-wide same-template/different-record rejection;
- full-page record-replacement vetoes;
- low-contrast dense text-erasure candidate recovery;
- dirty-exterior identity gating;
- multi-page residual-versus-text candidate competition;
- graph bridge safeguards requiring persisted localized evidence;
- exact cache invalidation for detector implementation changes.

Do not casually revert these changes.

The previous runtime materialized only 35 genuine reviewed originals. That partial cache was tested against 77 deliberately difficult same-family hard-negative pairs:

- 22 registered and reached content scoring;
- zero deterministic same-document links;
- zero review candidates at `P(same_occluded) >= 0.08`;
- maximum `P(same_occluded) = 0.068613`.

This is a useful false-positive diagnostic only. It is **not** the governed 422-image evaluation and provides no authoritative positive-recall estimate.

The exact partial-cache hashes are committed at:

`analysis/reviewed_real_occlusion/materialization_receipts/local_cache_20260804.json`

A separate 52-image Vermont roster reconciled to 35 present / 17 missing and correctly failed closed. Its receipt is:

`analysis/reviewed_real_occlusion/materialization_receipts/vermont_52_cache_check_20260804.json`

## Non-negotiable behavior

The user is specifically dissatisfied with work being described as complete when only scaffolding, synthetic controls, or a partial subset was run. Therefore:

- do not claim the detector has been tuned on the reviewed data until you have actually run the real reviewed images;
- do not claim perfect fit, recall, precision, safety, or successful overfitting without measured outputs;
- do not substitute the 35-image diagnostic for the 422-image run;
- do not stop after writing a runner, materializer, test fixture, or report template;
- do not say the images are unavailable without exhausting File Library retrieval, mounted paths, cache paths, exact filename lookup, and durable copying;
- if a service temporarily fails, retry with bounded exponential backoff while continuing independent work;
- never fabricate results;
- push completed code, compact reports, configs, receipts, and issue updates to `master` as you go.

Tune aggressively on **development**. Use **selection** for calibrator and threshold choice. Keep the **locked audit untouched** until every detector and calibration choice is frozen. If you inspect locked-audit errors, explicitly rename that split as inspected and create a new untouched audit; never continue calling it untouched.

## Phase 1 — materialize all 422 images and run the baseline

Complete issue #30 first.

Use the strict materializer, for example:

```bash
python analysis/reviewed_real_occlusion/materialize_library_cache.py \
  --assignments-csv /path/to/MY_REVIEWED_GT_DATASET/assignments.csv \
  --scan-root /mnt/data \
  --scan-root /other/transient/cache/root \
  --output-root /durable/MY_REVIEWED_GT_DATASET_materialized \
  --require-complete
```

The governed run must not begin until the materializer reports:

- `expected = 422`;
- `materialized = 422`;
- `missing = 0`;
- `collisions = 0`.

Preserve package-relative paths or reconstruct a verified equivalent package root. Save SHA-256 ledgers and a machine-readable completeness receipt.

Then run:

```bash
python analysis/reviewed_real_occlusion/run_pipeline.py \
  --package-root /durable/MY_REVIEWED_GT_DATASET \
  --assignments-csv /durable/MY_REVIEWED_GT_DATASET/assignments.csv \
  --assignments-jsonl /durable/MY_REVIEWED_GT_DATASET/assignments.jsonl \
  --output-root /durable/MY_REVIEWED_GT_DATASET/Results/reviewed_real_current_master \
  --config analysis/reviewed_real_occlusion/baseline_config.json
```

Run cold and warm. Confirm exact replay. Save the result bundle adjacent to the package, not only in a transient environment. Commit compact metrics, receipts, reports, and output links to `master`. Update #30.

## Phase 2 — inspect and improve the detector on real data

Complete issue #31.

Visually inspect all 134 accepted groups and fill the generated subtype sidecar. Keep these categories distinct:

1. identical or near-identical;
2. material physical occlusion with a better view and meaningful hidden-content risk;
3. same-document state/content change;
4. visual-only number, stamp, or card overlay excluded from the required material-occlusion metric;
5. uncertain/other with explicit notes.

Large central numbers and similar stamps must remain a separate visual-only category. The user does not require the detector to identify these as meaningful hidden-content occlusion.

For every false negative, false positive, weak registration, hard contradiction, and near-threshold case:

- open the actual images;
- inspect registration and residual diagnostics;
- classify the mechanism failure;
- determine whether the failure is candidate generation, registration, localization, probability scoring, deterministic graph gating, or ground-truth subtype ambiguity;
- add a focused regression before changing code;
- rerun complete development and selection splits after every meaningful mechanism change;
- push the fix, tests, measured before/after results, and documentation to `master`.

The key conceptual target is **same document with a meaningful contiguous physical occlusion**, not merely generic same-document similarity.

A proposed physical occlusion should generally satisfy:

- one contiguous block explains a large share of unmatched text/ink;
- text outside that block agrees strongly;
- the block has plausible material/shape evidence;
- disagreement is not distributed across the whole form;
- same-template/different-record pages are rejected when names, dates, residences, signatures, and other filled text differ broadly.

Do not improve recall by merging different people or filled records that share the same printed template.

## Probability contract and graph safety

The probability semantics are:

- `s = P(same physical document)`;
- `q = P(meaningfully occluded | same physical document)`;
- `P(same clean) = s * (1 - q)`;
- `P(same occluded) = s * q`;
- `P(different) = 1 - s`.

Probabilities are review-only. They may rank or enqueue cases but may not create graph edges.

Automatic edges still require all of:

- deterministic same-document decision;
- automatic-link eligibility;
- no hard contradiction.

Hard promotion gates:

- zero automatic edges among all 83 reviewed negative pairs;
- zero contaminated reviewed components.

## Phase 3 — selection-only calibration and untouched audit

Complete issue #32 only after detector decisions are frozen.

Fit the real calibrator on the selection split only. Report reliability, calibration error, risk-coverage, candidate precision/recall, and review volume. Then evaluate the frozen system exactly once on the locked audit.

The locked-audit report must include:

- pair and group counts;
- same-document connected recall;
- candidate recall and precision;
- reviewed-negative false-link rate;
- reviewed-negative review-candidate rate;
- contaminated component count;
- calibration metrics;
- cold/warm runtime and exact-cache replay status.

Commit the calibrator provenance, input hashes, configuration fingerprint, compact metrics, and final report to `master`. Save larger result files adjacent to the dataset and link them from #32.

## Phase 4 — 500-image safety/runtime stress and final bundle

Complete issue #33.

Run the frozen detector over the established 500-image Vermont sample and relevant embedding neighbors. Include:

- same printed form with different records;
- true physical occlusion with a better view;
- identical scans;
- same-document state change;
- visual-only numbers/stamps;
- blur, crop, rotation, exposure, contrast, bleed-through, and geometric variation;
- multi-page cases and candidate competition.

Inspect every automatic false link and every high-scoring hard negative. Fix safety failures before promotion.

Publish a final reproducible bundle containing:

- authoritative manifests;
- exact config;
- code commit;
- cache/materialization receipts;
- subtype sidecar;
- baseline and calibrated pair/group metrics;
- locked-audit report;
- 500-image stress report;
- runtime/cache report;
- failure diagnostics;
- SHA-256 output ledger;
- plain-English recommendation: promote, promote only for review ranking, or do not promote.

Update #16 and close it only when #30–#33 are genuinely complete and all promotion gates pass.

## Current outstanding issues

- #30: complete 422-image materialization and governed baseline.
- #31: accepted-group subtype annotation and real failure diagnosis.
- #32: selection-only calibration and untouched locked audit.
- #33: 500-image safety/runtime stress and final bundle.

Begin with #30. Keep working continuously through the issues, pushing completed work to `master` as you go. Do not return with another framework-only status update.