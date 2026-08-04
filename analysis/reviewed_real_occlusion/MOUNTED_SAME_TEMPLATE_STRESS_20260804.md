# Mounted same-template and physical-sheet stress cohort — 2026-08-04

## Purpose

This focused cohort was run after visually identifying a recurring Vermont failure:
two scans share an almost identical declaration form, but all record-specific names,
dates, places, family rows, signatures, and sometimes the photograph differ. The
important question is whether one physical block explains the disagreement, not
whether the printed template registers.

This is a real-image mechanism stress test, not the complete governed 422-image
selection/locked-audit run.

## Real images

Seven mounted Washington County declaration scans from the same film/form family:

- `i4071659-00042(1).jpg`
- `i4071659-00229(1).jpg`
- `i4071659-00438(1).jpg`
- `i4071659-00701(1).jpg`
- `i4071659-01057(1).jpg`
- `i4071659-01096(1).jpg`
- `i4071659-01126(1).jpg`

All 21 pairwise combinations are different filled records. The cohort intentionally
contains the dangerous photo/no-photo case and highly similar printed layouts.

## Method

The images were compared at the production 900-pixel working dimension using the
current pooled handwriting/print channel, tolerant ink mismatch, 18x24 residual grid,
page-local candidate construction, inside mismatch capture, exterior agreement, and
localized occlusion-evidence formula. SIFT/homography registration was cached across
the cohort.

For each real image, a matched positive control was made by placing a large physical
sheet over a fixed region of the same scan. Additional full-page controls varied page,
brightness, and material contrast. These controls test the mechanism only; they do not
replace reviewed real positives.

## Results after the final full-page veto

### Same-template different-record negatives

All 21 negatives fall between **0.0000 and 0.0415** occlusion evidence.
Mean evidence is **0.0264**.

| First | Second | Final evidence |
|---|---|---:|
| 00042 | 00229 | 0.0184 |
| 00042 | 00438 | 0.0415 |
| 00042 | 00701 | 0.0378 |
| 00042 | 01057 | 0.0391 |
| 00042 | 01096 | 0.0388 |
| 00042 | 01126 | 0.0347 |
| 00229 | 00438 | 0.0361 |
| 00229 | 00701 | 0.0387 |
| 00229 | 01057 | 0.0346 |
| 00229 | 01096 | 0.0000 |
| 00229 | 01126 | 0.0340 |
| 00438 | 00701 | 0.0344 |
| 00438 | 01057 | 0.0203 |
| 00438 | 01096 | 0.0367 |
| 00438 | 01126 | 0.0000 |
| 00701 | 01057 | 0.0353 |
| 00701 | 01096 | 0.0000 |
| 00701 | 01126 | 0.0000 |
| 01057 | 01096 | 0.0346 |
| 01057 | 01126 | 0.0000 |
| 01096 | 01126 | 0.0389 |

### Matched physical-sheet controls

All seven controls fall between **0.8375 and 0.9424** evidence.
Mean evidence is **0.8724**.

| Source | Evidence | Mismatch captured | Outside text-union mismatch |
|---|---:|---:|---:|
| 00042 | 0.8375 | 1.000 | 0.000 |
| 00229 | 0.8463 | 1.000 | 0.000 |
| 00438 | 0.8483 | 1.000 | 0.000 |
| 00701 | 0.8807 | 0.979 | 0.005 |
| 01057 | 0.9424 | 0.989 | 0.004 |
| 01096 | 0.8643 | 0.977 | 0.007 |
| 01126 | 0.8870 | 1.000 | 0.000 |

## Failures found and fixed

### Resolution-sensitive two-page shortcut

`00042` versus `00229` scored **0.0184** at 900 pixels but **0.6749** at 600
pixels. At the lower resolution, distributed record text satisfied the broad
full-page material shortcut on both pages.

The false state had:

- 73.5% mismatching text tiles;
- material median 0.0268;
- inside unmatched-text ratio 0.337;
- 99.8% candidate area.

PR #22 added a conservative full-page record-replacement veto.

### Photo/no-photo full-page failure

`00042` versus `00438` initially scored **0.8290** at 900 pixels. The printed form is
the same, but the records are different throughout and only one scan contains a
photograph. It is not an occluded copy of the same physical record.

The false state had:

- 86.8% mismatching text tiles;
- material median 0.0648;
- inside unmatched-text ratio 0.493;
- 99.9% candidate area.

Low-contrast true page controls established a safer escape boundary:

- page sheet with material median 0.099 and inside ratio 0.620: evidence 0.990;
- lower-material sheet with median 0.0574 but only 43.8% distributed text tiles:
  evidence 0.951.

PR #23 therefore requires a page-wide state to show material median at least 0.08 or
inside text replacement at least 0.60 when mismatch is distributed across at least
55% of text tiles. The false pair drops to approximately **0.0415**.

## Production changes resulting from the cohort

- PR #21 persisted localization metrics through pair serialization, cache/replay,
  review reconstruction, and graph decisions; it also made state-graph bridges fail
  closed when old caches lack localized evidence.
- PR #22 added the first reduced-resolution full-page text-replacement veto.
- PR #23 tightened the boundary using the complete 21-pair real cohort and weak
  material-sheet controls.

No global candidate probability threshold was moved.

## Remaining required work

The complete 422-image package must still be recomputed with current `master`, because
old pair caches lack the persisted localization fields. Threshold selection must use
only the designated selection groups, followed by untouched locked-audit reporting.
The complete run must report candidate recall, direct-edge recall, transitive group
recall, false automatic edges, contaminated components, review volume, and failure
subtypes. This cohort validates and improves the observed mechanism; it is not a
substitute for those final metrics.
