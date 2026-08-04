# Mounted low-contrast sheet stress cohort — 2026-08-04

## Purpose

This cohort tests a specific recall failure: a real paper overlay can be close to the
underlying page tone. It erases a dense rectangle of printed and handwritten text,
but the grayscale residual map may fragment or disappear. Candidate generation must
use the text channel without turning distributed same-form record differences into
false physical occlusions.

This is a mounted-image mechanism test, not the complete governed 422-image run.

## Real source families

Eleven mounted Vermont scans were used across three form families:

- five Petitions for Naturalization;
- three Affidavit/Oath pages;
- three Declarations of Intention.

The negative cohort also includes the seven Washington County declaration scans from
`MOUNTED_SAME_TEMPLATE_STRESS_20260804.md`.

## Before the fallback

Each real source image received a matched large paper-sheet control. Eight controls
were detected strongly, but three produced no occlusion candidate at all:

| Source | Candidate before | Same-and-occluded after fallback |
|---|---:|---:|
| `i4399024-00542.jpg` | none | 0.9837 |
| `i4399014-00153.jpg` | none | 0.9578 |
| `i4399024-00720.jpg` | none | 0.9705 |

For these controls, the text mismatch formed a dense rectangular band while the
residual map contained too few connected changed tiles.

## Implemented fallback

PR #24 adds a page-local dense text-erasure fallback. It runs only when the ordinary
residual component finder returns no candidate for the page.

A fallback rectangle requires:

- contiguous row mismatch density of at least 0.40;
- contiguous column mismatch density of at least 0.35;
- at least three tile rows;
- at least 35% of the page width;
- at least 45% mismatching valid tiles inside the rectangle;
- the existing minimum page-area requirement;
- no more than a one-tile internal gap.

The rectangle is filled as candidate support only after both projection and occupancy
criteria establish a dense block. Candidate generation does not create an edge. The
existing registration, mismatch-capture, material, exterior-agreement, full-page
record-replacement, hard-contradiction, filename-proximity, and graph-safety gates
remain authoritative.

## Positive-control results after the fallback

All 11 matched sheet controls are now review candidates.

| Source | Candidate source | Evidence | Same-and-occluded | Mismatch captured | Outside text-union mismatch |
|---|---|---:|---:|---:|---:|
| `i4399007-00128.jpg` | residual full-page | 0.9579 | 0.9579 | 1.000 | 0.000 |
| `i4399011-00328.jpg` | residual full-page | 0.9641 | 0.9641 | 1.000 | 0.000 |
| `i4399021-00192.jpg` | residual full-page | 0.9600 | 0.9600 | 1.000 | 0.000 |
| `i4399023-00320.jpg` | residual full-page | 0.9609 | 0.9609 | 1.000 | 0.000 |
| `i4399024-00542.jpg` | dense text erasure | 0.9892 | 0.9837 | 0.968 | 0.032 |
| `i4399014-00153.jpg` | dense text erasure | 0.9701 | 0.9578 | 0.902 | 0.062 |
| `i4399017-00257.jpg` | residual full-page | 0.9577 | 0.9576 | 1.000 | 0.000 |
| `i4399023-00163.jpg` | residual full-page | 0.9663 | 0.9663 | 1.000 | 0.000 |
| `i4399024-00720.jpg` | dense text erasure | 0.9818 | 0.9705 | 0.954 | 0.044 |
| `i4399031-00117.jpg` | residual full-page | 0.9791 | 0.9790 | 1.000 | 0.000 |
| `i4399033-00134.jpg` | residual full-page | 0.9695 | 0.9695 | 1.000 | 0.000 |

Same-and-occluded range: **0.9576–0.9837**.

Ten controls pass the deterministic physical-occlusion decision. One
(`i4399014-00153`) remains review-only because its outside text-union mismatch is
0.062, narrowly above the ordinary automatic threshold of 0.06. That conservative
behavior is intentional.

## Negative-cohort results

Thirty-seven same-form/different-record pair attempts were evaluated across the
mounted families. Twenty-two registered successfully and therefore reached content
scoring.

After enabling the fallback:

- maximum same-and-occluded score among registered negatives: **0.0378**;
- review-candidate threshold: **0.08**;
- deterministic physical-occlusion accepts among negatives: **0**.

The largest negative score is `i4071659-00042` versus `i4071659-00229`. The fallback
finds dense text disagreement, but identity probability and dirty exterior evidence
keep it below review threshold and block automatic linking.

## Safety interpretation

The fallback improves candidate recall, not identity inference. A same-template pair
may produce a dense text rectangle, but it still fails when:

- record-specific mismatch remains distributed outside the rectangle;
- geometric registration support is weak;
- one block does not capture enough total mismatch;
- full-page material/text ratios match record replacement;
- a hard contradiction exists; or
- automatic-link sequence safeguards fail.

## Remaining work

The complete 422-image package must still be recomputed using current `master`; old
caches lack the localization fields introduced in PR #21. Selection-only threshold
tuning and untouched locked-audit reporting remain required. The final report must
separate candidate recall, direct automatic-edge recall, transitive group recall,
false automatic edges, contaminated components, and review volume.
