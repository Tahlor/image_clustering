# Mounted multi-page candidate competition stress — 2026-08-04

## Purpose

This stress test targets multi-page scans where the ordinary grayscale residual finder
produces a weak candidate on a page even though a much stronger rectangular text-erasure
block is present. The previous dense text-channel logic ran only when no residual
candidate existed, so a weak residual could suppress the better physical explanation.

This is a mounted mechanism test, not the complete governed 422-image run.

## Failure

Source: `i4071660-00495(1).jpg`, an older two-page court spread.

A large bottom-page sheet was placed over the registered source. The ordinary selector
found one candidate per page, but the left-page residual captured only 14.2% of that
page's unmatched text. A dense rectangular text candidate was also available and
captured 88.5%.

Before candidate competition:

- same-and-occluded probability: **0.0303**;
- aggregate mismatch capture: **0.4742**;
- outside unmatched text-union fraction: **0.5749**;
- deterministic physical-occlusion decision: rejected.

## Implemented rule

PR #26 computes the dense text-erasure candidate even when a residual candidate
already exists. Residual evidence remains the default because it directly measures
material change. Dense text support replaces it only when it captures at least
**0.20 more** of the page's unmatched text.

The boundary is inclusive. A full-page residual cannot be displaced because it already
captures all page mismatch. No review probability, automatic-link, or graph threshold
changed.

## Recovered state

After selecting the better candidate separately on each page:

- selected candidates: dense text blocks on both pages;
- same-and-occluded probability: **0.7144**;
- aggregate mismatch capture: **0.9066**;
- outside unmatched text-union fraction: **0.2294**;
- outside mismatching-tile fraction: **0.3917**;
- occlusion evidence: **0.8646**;
- feature overlap: **0.3132**;
- deterministic physical-occlusion decision: accepted.

The opposite top-sheet state remained strongly accepted. The two mutually occluded
endpoint states did not register directly, which is expected; the unobscured base view
provides the legitimate state-graph bridge.

## Mounted safety cohorts with competition enabled

| Cohort | Registered / total | Maximum negative or minimum positive score | Review candidates | Deterministic accepts |
|---|---:|---:|---:|---:|
| real same-form negatives | 22 / 37 | max 0.0378 | 0 | 0 |
| noisy hard negatives | 19 / 20 | max 0.0439 | 0 | 0 |
| basic matched sheet controls | 11 / 11 | min 0.9576 | 11 | 10 |
| noisy/cropped sheet controls | 20 / 20 | min 0.4023 | 20 | 8 |
| older/faint/spread controls | 17 / 17 | min 0.4526 | 17 | conservative review/edge mix |

The competitive selector therefore fixes the observed multi-page miss without moving
any mounted hard negative into the review queue.

## Remaining work

The complete 422-image package still must be recomputed with current `master`, followed
by selection-only tuning and untouched locked-audit reporting. Final reporting must
separate candidate recall, direct automatic-edge recall, transitive group recall,
negative automatic edges, contaminated components, and review volume.
