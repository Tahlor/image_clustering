# Mounted occlusion noise stress — 2026-08-04

## Purpose

This cohort tests whether the review detector remains useful under capture noise while
rejecting the same printed form populated with a different record. The target is
review-candidate recall for physical occlusion; deterministic automatic linking is
allowed to remain more conservative.

This is a mounted mechanism stress test, not the complete governed 422-image run.

## Cohorts

### Positive controls

Five representative mounted scans received a matched large paper overlay, then one of
four capture perturbations:

- 1.5 degree rotation plus small translation;
- brightness increase;
- Gaussian blur;
- crop-like translation with a filled border.

Sources covered:

- modern Petition;
- Affidavit/Oath page;
- Declaration;
- dense older two-page court spread;
- extremely faint older record.

Total positive perturbations: 20.

### Hard negatives

Five visually confirmed same-template/different-record pairs received the same four
perturbations on the second image. The set includes:

- Washington declaration records;
- modern Declaration records;
- Petition records;
- Affidavit/Oath records.

Total negative perturbations: 20; 19 registered successfully.

## Failure found

Blurring `i4399023-00163` before comparing it with the different record
`i4399014-00153` raised same-and-occluded probability to **0.1135**, above the 0.08
review threshold. Automatic linking still rejected it.

The false candidate had:

- feature overlap: 0.0648;
- outside unmatched text-union fraction: 0.4458;
- outside mismatching-tile fraction: 0.4822;
- mismatch captured inside the candidate: 0.5611;
- content-only occlusion evidence: 0.6584.

The form registered, but document-specific identity support was weak and record text
continued to disagree broadly outside the candidate.

## Implemented gate

PR #25 applies a soft identity-support factor only when the exterior already meets the
existing dirty-exterior criteria.

For SIFT registration:

- zero weight begins at 40% of the existing strong feature-overlap threshold;
- full weight is reached at the existing strong threshold of 0.15.

For ECC fallback:

- the ramp begins at the ordinary minimum ECC correlation;
- full weight is reached at the existing dirty-exterior threshold of 0.55.

Clean-exterior candidates receive weight 1.0. Deterministic acceptance and graph
construction are unchanged.

## Final results

### Hard negatives

- 19 registered negative perturbations;
- maximum same-and-occluded probability before the gate: **0.1135**;
- maximum after the gate: **0.0439**;
- deterministic physical-occlusion accepts: **0**.

The exact blurred failure drops from **0.1135 to 0.0061**.

### Positive controls

- 20/20 positive perturbations remain above the 0.08 review threshold;
- final same-and-occluded range: **0.4023–0.9837**;
- 8/20 pass deterministic automatic physical-occlusion acceptance;
- the remaining 12 are intentionally review-only because registration or exterior
  cleanliness is insufficient for an automatic edge.

The lowest positive is a crop-shifted Affidavit/Oath control at 0.4023, still five
times the review threshold.

## Interpretation

The review detector and automatic linker now have deliberately different operating
points:

- review scoring retains noisy/cropped/blurred physical-sheet states;
- automatic linking requires substantially cleaner exterior and stronger identity;
- blurred same-template record replacement no longer enters the review queue merely
  because one dense mismatch region can be proposed.

## Remaining work

The current model must still be recomputed over the complete 422-image reviewed
package. Selection-only threshold tuning and untouched locked-audit reporting must
measure candidate recall, review volume, direct automatic edges, transitive grouping,
negative edges, and contaminated components separately.
