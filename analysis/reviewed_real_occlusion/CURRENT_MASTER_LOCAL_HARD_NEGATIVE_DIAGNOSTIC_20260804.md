# Current-master local hard-negative safety diagnostic — 2026-08-04

This is a diagnostic over the **35 reviewed originals currently materialized locally**. It is not the governed 422-image evaluation and contains no official positive-recall estimate.

## Pair construction

Pairs were restricted to the same broad document family and orientation, then required to have disjoint visible primary-person names in the canonical transcription. This creates a deliberately difficult same-template/different-record safety set.

## Results

- Candidate pairs: **77**
- Registered pairs reaching content scoring: **22** (28.6%)
- Deterministic same-document links: **0**
- Review candidates at `P(same_occluded) >= 0.08`: **0**
- Maximum `P(same_occluded)`: **0.068613**
- Margin below review threshold: **0.011387**

The two nearest-threshold pairs were visually inspected and are clearly different people/records:

1. `i4399011-00328` vs `i4399023-00320`: **0.068613**
2. `i4399014-00153` vs `i4399023-00163`: **0.050948**

## Interpretation

Current `master` cleanly rejects every locally available same-family hard negative. The result validates false-positive safety on this materialized subset only. It must not be substituted for the official 183 positive-pair / 83 negative-pair evaluation defined by the 422-row package.
