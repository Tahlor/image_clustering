# Vermont 500-image safety bundle

This directory contains the committed independent safety evidence:

- `REPORT.md` — plain-English run summary and interpretation.
- `integrity_receipt.json` — ZIP, manifest, crosswalk, and JPEG integrity gate.
- `summary.json` — frozen detector run counts.
- `warm_replay_receipt.json` — cold/warm exact-cache equality and timing.
- `near_gap_visual_audit.csv` — visual audit of all 42 operationally near filename pairs.

The frozen run used detector commit `11b50db0ee6339a232d643a3f757cdc9ad2086e1` and found zero automatic edges in 902 comparisons. The later v6 material-block change is review-only and cannot create an edge. A v6 replay is still required to refresh review-candidate volume, but not to preserve the zero-edge safety finding.
