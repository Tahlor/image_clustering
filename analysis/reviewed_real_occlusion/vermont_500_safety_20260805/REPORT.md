# Vermont 500-image independent safety run

## Provenance

- Dataset ZIP SHA-256: `f3639c7283a395c9f925e08da7ec0672abfdb46511ff53fe3052cf781b24bf9e`
- ZIP bytes: `400850419`
- ZIP members: `508`
- Manifest-hashed artifacts: `506`
- Images: **500/500**
- Crosswalk names: **500/500**
- JPEG decode, CRC, manifest, and crosswalk errors: **0**
- Detector commit used for the run: `11b50db0ee6339a232d643a3f757cdc9ad2086e1`
- Config fingerprint: `525987e7cb709990`

## Results

| Metric | Result |
|---|---:|
| Input images | 500 |
| Sequence families | 128 |
| Pair comparisons | 902 |
| Execution errors | 0 |
| Automatic edges | **0** |
| Review candidates | 115 |
| Hard contradictions | 361 |
| Singleton families | 49 |

The run found no automatic link in the independent population. The 42 pairs close enough in actual filename position to be operationally eligible for an edge were visually audited: 25 were different documents/pages, 11 appeared to be the same physical item or front/reverse state, and 6 were uncertain blank/reverse-card cases. None became an automatic edge.

## Runtime and exact replay

- Cold runtime: **237.905 seconds**.
- Warm exact-cache replay: **1.872 seconds**.
- `comparisons.jsonl`, `automatic_edges.jsonl`, and `review_candidates.jsonl` were byte-identical between cold and warm runs.
- Normalized summary equality passed.

## Interpretation

This run independently validates the deterministic automatic-link safety gate: no new automatic false link was found among 902 comparisons. It was executed before the later `v6-material-block-recall` review-only patch. That patch cannot create graph edges, so this zero-edge result remains valid evidence for automatic-link safety. Review-candidate volume and recall should be regenerated under the v6 fingerprint before issue #33 is treated as fully complete.
