# Governed reviewed-real 422-image result

- Git commit containing the safety gate and regression test: `11b50db0ee6339a232d643a3f757cdc9ad2086e1`
- Config fingerprint used for the governed run: `525987e7cb709990`
- Population: **422 images / 200 reviewed groups / 183 positive pairs / 83 reviewed-negative pairs**.
- Integrity: all 422 JPEGs decode; `assignments.csv` and `assignments.jsonl` agree semantically; package/source SHA reconciliation passes.

## Automatic-link safety

| Metric | Baseline | Safety-gated |
|---|---:|---:|
| Reviewed-negative automatic edges | 9 | **0** |
| Rejected groups kept separated | 59/66 | **66/66** |
| Positive direct automatic edges | 89 | 60 |
| Positive review candidates | 130 | **130** |
| Accepted groups fully connected | 62/134 | 38/134 |

The nine baseline false links were all deterministic `physical_occlusion` matches on same-template/different-filled-record pairs. The fix requires strong document-specific feature overlap (`>= 0.35`) for a physical-occlusion graph edge. Weaker matches remain review candidates. This removes every reviewed-negative edge without reducing the positive candidate count.

## Visual evidence sidecar

- Completed and validated rows: **134**.
- Material physical occlusion metric groups: **53**.
- Relationship counts: 67 identical/near-identical; 49 material physical occlusion; 13 visual-only overlay; 4 mixed/multi-state; 1 same-document state/content change.
- The 13 large foreground number overlays are explicitly visual-only and excluded from the material-occlusion metric.

## Runtime and replay

- Cold run: **350.28 seconds** for all 200 groups and 251 comparisons.
- Warm exact-cache replay: **2.90 seconds**.
- Comparisons, group results, positive/negative pair evaluations, and errors are byte-identical between cold and warm runs; normalized metrics are equal.

## Selection-only calibration

The isotonic artifact was fit from the selection split only. It was not fit from locked-audit labels and never changes graph edges.

It is **not recommended for promotion**. Its frozen conditional candidate threshold achieved 100% material-occlusion recall on selection but only 66.7% on the already-inspected audit. The existing recall-first candidate flag retained 93.3% there. The current candidate ranking therefore remains unchanged.

The original locked split is labeled **inspected_audit**, not represented as untouched, because aggregate outcomes were observed while developing the automatic-link safety fix.

## Recommendation

Promote the deterministic automatic-link safety gate and retain the existing review-candidate ranking. Do not declare final production promotion until the independent 500-image Drive stress population has been run and every automatic edge has been inspected.
