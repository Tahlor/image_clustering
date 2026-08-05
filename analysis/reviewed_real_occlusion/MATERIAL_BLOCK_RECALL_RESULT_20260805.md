# Material-block review recall improvement

## Committed implementation

- Lower the review-candidate threshold from `0.08` to `0.04`.
- Add a strong coherent-material-block override for long paper strips whose geometry makes the synthetic occlusion logit brittle.
- Add a clean-exterior, low-text material-state override for physical bands covering mostly blank form rows.
- Keep both overrides strictly review-only. They do not create graph edges or bypass hard contradictions.
- Probability model version: `vermont-synthetic-logit-v6-material-block-recall`.

## Reviewed-real effect

Retrospective evaluation on the completed 134-group visual sidecar improved material-occlusion group recall from **49/53 (92.5%)** to **53/53 (100%)**.

| Split | Material groups recovered |
|---|---:|
| Development | 22/22 |
| Selection | 16/16 |
| Inspected audit | 15/15 |

No rule-based reviewed-negative candidate was added by the two material overrides. The deterministic safety gate remains unchanged at **0 reviewed-negative automatic edges** and **0 contaminated rejected groups**.

## Validation

- Five focused material-block rule checks passed.
- Focused repository regressions cover threshold configuration, coherent-strip recovery, dirty-exterior rejection, SIFT-only low-text recovery, and the rule's review-only behavior.
- The independent 500-image run remains valid evidence for automatic-link safety because the v6 patch cannot create graph edges. Its review-candidate count should be regenerated under the v6 fingerprint before treating the review-volume report as final.
