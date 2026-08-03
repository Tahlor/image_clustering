# Real same-template failure analysis and text-localization redesign

## What was wrong

The prior review probability was trained on synthetic examples and could interpret
broad text disagreement as strong occlusion evidence even when the deterministic
linker correctly rejected the pair. This is especially dangerous for Vermont
naturalization forms: the printed template registers extremely well while names,
dates, places, occupations, family rows, and signatures belong to a completely
different person.

The detector must answer **“does one physical block explain the disagreement?”**
before it answers the weaker question **“do these pages have similar structure?”**

## Personally inspected failures

### `i4071659-00042` versus `i4071659-00229`

Both images are declarations of intention using nearly the same printed form. The
record-specific text differs throughout the right page. One image also contains a
photograph where the other does not. The old candidate finder selected that genuine
rectangular photo difference, but it explained only **21.6%** of the unmatched ink.
Outside that rectangle, unmatched ink occupied **25.7%** of the text union and **70.2%**
of valid text tiles. There is no single occlusion explanation for the rest of the
page.

Before the redesign:

- `P(same document) = 0.2982`
- raw `P(occluded | same) = 1.0000`
- `P(same and occluded) = 0.2982`

After the localization gate:

- physical-occlusion evidence = `0.0185`
- gated `P(occluded | same) = 0.0185`
- `P(same and occluded) = 0.0055`
- review candidate = `false`

### `i4071659-01057` versus `i4071659-01126`

These are again the same declaration template populated with different people and
families. The mismatch is distributed across the page and no contiguous physical
candidate is found. Outside unmatched ink is **37.8%** of the text union and **70.3%**
of valid text tiles.

The raw synthetic model again returned `P(occluded | same) = 1.0000`. The new
evidence score is exactly zero because no physical block exists.

## Positive control

A synthetic registered sheet overlay that hides a large region of the same form
produces the opposite signature:

- candidate captures `100%` of unmatched ink;
- inside unmatched-ink ratio is `0.724`;
- outside unmatched-ink ratio is `0.000`;
- material median is `0.028`;
- gated `P(same and occluded) = 0.971`;
- deterministic physical-occlusion link remains accepted.

## Implemented redesign

1. **Shared text-channel threshold.** Both registered views now use one pooled Otsu
   threshold over a local-darkness plus gradient stroke response. Independent
   thresholds previously made exposure differences look like asymmetric ink.
2. **Noise-safe stroke cleanup.** A very narrow morphological bridge reconnects
   anti-aliased fragments of real print and handwriting. Only original mismatch
   pixels survive, so the bridge cannot invent a large block.
3. **Text localization metrics.** Every candidate measures:
   - unmatched-ink ratio inside the candidate;
   - fraction of all unmatched ink captured by the candidate;
   - inside-versus-outside localization contrast.
4. **Physical evidence gate.** The synthetic occlusion probability is multiplied by
   a rule-based evidence score requiring contiguous support, residual capture,
   text-mismatch capture, block replacement, and exterior agreement.
5. **Distributed replacement penalty.** Broad mismatch outside the candidate, or a
   candidate consisting only of thin changed text without sheet-level material,
   receives a 95% evidence penalty.
6. **Deterministic protection.** The automatic physical-occlusion branch now requires
   the same block-localization evidence; this is not only a review-ranking change.

## Remaining weaknesses to evaluate on the complete 422-image package

- Very light blank sheets may have weak material contrast; they must still hide enough
  text or form structure to produce high inside mismatch capture.
- A true occluder plus unrelated page-wide scan damage may fail the strict exterior
  gate. That is an acceptable review false negative, but it must not become a merge.
- A photograph, stamp, or seal can be a genuine compact material difference without
  being an occluding sheet. It is rejected when it explains only a minority of the
  page’s text mismatch, as in the inspected failure above.
- Full-page inserts have little exterior evidence. They remain conservative and rely
  on material change rather than template similarity.
- The complete 422-image run is still required to tune the evidence thresholds on the
  selection split and then report the untouched locked audit. The current regression
  establishes the mechanism and removes the observed false-positive pattern; it is
  not a substitute for that full run.

Use `diagnose_pair.py` to produce `metrics.json` and a four-panel visualization for
any future false positive without adding one-off scripts to the production CLI.
