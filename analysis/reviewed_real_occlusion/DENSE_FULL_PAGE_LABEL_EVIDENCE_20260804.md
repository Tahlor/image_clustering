# Dense near-total candidates must use full-page semantics

## Failure

ECC can align the shared printed template of two different filled records after SIFT
rejects them. Dense text-erasure candidate generation may then cover essentially the
entire page. Those candidates were marked `full_page=0` because only the older
material shortcut emitted the full-page flag.

That source-dependent label bypassed the full-page record-replacement veto. In the
mounted ECC hard-negative cohort, false review scores ranged approximately 0.14–0.37.

## Correction

A candidate is now full-page when either:

- its candidate generator explicitly marks it full-page; or
- its actual connected support covers at least 75% of its own page's valid tiles.

The calculation is page-local on two-page spreads. It uses connected support rather
than bounding-box area, so a sparse large box does not become full-page.

## Measured effect

Applying the source-independent label activates the existing material/text
record-replacement rule. Mounted ECC same-template false review scores drop to
approximately 0.007–0.019. Genuine heavy-occlusion states remain eligible when they
show stronger inside text replacement or stronger physical material evidence.

No ECC launch, correlation, review-threshold, automatic-link, or graph threshold is
changed.
