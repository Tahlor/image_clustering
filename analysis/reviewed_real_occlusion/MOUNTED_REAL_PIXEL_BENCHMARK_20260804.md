# Mounted real-pixel occlusion benchmark — 2026-08-04

This note records mechanism validation performed on private mounted Vermont scans
without committing private filenames, identities, images, or case-level result rows.
Those artifacts remain outside Git.

## Scope

- 21 same-template, different-record pairs from seven registered declaration scans;
- 28 material-sheet occlusions applied to those real scans;
- 14 clean same-document perturbations;
- 7 large-number overlays treated as a non-semantic control;
- 86 additional different-record pairs across compatible mounted layouts.

The broader 86-pair table was first scored with corrected pixel features, then the
final localization-contrast rule was replayed over its saved content metrics. This is
not the complete 422-image reviewed-package run.

## Final measured operating behavior

Primary 70-case cohort:

- material-sheet occlusion: 28/28 deterministic physical-occlusion decisions and
  28/28 recall-first review flags;
- same-template different record: 0/21 automatic links and 0/21 review flags;
- clean repeat: 14/14 retained as same-document, with 0 occlusion flags;
- large-number overlay: 0/7 physical-occlusion decisions and 0/7 occlusion flags.

Extended different-record stress:

- 0/86 automatic links after the localization-contrast gate;
- 0/86 occlusion review flags at the configured operating point;
- the sole pre-gate automatic error had zero inside-versus-outside localization
  contrast: its proposed block did not explain disagreement better than the page.

## Failure mechanisms corrected

1. Identical dark borders produced false SSIM residuals because the denominator was
   floored above its valid dark-region scale.
2. Whole-image exposure normalization was biased by large dark or light covers and
   manufactured exterior disagreement.
3. Near-background sheets with almost complete localized text erasure were rejected
   because the legacy synthetic occlusion logit expected stronger material contrast.
4. The extreme-change branch could accept a candidate whose text mismatch was no
   more localized than the exterior.

The current master already classifies candidate support covering at least 75% of its
own page as full-page, so this branch relies on that source-independent safeguard
rather than duplicating it.

The complete governed 422-image selection/locked-audit run and the separate 500-image
safety stress remain required before declaring final production performance.
