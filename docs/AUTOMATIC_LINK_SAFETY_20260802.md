# Automatic-link safety stress result

Date: 2026-08-02

## Motivation

The continuous same-document and same-document-occluded probabilities are intended for
recall-oriented candidate ranking. They are not sufficient evidence for an automatic
graph edge. In a 500-image real Vermont stress sample, the earlier deterministic gate
created twelve links; visual review found all twelve to be false.

## Guarded policy

The guarded implementation preserves the candidate probabilities while demoting weak or
operationally implausible deterministic matches to review:

- recognized archival filenames must share a sequence prefix and be within twelve
  capture positions;
- full-page matches supported only by ECC require review;
- dirty occlusion exteriors require stronger feature or alignment support;
- expensive ECC is attempted only with a bounded affine seed, at least fifty exact
  descriptor matches, or sufficient coarse phase-correlation evidence;
- ECC estimates motion on a bounded 384-pixel canvas and maps the transform back to
  full-resolution content coordinates;
- OpenCV ECC calls are serialized because concurrent calls caused pathological runtime,
  while ordinary BF/SIFT scoring remains parallel.

## Validation

The full local test suite passes with 85 tests. A fresh run of the 53-image reviewed
handoff passes all 26 evaluable clustering cases; `i4071662-00111` remains unavailable.
The canonical fixture now includes the previously unlabelled
`i4071657-00203` / `i4071657-00317` different-record pair.

The safety checks do not suppress the continuous review score and do not invent a hard
content contradiction. They affect only automatic graph eligibility.
