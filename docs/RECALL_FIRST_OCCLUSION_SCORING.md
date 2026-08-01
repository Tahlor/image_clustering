# Recall-first occlusion candidate scoring

Date: 2026-08-01

## Purpose

The production clustering decision remains conservative: a pair enters the same
physical-document graph only when the existing near-duplicate or physical-occlusion
rules accept it. This feature adds a separate, continuous score for finding plausible
same-document occlusion cases that the conservative gate may reject.

The score is intended for ranking, review queues, and later real-data calibration. It
must not create graph edges by itself.

## Probability contract

The runtime emits a hierarchical pair of probabilities:

- `same_document_probability = P(same physical document)`;
- `occluded_given_same_probability = P(partially occluded | same document)`.

The three coherent state probabilities are then:

- `same_clean_probability = P(same) * (1 - P(occluded | same))`;
- `same_occluded_probability = P(same) * P(occluded | same)`;
- `different_document_probability = 1 - P(same)`.

They sum to one by construction. The current model is a compact pair of logistic
functions fit on the versioned Vermont synthetic benchmark. It adds no runtime machine
learning dependency.

## Two operating decisions

`occlusion_candidate_flag` is recall-oriented. Its checked-in threshold is 0.08 on
`same_occluded_probability`. A flagged pair is a candidate for review or stronger
analysis; it is not necessarily a correct match.

`automatic_link_eligible` follows only the deterministic clustering result and the
hard-contradiction safeguard. A high probability can never override a rejected pair or
an ink contradiction.

This separation is required because the synthetic audit reached high occlusion recall,
but the synthetic-only probabilities overestimated several same-template/different-
record pairs in the original reviewed images.

## Registration fallback

SIFT/RANSAC remains the primary registration path. Heavy card or half-page occlusions
can remove enough local features that SIFT fails before content identity is evaluated.
After such a failure, the implementation may try ECC registration under strict
production assumptions:

- both captures have the same orientation;
- rotation is at most 5 degrees;
- translation is at most 18 percent of either image dimension;
- image aspect ratios remain compatible;
- ECC correlation is at least 0.30.

A loose SIFT similarity estimate initializes ECC when at least four descriptor matches
exist. Otherwise phase correlation supplies only a translation seed. Successful ECC
registration establishes coordinates; it does not establish identity. The same ink,
material, exterior-agreement, and hard-negative rules run afterward.

## Synthetic calibration result

The versioned Library dataset contains 3,250 rendered pairs and 32,500 deterministic
recipes, grouped by source form family. On the frozen synthetic audit, the recall-first
flag found 222 of 225 occluded positives (98.67 percent). Recall was evaluated
separately for index cards, strips, one-third sheets, half sheets, and larger stress
cases.

Those figures describe the synthetic benchmark, not a production probability guarantee.
A larger independent real reviewed set is required before interpreting the probability
values literally or enabling any probability-based automatic link.

## Required evaluation

Report these channels separately:

1. registration success by label and occlusion-size bucket;
2. candidate-flag recall and precision;
3. deterministic accepted-pair precision and recall;
4. hard-contradiction recall on same-template negatives;
5. contaminated graph components;
6. review coverage at candidate-score thresholds;
7. runtime, including ECC fallback frequency and cost.

The primary recall metric is detection of true same-document partial occlusions. The
primary safety metric remains contaminated components, not average pair accuracy.
