# Clustering calibration: large occlusions versus different filled records

Date: 2026-07-26  
Scope: pairwise document clustering only. Crop recovery was intentionally left unchanged.

## Non-negotiable problem definition

A matching printed form is not sufficient evidence that two images contain the same
physical record. Two different people can have nearly identical pages where only the
handwritten or typewritten values differ. Those pairs are hard negatives and must not
share a cluster, directly or through a transitive bridge.

A true occlusion has a different geometry:

- one or two large, contiguous polygonal regions;
- usually rectangular, but mild skew or warping is expected;
- normally at least about one third of a page, and often one half or more;
- near-pixel-perfect form and ink agreement outside the polygon;
- dense residual/material change inside the polygon.

Scattered residual or ink disagreement across the page is not an occlusion. A large
bounding box around sparse handwriting changes is not an occlusion either.

The implementation uses lower support thresholds than one third because the residual
support is the visibly changed subset of the physical polygon. Blank or similarly toned
parts of an overlaid sheet may contribute no residual even though they are inside the
physical occluder.

## July 2026 reviewed handoff

The calibration archive contained 26 reviewed cases and 53 source images. It yielded 27
available direct pair comparisons: 16 different-document negatives and 11 positives
(10 occlusion combinations and one near duplicate). One reviewed case referenced a
source image that was not present in the archive and was not scored.

This set was used for calibration, not as an independent holdout. Future work must append
new reviewed cases and report them separately before retuning these thresholds.

## Baseline failure

The previous content path used a 10 by 14 residual grid, a square 3 by 3 morphological
close, and then replaced each connected component with every tile in its padded bounding
rectangle. Distributed handwriting could therefore become a page-sized rectangle. The
scorer then measured the exterior after hiding most of the disagreement inside that
rectangle.

Direct reviewed-pair result from the supplied run:

|                    | Predicted different | Predicted same |
|--------------------|--------------------:|---------------:|
| Different document | 7                   | 9              |
| Same document      | 1                   | 10             |

The nine false merges were the dominant error and are much more damaging than a split.

## Approaches evaluated

Registration was not the bottleneck. At a 900-pixel working dimension, every available
reviewed pair registered under all tested variants:

| Features / transform | Registered | Mean match + fit per pair | Median reprojection error |
|----------------------|-----------:|--------------------------:|--------------------------:|
| SIFT + affine RANSAC | 27 / 27    | 0.0296 s                  | 0.279 px                  |
| SIFT + homography RANSAC | 27 / 27 | 0.0270 s                 | 0.250 px                  |
| SIFT + homography MAGSAC | 27 / 27 | 0.0288 s                 | 0.267 px                  |
| AKAZE + affine RANSAC | 27 / 27   | 0.0305 s                  | 0.305 px                  |
| ORB + affine RANSAC  | 27 / 27    | 0.0186 s                  | 0.741 px                  |

SIFT + affine was retained. Homography did not recover any missed cases, while an
unnecessary extra degree of freedom can hide content mismatch. ORB was faster but had
substantially larger reprojection error.

Content experiments included adaptive binarized ink maps, Scharr gradients,
low-frequency material difference, seam/boundary measurements, exhaustive coarse
rectangle search, and several residual-grid resolutions. The best accuracy/runtime trade
was the simpler support-aware grid described below.

## Implemented scorer

1. Register with the existing SIFT + affine RANSAC path.
2. Build tolerant, locally normalized ink mismatch and residual maps.
3. Aggregate on an 18 by 24 grid.
4. Close only with a cross-shaped 3 by 3 kernel. This joins narrow seams without joining
   diagonal or widely distributed handwriting.
5. Preserve the actual connected component support. Keep its padded bounding rectangle
   only for geometry and diagnostics; never use the rectangle to erase exterior errors.
6. Permit a full-page candidate only when broad residual change is accompanied by broad
   low-frequency material change. Page-wide ink mismatch alone is a different filled
   record signal.
7. Accept an occlusion only when connected support is large/dense, captures most of the
   residual, has material evidence or exceptionally strong change, and leaves a clean
   exterior.
8. Mark distributed text replacement or exterior residual as a hard contradiction so it
   blocks a transitive graph merge.

A geometric/material route handles a large contiguous rectangle or skewed polygon even
when its global changed-pixel fraction is modest: it requires at least 30% page support,
75% residual capture, material median at least 0.020, and exterior unmatched-ink union at
most 0.015.

## Calibrated result

With the checked-in default configuration:

|                    | Predicted different | Predicted same |
|--------------------|--------------------:|---------------:|
| Different document | 16                  | 0              |
| Same document      | 0                   | 11             |

The exact production content/scoring implementation took 22.4 seconds for the 27 pairs,
or 0.83 seconds per pair, using cached 900-pixel working images and the supplied
registration transforms. SIFT matching plus transform fitting took about 0.03 seconds per
pair once descriptors were available. JPEG2000 decode and first-time feature extraction
are separate, amortized costs and remain hardware dependent; this is not an end-to-end
latency guarantee.

Synthetic regressions additionally require:

- rejection and hard-contradiction status for the same form populated with different
  names, dates, occupations, places, and signatures;
- acceptance of a contiguous rectangular overlay covering roughly one third of a page;
- acceptance of a similarly large skewed quadrilateral overlay;
- preservation of a clean two-page/multi-occlusion bridge case.

## Calibration discipline

Do not tune against aggregate cluster count. Use reviewed pair labels and report at least:
false merges, false splits, near-duplicate accuracy, occlusion accuracy, hard-
contradiction accuracy, and pair runtime. False merges are the primary failure metric.

When new cases are added:

1. freeze this configuration;
2. score the new cases as a holdout;
3. inspect every false merge first;
4. only then retune, reporting old-calibration and new-holdout results separately;
5. keep registration and content-classification changes separable so failures remain
   diagnosable.
