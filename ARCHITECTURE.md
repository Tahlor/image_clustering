# Architecture

## Current ownership: `image_clustering.clustering`

The clustering submodule answers one question:

> Which ordered captures are sufficiently supported as views of the same physical document scene?

Its output is `ClusteringResult`:

- `images`: stable IDs, absolute paths, sequence IDs, and sequence order;
- `clusters`: complete physical components, without downstream batch splitting;
- `comparisons`: all nearby candidate pairs, including rejected pairs;
- accepted `PairComparison.transform` values: source-pixel transforms from the second image into the first.

## Future ownership: unique crops

A separate module may be added under a sibling namespace such as `image_clustering.unique_crops`. It should consume `ClusteringResult` rather than reimplementing cluster membership. Its responsibilities may include:

- composing direct pair transforms across a cluster;
- selecting a reference image;
- detecting regions visible in only a subset of views;
- rejecting crops that remain partially occluded;
- emitting each unique physical document region once;
- selecting bounded image subsets for a multimodal model.

The unique-crop module may decide that a cluster contains several independently extractable regions. It must not merge separate `ImageCluster` objects based on form-template similarity.
