# Exact performance caching

The clustering pipeline supports two deterministic caches under `cache_dir`. Neither
cache changes registration, content analysis, probabilities, pair decisions, or graph
construction.

## Feature cache

The feature cache stores the exact resized grayscale working image, resize scale, SIFT
keypoint coordinates, and descriptors. A warm cache hit therefore does not reopen or
resize the original source image.

The default cache remains compressed to control disk use. Set
`feature_cache_compressed=false` for a larger but faster local-NVMe cache. The cache key
includes source path, size, modification time, working dimension, SIFT parameters, and
cache format, so changed sources or feature settings cannot reuse stale entries.

## Pair-comparison cache

Every completed `PairComparison` is checkpointed atomically under
`pair_comparisons/<prefix>/<hash>.json`. Interrupted runs can resume without repeating
finished registration and content scoring, and an identical warm rerun can replay all
pair results directly.

The key includes:

- both source identities and file metadata;
- the pair index gap and sequence;
- all configuration fields that can affect scoring;
- a hash of the clustering source modules that implement feature extraction,
  registration, content analysis, probability scoring, and pair decisions.

Changing scoring code or a scoring threshold automatically invalidates the relevant
cache. Operational settings such as worker count do not.

## Configuration

- `cache_features=true`: enable feature caching;
- `cache_working_images=true`: include the exact resized grayscale image;
- `feature_cache_compressed=true`: use the disk-efficient default format;
- `cache_pairs=true`: checkpoint complete pair comparisons.

For production, place the cache on local NVMe. It is safe to delete at any time; doing
so affects runtime only.

## Measured smoke benchmark

A 40-image real Vermont smoke run with 39 pair comparisons was measured on the same
host with eight workers and ECC disabled so the cache path itself was isolated:

- existing cache: 7.1 seconds cold and 3.6 seconds warm;
- exact working-image plus pair cache: 6.0 to 7.5 seconds cold and 0.11 to 0.12 seconds
  warm;
- cold and warm pair diagnostics were byte-for-byte identical after serialization.

The principal benefit is fast iteration and restartability. The first full run still
performs the same scoring work, while repeated identical runs can avoid source decode,
feature extraction, registration, and content analysis.
