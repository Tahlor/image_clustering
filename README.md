# image_clustering

A Python package for conservatively grouping ordered document images that show the **same physical document scene** under changing occlusions.

It is deliberately not a form-template clusterer. Two filled copies of the same printed form remain separate even when their layouts are almost identical.

The `image_clustering.clustering` submodule owns cluster membership and pairwise registration. Unique-crop discovery, model batching, and extraction belong in separate downstream modules.
