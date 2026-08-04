# Reviewed 422-image local materialization

The authoritative File Library package contains 422 reviewed image rows:

- 282 accepted images in 134 approved clusters;
- 140 rejected images from 66 dissolved clusters.

The image files are exposed to the execution runtime through transient per-user File
Library cache paths after the corresponding image card is opened. To prevent later
retrieval instability, each materialized JPEG is copied into
`/mnt/data/reviewed_422_local/images_by_basename` with a normalized basename and
SHA-256 ledger. Parenthetical upload-copy suffixes such as `(1)` are removed only from
the local filename; bytes are not modified.

The local reconciliation is complete only when all 422 manifest rows have exactly one
matching JPEG and no differing-byte basename collision. Until then, mounted subset
benchmarks must be labeled as partial mechanism stress tests rather than the governed
reviewed-real evaluation.

The authoritative manifest is `assignments.csv` from
`/Vermont Naturalization/Images/Occlusion Review/MY_REVIEWED_GT_DATASET` and includes
`image_id`, review decision, cluster status, accepted/dissolved assignment, original
cluster identity, and package-relative path.

## Materialization command

After opening another batch of File Library image cards, rerun:

```bash
python analysis/reviewed_real_occlusion/materialize_library_cache.py \
  --assignments-csv /path/to/MY_REVIEWED_GT_DATASET/assignments.csv \
  --scan-root /mnt/data \
  --scan-root /mnt/data/user-CURRENT_USER_CACHE \
  --output-root /mnt/data/reviewed_422_local
```

The command:

- reads only manifest-listed JPEG basenames;
- normalizes upload-copy suffixes such as `(1)` without changing bytes;
- ignores generated diagnostics and unrelated images;
- permits duplicate cache paths only when their SHA-256 values match;
- fails on differing-byte basename collisions;
- copies through a temporary file and atomic rename;
- rewrites the SHA ledger, missing-basename list, and JSON summary.

Use `--require-complete` for the governed run. It returns nonzero until all manifest
rows are present. Do not start selection calibration or the locked audit until the
summary reports `expected=422`, `materialized=422`, `missing=0`, and `collisions=0`.
