# Reviewed clustering and crop cases

`reviewed_cases.jsonl` is the canonical, append-friendly calibration dataset.
Each line is one independent reviewed case and can contain both clustering truth
and exact normalized crop targets. Private source images are intentionally not
committed; cases reference filenames that the local evaluation runner resolves.

## Add a non-cluster

```bash
image-label pair examples/evaluation/reviewed_cases.jsonl \
  image_a.j2k image_b.j2k --different \
  --notes "Same form, different handwriting"
```

## Add a true duplicate or occlusion cluster

```bash
image-label pair examples/evaluation/reviewed_cases.jsonl \
  image_a.j2k image_b.j2k --near-duplicate

image-label pair examples/evaluation/reviewed_cases.jsonl \
  image_a.j2k image_b.j2k --occlusion

image-label cluster examples/evaluation/reviewed_cases.jsonl \
  image_a.j2k image_b.j2k image_c.j2k \
  --notes "Three views of the same spread"
```

Every same-document case defaults to `expected_min_submissions: 1`. This is an
important invariant: near duplicates still need recognizer input even when no
change region exists.

## Add a reviewed crop

Coordinates are normalized by default:

```bash
image-label crop examples/evaluation/reviewed_cases.jsonl \
  CASE_ID image_a.j2k 0.02 0.015 0.50 0.985 \
  --kind base_page --completeness complete --side left
```

Use `--pixels` for source-pixel coordinates.

## Validate and inspect

```bash
image-label validate examples/evaluation/reviewed_cases.jsonl
image-label list examples/evaluation/reviewed_cases.jsonl
```

Agents should treat these labels as regression/evaluation truth, never as
filename-specific production exceptions. HTML evaluation should show expected
and actual cluster membership, expected and actual submission counts, crop-box
IoU where target boxes exist, and a prominent failure when a positive case
produces zero recognizer submissions.
