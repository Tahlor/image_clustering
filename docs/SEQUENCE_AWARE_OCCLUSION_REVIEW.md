# Sequence-aware occlusion review queue

Date: 2026-08-01

## Why pairwise probability is not enough

The synthetic benchmark provides a useful continuous score for finding likely
same-document occlusion pairs. On the original reviewed images, however, several
same-template/different-record pairs also received high scores. Conversely, some true
occlusion pairs looked contradictory when examined directly because a third capture
provided the clean bridge between them.

The safe response is not to relax the hard-negative rules. Instead, use ordered-sequence
context to prioritize review while leaving graph construction unchanged.

## Review tiers

The review report ranks pairs lexicographically before sorting by probability:

- **Tier 0 — accepted audit:** deterministically accepted pairs, included only when
  `--include_accepted` is requested.
- **Tier 1 — common-neighbor bridge:** a rejected candidate whose endpoints both have a
  conservative accepted edge to the same intermediate image. This is the strongest
  sequence-context evidence and is reviewed first.
- **Tier 2 — transitive component:** a rejected candidate whose endpoints are already in
  the same conservative component, but no single common accepted neighbor is present.
- **Tier 3 — unsupported candidate:** a high-scoring rejected pair without conservative
  graph support. These may be genuine occlusions, but the synthetic score is the main
  positive evidence.
- **Tier 4 — hard contradiction:** a candidate with distributed document-specific ink or
  other hard-negative evidence. These are isolated for expert review and never promoted
  to links by sequence context.

A tier is a review priority, not a probability and not a clustering decision.

## Usage

After clustering has produced `clustering.json`:

```bash
image-occlusion-review \
  --clustering_json /path/to/clustering.json \
  --output_dir /path/to/occlusion-review
```

The command writes:

- `occlusion_candidates.csv`;
- `occlusion_candidates.jsonl`;
- `occlusion_candidate_summary.json`.

Use `--include_accepted` to include deterministic positives for audit. Use
`--include_unflagged` for threshold analysis over every candidate pair.

## Safety invariant

The report is read-only with respect to the clustering result. Common-neighbor evidence,
same-component evidence, and candidate probabilities do not create or alter graph edges.
The existing deterministic pair decision and hard-contradiction checks remain the only
inputs to conservative cluster construction.

## Next calibration step

Review the highest-priority real candidates and append decisions to the canonical
reviewed-case fixture. Refit or calibrate probabilities only on a grouped development
partition, then measure candidate recall, candidate precision, accepted hard negatives,
and contaminated components on a new locked real holdout.
