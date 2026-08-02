# Adjacent Extraction Occlusion Linkage

This investigation links nearby Vermont scans that may show the **same physical document under different occlusions**. It is deliberately separate from the image clustering and crop-recovery implementation: CV proposes same-scene evidence, while this analysis asks whether the extracted people and fields are consistent with one view being a partial observation of another.

The analysis does not overwrite extraction records. It emits pair evidence, image-level flags, and multi-image groups that downstream review or delivery shaping can consume.

## Questions answered

1. Are two filename-adjacent images plausibly observations of the same physical document?
2. Is one extraction substantially contained in the other, with few contradictory fields?
3. Does an embedding or existing CV pair score support or contradict the link?
4. Which image is the likely occluded/partial view, and which is the better available companion?
5. Do three or more images form one document group with several different occlusion states?

A shared name alone is not enough. Adjacent pages can concern the same person while being different legal pages. Automatic links require extraction consistency plus either sufficient containment or strong embedding/CV support. Lower-scoring plausible pairs are retained in a review queue and are not used for transitive grouping.

## Inputs

`--extractions` accepts JSON or JSONL in either production shape:

- batch rows with `image_stem` and `parsed_response`;
- flattened rows with `source_filename` and `records`;
- canonical nested rows with `events[*].records[*]`.

Repeat `--extractions` for multiple model/prompt runs. Runs are merged per image: records are aligned by role and identity, normalized agreement is used as field consensus, and unmatched records are retained.

Optional evidence:

- `--embeddings`: `.npz` with `filenames`/`image_ids` plus `embeddings`, an `.npz` keyed directly by image ID, or JSONL with an `embedding` array;
- `--cv-evidence`: JSONL/CSV with `image_a`, `image_b`, and optional `same_scene_probability`, `occlusion_probability`, or `relation`; the production `first_image_id`, `second_image_id`, `same_document_probability`, and `occluded_given_same_probability` fields are also accepted;
- `--quality-metadata`: JSON/JSONL containing the production quality risks, especially `occlusion_risk`, `crop_risk`, `page_boundary_risk`, and `indexability_score`.

Images are compared only within the same filename series and within `max_gap`. The Vermont source filename is recovered from batch stems such as `0170_..._63129_2327225e_0014-00136`.

## Run

```bash
python analysis/adjacent_extraction_occlusion/adjacent_occlusion.py \
  --extractions /path/to/production_extractions.jsonl \
  --extractions /path/to/second_pass.jsonl \
  --embeddings /path/to/dinov3_embeddings.npz \
  --cv-evidence /path/to/pair_scores.csv \
  --quality-metadata /path/to/quality.jsonl \
  --config analysis/adjacent_extraction_occlusion/config/default.json \
  --output-dir /path/to/adjacent_occlusion_analysis
```

Minimal synthetic example:

```bash
python analysis/adjacent_extraction_occlusion/adjacent_occlusion.py \
  --extractions analysis/adjacent_extraction_occlusion/examples/synthetic_extractions.jsonl \
  --cv-evidence analysis/adjacent_extraction_occlusion/examples/synthetic_cv_evidence.jsonl \
  --output-dir /tmp/adjacent-occlusion-demo
```

## Outputs

- `adjacent_pair_scores.jsonl`: every tested adjacent pair, component signals, score, automatic/review/rejected tier, and likely view roles.
- `occlusion_groups.jsonl`: automatic-link connected components, canonical best view, all linked images, and whether multiple occlusion states are present.
- `image_flags.jsonl`: flags for the likely occluded image **and** its more-complete companion, plus all linked IDs and group IDs.
- `review_queue.jsonl`: plausible pairs below the automatic threshold. These never create transitive links.
- `review.html`: lightweight score-ordered inspection page.
- `summary.json`: counts and exact configuration.

Important image flags are:

- `possible_occlusion_event`;
- `possible_occluded_view`;
- `possible_more_complete_companion`;
- `linked_image_ids`;
- `automatic_linked_image_ids`;
- `occlusion_group_ids`;
- `canonical_for_group_ids`.

The “more complete companion” is not asserted to be perfectly unoccluded. It is the best available observation according to extraction coverage and quality evidence.

## Scoring and safeguards

The same-document score combines:

- record alignment by role, name, petition number, birth year, and event type;
- weighted field containment in both directions;
- contradictory-field rate;
- strong identity anchors such as exact petition number or full name;
- optional embedding cosine similarity;
- optional CV same-scene probability;
- a small adjacency prior.

Event date/place fields receive low weight so that an oath page is not linked to a petition merely because it names the same person. Conflicting document-specific fields are an explicit penalty. Review-tier edges remain visible but do not enter union-find grouping.

Occlusion-role assignment is separate from identity linkage. It uses extraction field/record asymmetry, production occlusion/crop risks, indexability, and optional CV occlusion probability. When the evidence does not clearly distinguish the views, the pair remains linked but role fields stay null.

## Calibration plan

The defaults are a starting operating point, not a claimed final classifier. Calibrate on grouped physical-document labels with these classes:

- same document, one view occluded;
- same document, near duplicate/no meaningful occlusion;
- same person, different page/document;
- different person, same form template;
- unrelated adjacent images.

Optimize occlusion-event recall first, while tracking false transitive groups separately. Report pair PR curves, group purity/recall, role-direction accuracy, and the incremental value of extraction, embedding, CV, and quality features. Keep a locked group-level audit partition after thresholds are selected.

## Tests

From the repository root:

```bash
pytest -q analysis/adjacent_extraction_occlusion/tests
```

The tests cover source-name recovery, subset extraction linkage, same-person/different-page rejection, multiple occlusion states, embedding rescue, cross-run consensus, role flags, and output artifacts.
