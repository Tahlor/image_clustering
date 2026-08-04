# Accepted-group visual evidence sidecar

The visual-review sidecar contains exactly one row for each of the 134 accepted
`original_cluster_id` values derived from `assignments.csv`. It never changes the
accepted/rejected same-document truth.

## Required relationship categories

`visual_relationship_category` must resolve to one of:

- `identical_or_near_identical`
- `material_physical_occlusion`
- `same_document_state_or_content_change`
- `visual_only_overlay`
- `mixed_or_multi_state`
- `uncertain_or_other` while work is in progress only

Large foreground numbers, stamps, seals, cards, and labels are represented by
`visual_only_overlay` and a non-`none` `visual_overlay_category`. They must use
`material_occlusion_metric_included=false` unless the original physical source
content is meaningfully hidden by a real material relationship independently of
the visual overlay. A non-material row cannot use the `same_occluded` subtype.

## Required fields

Each row records:

- `original_cluster_id`
- `member_image_ids_json`, exactly matching the authoritative accepted group
- `occlusion_subtype`
- `visual_relationship_category`
- `visual_overlay_category`
- `material_occlusion_metric_included`
- `affected_image_id` and `affected_image_ids_json`
- `occluded_image_id` and `occluded_image_ids_json`
- `better_view_image_id` and `better_view_image_ids_json`
- `meaningful_hidden_content_risk`
- `occlusion_size_category`
- `registration_difficulty`
- `evidence`
- `uncertainty_notes`
- `annotator_method`

The singular image fields identify the primary example for convenient reporting.
The JSON list fields enumerate every affected, occluded, and better-view image in
a multi-image accepted group, in authoritative member order.

A material-occlusion row must identify non-empty affected, occluded, and
better-view image lists, use a non-`none` hidden-content risk and size, and use a
compatible subtype and relationship category. Occluded images must be a subset of
affected images and cannot also be better views. Visual-only overlays cannot enter
the material-occlusion metric.

## Preparation and preservation

`prepare_dataset` always writes a fresh
`accepted_group_occlusion_subtypes_template.csv`. It creates
`accepted_group_occlusion_subtypes.csv` only when that working sidecar does not
already exist. Rerunning preparation therefore refreshes the authority-derived
template without overwriting completed human evidence.

## Validation

Validate the completed file before calibration:

```bash
image-reviewed-eval validate-subtypes \
  --prepared-dir /path/to/Results/reviewed_real_v1/prepared \
  --subtypes /path/to/completed_accepted_group_occlusion_subtypes.csv
```

Validation fails closed for duplicate rows, missing or extra accepted groups,
member-image disagreement, unresolved categories, nonmember image references,
contradictory overlay/material fields, incomplete multi-image lists, or missing
evidence.

The conditional probability calibrator uses
`material_occlusion_metric_included` as the truth for
`P(meaningfully occluded | same physical document)`. It does not treat every
`same_occluded`-like visual state, large number, stamp, or card as meaningful
hidden-content occlusion.
