# Output contract

All records carry `schema_version: "1.0"`.

## Pair record

`adjacent_pair_scores.jsonl` has one row per compared pair:

- identifiers: `image_a`, `image_b`, filenames, sequence, and gap;
- `signals`: extraction containment, conflict rate, identity anchors, field masses, optional embedding/CV evidence, production risks, completeness, and asymmetry;
- `same_document_score`;
- `link_tier`: `automatic`, `review`, or `rejected`;
- `possible_same_document` and `possible_occlusion_event`;
- `likely_occluded_image_id` and `likely_more_complete_image_id`, nullable when direction is ambiguous;
- `role_confidence` and machine-readable `reasons`.

Only `automatic` edges form transitive groups.

## Group record

`occlusion_groups.jsonl` includes:

- stable content-derived `group_id`;
- ordered `image_ids`;
- `canonical_image_id` for the best available observation;
- all `possible_occluded_view_ids` and `possible_more_complete_companion_ids`;
- `multiple_occlusion_states`;
- minimum automatic-edge confidence and edge list.

A group may have several occluded views and more than one usable companion.

## Image record

`image_flags.jsonl` is the delivery-friendly output. It intentionally flags both sides of an event:

- the image that may be occluded/partial;
- the linked image that is probably more complete;
- ambiguous linked peers when direction cannot be established.

It retains candidate links and automatic links separately so downstream review does not accidentally treat a weak pair as established identity.
