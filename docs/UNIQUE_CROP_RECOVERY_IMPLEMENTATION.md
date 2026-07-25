# Unique Document Crop Recovery: Implementation Strategy

## Objective

Given a trusted `ClusteringResult`, identify every **unique data-bearing physical document or page state** visible anywhere in the cluster and submit each one once, using its best available observation.

This is downstream of clustering. The cropper may refine geometry inside a cluster, but it must not silently merge clusters, split clusters, or reinterpret same-template similarity as physical identity.

A cluster can contain:

- literal duplicate captures;
- one base page observed with different occluders;
- a base page plus one or more foreground sheets or inserts;
- a two-page spread with one page partly covered;
- reverse-facing, translucent, or blank paper that should not be sent to recognition;
- a page that is never fully visible and therefore requires `partial_best_available` output.

The cropper should recover logical document states, not simply emit every pairwise difference rectangle.

## Core invariants

1. Every content-bearing singleton or accepted cluster produces at least one recognizer submission.
2. A literal or near-duplicate cluster produces the best complete observation once, not zero crops.
3. A unique foreground sheet is emitted once even if it appears in several captures.
4. A persistent base page is emitted once even when most pairwise changes concern overlays.
5. Blank, reverse-facing, translucent, or purely obstructing sheets are retained as geometric evidence but suppressed from recognition unless genuinely uncertain.
6. If no complete observation exists, emit the best partial observation with an explicit completeness label and `review_required=true`.
7. Cluster size is not capped at the model's image limit. Model batching happens after logical document states are identified.
8. Different handwriting is never deduplicated merely because two crops share a form template.

## Evaluation protocol

Evaluate the cropper in two modes:

### Oracle-cluster evaluation

Feed ground-truth physical clusters to the cropper. This isolates page-state discovery, crop geometry, deduplication, and best-view selection from clustering errors.

### End-to-end evaluation

Feed predicted clusters. This measures the actual production system. A contaminated cluster should fail the end-to-end case even if the cropper emits visually plausible boxes.

Once crop parameters are selected using a curated set, that set is development data. Preserve a grouped locked audit partition exactly as described in `CLUSTERING_EXPERIMENT_STRATEGY.md`.

Crop truth should describe logical targets rather than only one hand-picked source box. For each expected target, store when practical:

- stable logical target ID;
- target kind and side/page assignment;
- acceptable source observations;
- expected normalized box or polygon in each labeled observation;
- completeness requirement;
- whether a partial observation is acceptable;
- whether the target must be suppressed from recognition;
- minimum and exact expected submission counts where known.

This avoids penalizing the system for selecting an equally good alternate view.

## Public package boundary

Keep the implementation under `image_clustering.cropping` with explicit stage interfaces. Recommended high-level API:

```python
result = recover_unique_crops(
    clustering_result=clustering_result,
    config=crop_config,
    output_dir=output_dir,
)
```

The output should contain logical targets and observations separately:

- `DocumentState`: one unique physical page, insert, or meaningful foreground sheet;
- `CropObservation`: one image/bbox/polygon observation of a state;
- `RecognizerSubmission`: the selected canonical crop, optionally with complementary supporting views;
- `CropRecoveryResult`: all states, candidates, suppressed regions, diagnostics, and submissions.

Do not use import-time monkey-patching to replace pipeline functions. Each stage should be a normal function or strategy object passed explicitly through the pipeline, independently testable and serializable.

## Recommended pipeline

### 1. Build a reliable transform graph

Use accepted pair registrations from `ClusteringResult` as the initial graph. Choose one or more reference images by:

- registration confidence and transform connectivity;
- visible page area;
- sharpness and exposure;
- low occlusion burden;
- centrality in the accepted-edge graph.

Compose transforms to map each image into a reference coordinate system. Where a direct transform exists, prefer it over a long composition path. Check cycle consistency on available triangles and mark unstable regions or views.

The cropper may perform page-local refinement after a preliminary page or change mask is known. Reuse clustering registration utilities, but keep the distinction clear:

- cluster registration establishes the relationship between captures;
- crop registration optimizes local alignment for state recovery;
- crop registration failure does not revoke cluster membership.

### 2. Decompose each capture into physical page regions

Before change analysis, identify coarse page/spread geometry:

- single page versus two-page spread;
- gutter or central fold;
- outer page boundaries;
- large foreground sheet candidates;
- valid visible area after warping.

Use multiple inexpensive cues:

- projection profiles and central low-ink gutters;
- long line segments and rectangular/convex contours;
- paper-to-background intensity or color changes;
- persistent boundaries across aligned views;
- shadow and edge evidence;
- known image borders only as weak priors.

Do not assume every page boundary is complete. Historical capture backgrounds, folded paper, and overlapping sheets frequently erase one or more edges. Represent uncertain page geometry explicitly.

### 3. Create a joint aligned stack

For each coarse page region, warp all relevant views into a common page coordinate frame and retain:

- grayscale/color observations;
- validity and visibility masks;
- soft ink maps;
- material residual channels;
- gradient and boundary channels;
- per-view quality measurements.

Compute robust stack summaries such as median appearance, median ink support, and per-pixel visibility frequency. Pairwise differences remain useful diagnostics, but unique-state discovery should use all cluster views jointly. A region visible in views 1 and 3 but covered in view 2 should not be mistaken for two unrelated pairwise events.

### 4. Discover state-changing regions jointly

Build a residual tensor for every view relative to robust stack references and to directly connected neighboring views. Threshold with robust local baselines, then form candidate components in reference coordinates.

For each spatial component, create a **view-state signature**: which views expose the same appearance, which expose a different sheet, and which are invalid or occluded. Merge components when they:

- overlap strongly in reference coordinates;
- have compatible view-state signatures;
- share a plausible physical sheet boundary;
- are separated only by text-line gaps or small registration artifacts.

Keep components separate when they represent different foreground sheets, different pages of a spread, or unrelated small annotations.

The result should be a small set of candidate physical states rather than one rectangle for every accepted pair.

### 5. Distinguish material regions from text-only residuals

Use the same material-versus-ink decomposition as clustering.

A foreground sheet or page turn should usually have some combination of:

- coherent low-frequency material change;
- a physical boundary, fold, or shadow edge;
- a large contiguous visibility region;
- consistent interior texture or tone;
- a stable view-state signature across more than one comparison.

Different handwriting without a physical sheet change produces thin, high-frequency unmatched ink rather than a solid material region. Such differences may help separate two document states, but they should not define a crop boundary by themselves.

### 6. Recover the physical sheet boundary

The raw residual component normally covers active text and shadows, not the complete paper. Starting from the residual support, expand toward a plausible physical boundary using:

- aligned gradient peaks and long line segments;
- paper/background or paper/paper color and texture discontinuity;
- row/column projection changes;
- contour and convexity support;
- stable edges observed in other views;
- expected continuation to the nearest page or gutter boundary;
- moderate aspect-ratio and area priors, never fixed form-specific dimensions.

Score several candidate rectangles or polygons. The score should reward:

- inclusion of residual and ink support;
- boundary evidence along the proposed perimeter;
- limited unexplained material outside the candidate;
- physical plausibility;
- consistency across views.

Avoid the current failure mode where a broad overlay is cropped only to its central text band. Preserve the polygon or mask internally even if the recognizer ultimately receives an axis-aligned padded rectangle.

### 7. Identify logical document states

At minimum classify candidates as:

- `base_page`;
- `foreground_sheet` or `overlay`;
- `alternate_page_state`;
- `partial_best_available`;
- `blank_occluder`;
- `reverse_or_translucent`;
- `uncertain`.

A persistent base page is inferred from page geometry and repeated visibility, not merely from the complement of an overlay box. A foreground sheet should be represented independently of the base page it covers.

Use geometry and appearance first. OCR or an MLLM should not be required to discover states. Optional frozen visual features may assist ambiguous local matching, but semantic similarity must not merge two filled copies of the same form.

### 8. Determine whether a state is recognition-worthy

Recognition-worthy evidence includes substantial front-facing printed or handwritten content. Suppression evidence includes:

- near-empty ink support;
- only mirrored or weak translucent show-through;
- blank separator or cover paper;
- a region whose only purpose is to occlude another page;
- severe incompleteness when a substantially better observation exists.

Use conservative reason codes. When frontness or content is ambiguous, retain the candidate with `review_required=true` rather than silently discarding potentially unique data.

### 9. Strictly deduplicate observations of the same state

Within each candidate state, locally register crop observations and apply a strict near-duplicate test based on:

- soft ink agreement with spatial tolerance;
- local intensity/gradient agreement;
- physical boundary agreement;
- document-specific feature agreement;
- absence of distributed unmatched handwriting.

Global form-template similarity is not sufficient. The deduplication rule should be at least as strict as the main clustering rule because crop-level false deduplication can erase a unique record.

A frozen local matcher or patch representation may be used as auxiliary evidence after geometric alignment. A frozen global embedding should never be the sole deduplication signal.

### 10. Select the best observation or complementary view set

Score each observation by:

- visible fraction of the physical state;
- boundary completeness and crop margin;
- front-facing confidence;
- sharpness and local contrast;
- exposure and compression quality;
- low perspective distortion;
- low residual occlusion;
- amount of data-bearing ink visible.

For a fully visible state, emit the highest-scoring observation once. If no view is complete, solve a small weighted set-cover problem to select the fewest complementary observations that maximize visible content. Mark the canonical observation `partial_best_available`; attach up to the downstream image limit as `supporting_observations`.

This preserves the original multi-image extraction idea without confusing model batching with cluster construction.

### 11. Explicit fallbacks

- **Singleton:** emit the best detected page or full valid page region.
- **Near-duplicate cluster with no change component:** emit the best complete page once.
- **Positive cluster with only uncertain crop geometry:** emit a padded whole-page or page-side fallback and require review.
- **No complete view:** emit the best partial crop and optional complementary observations.
- **Blank/non-data cluster:** zero recognizer submissions is allowed only when the cluster is explicitly classified non-data with supporting reason codes.

A content-bearing accepted cluster that produces zero submissions is an automatic pipeline failure.

## Experiment program

Cache page geometry, aligned stacks, residual channels, and candidate components. Tune stages separately.

### Stage A: reference and page geometry

Compare:

- best-quality reference versus registration-graph medoid;
- one reference versus per-page references;
- direct transforms versus composed transforms with local refinement;
- projection/gutter, contour, and line-based page segmentation combinations.

### Stage B: joint residual aggregation

Compare:

- adjacent-pair union;
- all-pair union within a cluster;
- residual to a robust median stack;
- per-pixel view-state signatures;
- material-only, ink-only, and combined residuals;
- single-scale and multiscale residuals.

The preferred approach should recover states visible non-adjacently and avoid multiplying one physical sheet into several pairwise crops.

### Stage C: component merging and boundary expansion

Sweep:

- residual thresholds and morphology scale;
- component overlap and view-signature similarity;
- boundary-search radius;
- line/contour support weight;
- physical fill and compactness requirements;
- crop padding;
- split-versus-merge criteria for nearby sheets.

### Stage D: state suppression and deduplication

Sweep:

- minimum data-bearing ink;
- frontness and translucency indicators;
- strict crop near-duplicate thresholds;
- allowed local registration tolerance;
- state-signature equality and subset rules;
- uncertainty thresholds.

### Stage E: best-view selection

Compare quality weightings and complementary-view selection. Favor target recall and completeness over small improvements in aesthetic crop tightness.

Use a constrained objective:

1. no positive content-bearing case may produce zero submissions;
2. maximize logical target recall;
3. minimize duplicate, blank, reverse, and unrelated submissions;
4. maximize crop coverage and IoU;
5. minimize unnecessary supporting views and runtime.

## Metrics

Match predicted submissions to expected logical targets with maximum-weight bipartite matching. Matching should consider target identity, allowed source views, kind/side, completeness, and bbox or polygon overlap.

Report:

- logical target precision, recall, and F1;
- exact and minimum submission-count accuracy;
- zero-crop positive failures;
- duplicate-submission rate;
- blank/reverse/unrelated submission rate;
- bbox IoU and polygon IoU when available;
- expected-content coverage;
- over-crop and under-crop fractions;
- complete-versus-partial classification accuracy;
- best-view selection accuracy;
- oracle-cluster versus end-to-end performance;
- runtime per cluster and per output state.

A cluster-level reviewed case passes only when all required logical targets are represented, the minimum submission count is met, and no forbidden false submission is present.

## Required diagnostics and review interface

For every cluster, save:

- source thumbnails and full-resolution links;
- transform graph and registration confidence;
- page/spread boundaries;
- aligned stack blink/slider views;
- robust reference image;
- material, ink, and combined residuals;
- candidate components and view-state signatures;
- raw residual boxes versus expanded physical boundaries;
- all observations grouped by logical state;
- suppressed candidates with reason codes;
- selected canonical and supporting submissions;
- expected versus predicted crop overlays when labels exist.

The HTML interface should support instant A/B blinking, zoom/pan, keyboard navigation, persistent labels, and filters for:

- oracle versus predicted clusters;
- pass/fail;
- base/overlay/partial/blank/reverse/uncertain;
- complete versus partial;
- zero-crop failures;
- duplicate and extra submissions;
- source folder and form family.

## Refactoring requirements for the current prototype

The existing cropper branch is a useful baseline, not an architecture constraint. Before large sweeps:

1. Replace import-time monkey-patching with explicit pipeline stages.
2. Make crop recovery consume `ClusteringResult` directly; do not recreate cluster membership internally.
3. Share decoding, soft-ink, registration, transform, and residual utilities with clustering through stable low-level modules.
4. Keep crop-specific page-local refinement separate from pair identity decisions.
5. Make every threshold part of a serializable `CropConfig` with no filename- or form-specific exceptions.
6. Cache expensive per-image and per-pair intermediates by content/configuration hash.
7. Preserve masks, polygons, transforms, and reason codes in the result schema rather than only rendered JPEG crops.
8. Add schema-versioned serialization and round-trip tests.

## Agent deliverables

1. Convert the expanded crop annotations into versioned logical-target manifests.
2. Implement oracle-cluster and end-to-end evaluators using the same metrics.
3. Refactor the prototype into explicit stages before broad parameter search.
4. Run the staged ablations and parameter sweeps above.
5. Produce a Pareto table and reviewed HTML, not only aggregate JSON.
6. Add regression tests for literal duplicates, one-overlay pairs, multi-view triplets, two-page spreads, blank occluders, reverse sheets, and partial-best-available cases.
7. Guarantee that every content-bearing singleton or positive cluster emits at least one submission.
8. Preserve the entire physical cluster; select at most four supporting observations only at the final submission stage.
9. Do not add record-, filename-, folder-, or form-specific heuristics.
10. Freeze the approach before running the locked audit partition.
