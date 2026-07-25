# Conservative Document-View Clustering: Experiment Strategy

## Objective

Determine whether nearby archival images show the **same physical document scene**, possibly under a different occlusion. This is not form-template clustering.

A pair may be accepted only under one of two hypotheses:

1. **Near duplicate:** after registration, the visible material and document-specific content agree almost everywhere.
2. **Physical occlusion:** after registration, one large locally coherent material region explains nearly all disagreement, while the visible content outside that region agrees extremely well.

A pair must be rejected when document-specific ink differs outside the proposed occlusion. Different handwriting, names, dates, signatures, annotations, stamps, or stains are negative evidence even when the printed form aligns perfectly.

The cost is asymmetric. A false split causes redundant downstream work; a false merge can combine different people or records. Optimize for extremely high precision and allow an abstention/review band.

## Dataset protocol

The new curated dataset is appropriate for parameter selection and validation, but not for fitting a learned task-specific classifier. Operationally, once a parameter sweep uses examples to choose a configuration, those examples are a **development set**, not a final untouched validation set.

Create three grouped partitions when the set is large enough:

- **Sweep/development:** broad parameter search and ablations.
- **Selection:** choose among the small Pareto set produced on development.
- **Locked audit:** run only after the implementation and thresholds are frozen.

If the data is too small for three partitions, use repeated grouped cross-validation for development and preserve the largest feasible locked audit subset.

Never split individual pairs randomly. Keep together:

- every image from the same physical-document cluster;
- all candidate pairs from the same capture sequence;
- nearby same-template hard negatives;
- preferably each form family or volume/folder.

This prevents a configuration from being rewarded for memorizing one capture run or one form family. The label schema should distinguish at least:

- `same_near_duplicate`;
- `same_physical_occlusion`;
- `different_same_template`;
- `different_other`;
- `uncertain`.

Also maintain canonical ground-truth component IDs so evaluation is not limited to labeled pair edges.

## Primary success criteria

Choose configurations lexicographically rather than by ordinary accuracy:

1. Minimize contaminated output components: a predicted cluster containing more than one ground-truth physical document.
2. Minimize accepted `different_same_template` pairs.
3. Maximize recall on `same_physical_occlusion` pairs.
4. Maximize recall on `same_near_duplicate` pairs.
5. Minimize runtime and memory.

The preferred operating point is the highest positive recall among configurations with **zero contaminated components and zero curated hard-negative merges** on the development folds and selection set. If no configuration satisfies that constraint, fail closed and expose the uncertain pairs for review rather than relaxing until the headline average looks good.

Report at least:

- pair precision and recall by label subtype;
- accepted hard-negative count and rate;
- contaminated-component count and image-weighted contamination rate;
- ground-truth component split rate;
- near-duplicate recall;
- physical-occlusion recall;
- coverage at each confidence threshold;
- runtime per image and per candidate pair.

AUROC or average accuracy may be reported, but neither should select the production threshold.

## Recommended pipeline

### 1. Candidate generation

Compare only images from the same independent capture folder. Preserve filename order and test a small forward window such as `max_gap in {1, 2, 3, 4, 6}`. Candidate-generation recall must be reported separately from pair-classification recall.

Do not use global image embeddings to accept a pair. They are likely to retrieve visually similar copies of the same blank form, which is precisely the hard-negative case. A cheap perceptual hash or frozen global embedding may be used only to avoid obviously irrelevant comparisons.

### 2. Registration defines coordinates; it does not establish identity

Printed form lines and labels can produce an excellent affine transform for two different records. Consequently, SIFT/AKAZE/SuperPoint support, inlier count, and reprojection error are registration-quality evidence only.

Use this registration sequence:

1. Estimate a robust affine transform from local matches.
2. Reject implausible transforms, insufficient spatial support, and inadequate valid overlap.
3. Compute a preliminary change region.
4. Re-estimate the transform using only the proposed stable exterior.
5. Keep the refined transform only when exterior disagreement improves materially.
6. Try a homography only when it materially improves stable-area alignment without introducing implausible warping.

Sweep classical robust estimators, including USAC/MAGSAC++ where available, against the current RANSAC baseline. MAGSAC++ is specifically designed to improve robust geometric model estimation without requiring a single fixed noise scale.

Frozen pretrained local matchers are allowed as optional registration fallbacks. SuperPoint plus LightGlue, LoFTR, or Efficient LoFTR should be benchmarked only where classical registration fails. Their matches must pass the same geometric plausibility and downstream content tests. They must never directly vote that two records are the same.

### 3. Separate material change from ink change

Use two complementary residual families after alignment.

#### Material residual

This should respond to a physical overlay, folded sheet, page edge, shadow, paper tone, or changed exposed background. Candidate channels include:

- locally brightness-normalized grayscale residual;
- low-frequency residual after Gaussian or morphological background estimation;
- local mean and variance differences;
- boundary-gradient or shadow evidence;
- optional color residual when color is available.

#### Ink residual

This should respond strongly to different handwriting while tolerating compression noise, line-thickness variation, and slight registration error. Build a soft foreground representation from a small ensemble of inexpensive channels:

- background-subtracted dark-ink response;
- Scharr gradient magnitude;
- morphological black-hat response at handwriting stroke scales;
- one or more threshold masks from Otsu, Generalized Histogram Thresholding, or a local Sauvola-style threshold.

Prefer continuous response maps for scoring. Binary masks are useful for topology and connected components, but a single brittle threshold should not be the entire representation. Generalized Histogram Thresholding is a cheap candidate because it generalizes Otsu and Minimum Error Thresholding while retaining histogram-level cost.

### 4. Use tolerant symmetric unmatched-ink maps

Do not XOR two binary masks. Let small localization and stroke-width differences match within a distance tolerance.

For each direction, compute ink that has no corresponding foreground within radius `r` in the other image, using dilation or a distance transform. Combine the two directions into a symmetric unmatched-ink map. Sweep `r` in normalized image units so behavior is stable across resolutions.

Measure both mass and topology:

- unmatched-ink fraction;
- unmatched-ink component count and component areas;
- skeleton or centerline length;
- perimeter-to-area ratio;
- number and fraction of affected tiles;
- row/column dispersion;
- largest-component share;
- bidirectional balance.

Random noise should remain small and fragmented. Different handwriting should create locally coherent stroke systems, often distributed over multiple fields. This is the principal hard-negative signal.

### 5. Propose a physical occlusion region

Generate candidate regions primarily from robust material residual, supplemented by aggregate ink/intensity residual. Use robust local baselines rather than a global fixed residual threshold.

A valid occlusion candidate should satisfy most of the following:

- one dominant connected region, or a very small number of physically plausible regions;
- sufficient area but enough remaining exterior to verify identity;
- high share of total residual mass captured;
- local spatial coherence after morphology;
- plausible compactness, rectangularity, or convexity;
- physical-boundary, shadow, or paper-tone evidence when available;
- substantially higher disagreement inside than outside;
- stability across nearby threshold values and working resolutions.

The bounding rectangle alone is not sufficient. A rectangle drawn around distributed handwriting may be large, but its support will be thin, perforated, high-perimeter, and weak in low-frequency material evidence.

### 6. Apply a strict outside-content contradiction test

Dilate the candidate occlusion mask enough to exclude its uncertain boundary. On the remaining valid overlap, require extremely strong agreement in both soft ink and ordinary appearance.

Reject when any of these are pronounced outside the candidate:

- multiple coherent unmatched-ink components;
- long stroke or skeleton support;
- unmatched ink spread over many tiles or text rows;
- high local patch residual despite good registration;
- several separated name/date/signature-like fields changing;
- a second unexplained material-change region;
- page-wide residual inconsistent with one physical occluder.

Printed form agreement must not offset this veto. Acceptance evidence and contradiction evidence should remain separate; a very good form registration cannot compensate for different handwriting.

### 7. Keep two explicit acceptance routes

#### Route A: `near_duplicate`

Require:

- strong and spatially distributed registration support;
- high valid overlap;
- very low global material residual;
- very low tolerant unmatched-ink mass and dispersion;
- no coherent contradiction components.

This route should accept literal or near-literal recaptures such as `i4071655-00398` and `i4071655-00400` without inventing an occlusion.

#### Route B: `physical_occlusion`

Require:

- valid registration and sufficient stable exterior;
- one plausible large contiguous material region;
- high residual-mass capture by that region;
- very high appearance and ink agreement outside it;
- no distributed document-specific contradiction outside it.

Keep an explicit `uncertain` outcome between accept and reject thresholds. This will make production precision substantially easier to protect.

### 8. Construct clusters with cannot-link constraints

A positive pair edge is not enough to run unconstrained connected components. Build clusters with a confidence-ordered union procedure:

- accepted pair edges are candidate joins;
- direct hard contradictions are cannot-link constraints;
- two components may merge only when no known contradiction exists between any members;
- every non-singleton member must have at least one accepted direct edge;
- check transform-cycle consistency where triangles exist;
- never split a valid physical component merely because the downstream model accepts only four images.

Report pair decisions and final component decisions separately. One false positive edge can contaminate an entire component.

## Experiment program

Do not launch one enormous Cartesian grid. Cache registration, aligned images, residual channels, and ink representations, then tune in stages.

### Stage A: registration

Compare:

- SIFT versus AKAZE;
- RANSAC versus USAC/MAGSAC++;
- affine only versus affine with guarded homography fallback;
- one-pass versus stable-exterior refinement;
- optional frozen SuperPoint/LightGlue or LoFTR fallback.

Select for registration success on true positives while retaining transform plausibility. Do not use cluster labels to reward high feature overlap on same-template negatives.

### Stage B: content representation

Ablate:

- current grayscale residual only;
- background-subtracted ink;
- ink plus Scharr;
- ink plus black-hat;
- Otsu versus GHT versus local thresholding;
- hard masks versus soft maps;
- single-scale versus two- or three-scale foreground extraction.

The winning representation should sharply separate true positive exteriors from same-template/different-handwriting exteriors.

### Stage C: occlusion geometry

Sweep:

- working resolution;
- robust residual tile size;
- threshold quantile or robust-z threshold;
- morphology scale;
- minimum/maximum region area;
- residual-mass capture;
- compactness and fill requirements;
- exterior dilation margin;
- minimum verified exterior area.

### Stage D: contradiction and decision thresholds

Sweep:

- distance-transform tolerance;
- outside unmatched-ink mass;
- coherent-component area and skeleton length;
- affected-tile fraction;
- material residual outside the candidate;
- near-duplicate global residual;
- acceptance and abstention margins.

Use random search, Latin hypercube sampling, or Optuna/TPE over a bounded space rather than exhaustive enumeration. The objective should be lexicographic or constrained: maximize positive recall subject to zero hard-negative merges and zero contaminated components.

### Stage E: graph construction

Compare:

- current constrained union-find;
- maximum-spanning-forest joins with cannot-links;
- requirements for triangle/cycle consistency;
- confidence thresholds dependent on pair subtype.

A graph method should improve recall through legitimate intermediate occlusion views without allowing a same-template bridge.

## Robustness tests

For each promising configuration, rerun the same labels after synthetic nuisance transforms that preserve identity:

- mild brightness and contrast changes;
- blur and sharpening;
- JPEG/J2K recompression;
- low-amplitude noise;
- small affine perturbations;
- modest cropping at image boundaries.

The decision and reason code should be stable. These perturbations test invariance; they do not create additional independent labeled records.

## Required diagnostics and artifacts

Every experiment must be reproducible from a versioned config and emit:

- run ID, Git commit, dataset manifest hash, and partition ID;
- all pair metrics and reason codes in Parquet or CSV;
- predicted components and ground-truth component comparison;
- aligned A/B images;
- material residual and soft-ink maps;
- tolerant unmatched-ink map;
- proposed occlusion mask and boundary;
- exterior mask and contradiction components;
- threshold/coverage curves and Pareto frontier;
- a fast HTML review ordered first by contaminated components, false accepts, false rejects, then low-confidence correct decisions.

For every false merge, the report must show what positive evidence allowed acceptance and why the handwriting/material contradiction failed to veto it. For every false split, show which gate failed.

## Agent deliverables

1. Normalize the expanded curated labels into versioned pair and component manifests.
2. Freeze grouped development, selection, and audit partitions.
3. Implement cached stage interfaces so sweeps do not recompute image decoding and registration unnecessarily.
4. Run the staged ablations and parameter searches above.
5. Produce a Pareto table rather than only one aggregate score.
6. Select one conservative production configuration and one higher-recall review-assisted configuration.
7. Add regression tests for every curated hard negative and representative positive archetype.
8. Do not add filename-, folder-, form-, or person-specific exceptions.
9. Do not fit a task-specific neural network or learned classifier to this dataset.
10. Leave the locked audit partition untouched until the approach and thresholds are frozen.

## References

- Jonathan T. Barron, [A Generalization of Otsu's Method and Minimum Error Thresholding](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3657_ECCV_2020_paper.php), ECCV 2020.
- Daniel Barath et al., [MAGSAC++, a Fast, Reliable and Accurate Robust Estimator](https://openaccess.thecvf.com/content_CVPR_2020/html/Barath_MAGSAC_a_Fast_Reliable_and_Accurate_Robust_Estimator_CVPR_2020_paper.html), CVPR 2020.
- Daniel DeTone et al., [SuperPoint: Self-Supervised Interest Point Detection and Description](https://openaccess.thecvf.com/content_cvpr_2018_workshops/w9/html/DeTone_SuperPoint_Self-Supervised_Interest_CVPR_2018_paper.html), CVPR Workshops 2018.
- Philipp Lindenberger et al., [LightGlue: Local Feature Matching at Light Speed](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html), ICCV 2023.
- Jiaming Sun et al., [LoFTR: Detector-Free Local Feature Matching With Transformers](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html), CVPR 2021.
