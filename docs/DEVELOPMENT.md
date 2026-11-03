# Development and operations

This page covers local setup, the supported command-line workflows, tests, and the limits of what this repository can deploy. It is intentionally concrete about repository commands and does not invent a hosted or cloud deployment procedure that the project does not contain.

## Prerequisites

- Python **3.10 or newer**. The package metadata declares `requires-python = ">=3.10"`; CI currently tests Python 3.10 and 3.12 on Ubuntu.
- A local filesystem containing readable source images. Supported evaluation suffixes are `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.j2k`, and `.jp2`.
- OpenCV image decoding support for the formats used by the input dataset.
- Enough disk space for feature caches, crops, annotated images, HTML review assets, and JSON/CSV reports.

There is no lockfile, checked-in virtual environment, container definition, cloud credential configuration, or deployment manifest in this repository. Pin the environment externally if a reproducible operational environment is required.

## Install

From the repository root:

```bash
python -m pip install -e .
```

For development tools and the test suite:

```bash
python -m pip install -e ".[dev]"
```

The runtime dependencies are declared in `pyproject.toml`: NumPy, headless OpenCV, PyYAML, and tqdm. The development extras add pytest and Ruff. Do not add credentials or private data to the repository while configuring a local environment.

## Repository layout

```text
image_clustering/
├── REVIEW_TOOL.md          canonical agent/operator entry point for physical-document review
├── src/image_clustering/
│   ├── clustering/         identity and registration of document scenes
│   ├── cropping/           unique-page and foreground-sheet recovery
│   ├── evaluation/         reviewed labels and saved-run comparison
│   └── review/             interactive HTTP review application
├── scripts/
│   ├── evaluate_dataset.py  complete evaluation/reporting layer
│   └── review_clusters.py   compatibility wrapper for image-review
├── configs/                checked-in clustering and cropper configuration
├── examples/evaluation/    reviewed label examples; source images stay private
├── tests/                  unit and integration tests
├── docs/                   workflow, architecture, calibration, and maintenance docs
└── .github/workflows/      CI validation
```

`ARCHITECTURE.md` describes ownership and data flow. [`../REVIEW_TOOL.md`](../REVIEW_TOOL.md) is the canonical identity and launch guide for the standalone Physical Document Duplicate/Occlusion Review Tool; [`REVIEW_WORKFLOW.md`](REVIEW_WORKFLOW.md) is its detailed operating guide.

## Console entry points

The five console scripts are declared in `pyproject.toml`:

| Command | Source | Use |
| --- | --- | --- |
| `image-cluster` | `src/image_clustering/cli.py` | Cluster nearby ordered captures and write a saved clustering result. |
| `image-crop` | `src/image_clustering/cropping/cli.py` | Recover unique pages/sheets from an input directory or saved `clustering.json`. |
| `image-label` | `src/image_clustering/evaluation/cli.py` | Append, validate, or list reviewed pair/cluster/crop labels. |
| `image-review` | `src/image_clustering/review/cli.py` | Start the local interactive review server for a completed evaluation root. |
| `image-compare` | `src/image_clustering/evaluation/comparison_cli.py` | Compare two saved clustering runs by exact stable image-ID membership sets. |

Use `--help` after installation to inspect the current parser. Flag spelling is part of the current interface: the package CLIs use names such as `--input_dir` and `--output_dir`, while `scripts/evaluate_dataset.py` uses hyphenated flags such as `--input-dir`.

## Basic clustering and crop commands

Cluster a directory using the checked-in configuration:

```bash
image-cluster \
  --input_dir /path/to/images \
  --output_dir /path/to/results \
  --config configs/default.json
```

The clustering output contains `clustering.json`, `pair_scores.csv`, and `run.json` under the output directory.

Recover crops from the saved clustering result:

```bash
image-crop \
  --clustering_json /path/to/results/clustering.json \
  --output_dir /path/to/crop-results \
  --crop_config configs/cropping_default.yaml
```

The cropper writes crop images and annotations plus aggregate `cropping.json`. For a curated neighbor sample, use the optional `--triplet_manifest` path described in the root README and the clustering discovery code; images in independent manifest groups are not compared across groups.

## Full evaluation and review

For a complete, reproducible evaluation root, use `scripts/evaluate_dataset.py`. It inventories source files, reuses only matching saved clustering provenance, runs clustering and crop recovery, writes reports and static HTML review assets, and produces validation/final-report files.

The script requires `--input-dir`, `--output-dir`, `--cluster-config`, `--crop-config`, `--git-commit`, and a final `--commands` list. A minimal POSIX-shell example is:

```bash
python scripts/evaluate_dataset.py \
  --input-dir /path/to/images \
  --output-dir /path/to/evaluation \
  --cluster-config configs/default.json \
  --crop-config configs/cropping_default.yaml \
  --git-commit "$(git rev-parse HEAD)" \
  --commands python scripts/evaluate_dataset.py --input-dir /path/to/images --output-dir /path/to/evaluation
```

On PowerShell, replace the line-continuation backslashes with backticks or use one line; `$(git rev-parse HEAD)` is also a valid PowerShell subexpression. Because `--commands` consumes the remaining arguments, it must be last. Add `--pair-labels`, `--triplet-manifest`, or `--show-progress` only when required by the run.

After the evaluation succeeds, use the purpose-built interactive review server:

```bash
image-review --output_dir /path/to/evaluation
```

Read [`REVIEW_WORKFLOW.md`](REVIEW_WORKFLOW.md) before reviewing or exporting decisions. It explains the input contract, state location, status semantics, and the difference between the interactive server and the generated static HTML review.

To compare two saved clustering manifests:

```bash
image-compare \
  --run-a /path/to/run-a \
  --run-b /path/to/run-b \
  --label-a baseline \
  --label-b candidate
```

Each run can be an evaluation root containing `clustering/clustering.json` or a direct manifest path. The default outputs are `run-a/review/compare.html` and the adjacent `compare.json`.

## Configuration

- `configs/default.json` contains clustering thresholds, feature limits, working dimension, candidate gap, contradiction gates, occlusion gates, and worker settings. It is a calibrated conservative baseline, not a universal production profile.
- `configs/cropping_default.yaml` overrides packaged cropper defaults for saving crops/annotations, resume behavior, pair search, worker count, OpenCV threads, and logging.
- `examples/evaluation/content_pair_labels.json` is the default label file used by `evaluate_dataset.py`; reviewed cases are append-friendly JSONL/JSON data and should be kept separate from private source images.

When changing thresholds, update the calibration/experiment documentation and add or update regression cases. Do not introduce filename-, folder-, person-, or form-specific exceptions.

## Testing and CI

Run the full local checks from the repository root:

```bash
python -m ruff check . --output-format concise
python -m pytest -q
```

For a focused review check:

```bash
python -m pytest -q tests/test_review_server.py tests/test_review_decisions.py
```

The CI workflow at `.github/workflows/tests.yml` runs the editable development install, Ruff, and pytest on Python 3.10 and 3.12. Review-server tests use temporary local images and an ephemeral port; they do not require a network service or private dataset.

When adding behavior, update tests near the owning subsystem. The review contract is covered by `test_review_server.py`, `test_review_decisions.py`, and evaluation fixture tests in `test_evaluation_dataset.py`.

## Artifacts and cleanup

Keep generated evaluation output outside the source tree when possible, for example `/path/to/results` or a dated run directory. A typical evaluation root includes:

```text
inventory/
clustering/
cropping.json
annotated/
submissions/
reports/
review/
review_labels/
logs/
```

The repository ignores common generated directories, caches, logs, and `review_labels/`. Do not delete a review directory until its decisions and exports have been archived or intentionally discarded. Canonical clustering/crop artifacts should be treated as immutable inputs to review; regenerate a new evaluation root when changing the source/configuration provenance.

## Deployment and production boundary

No deployment procedure is defined here. The review server is a local loopback tool with no authentication, not a production web service. There is also no repository evidence that a large unattended production run is approved: the generated final report recommends a staged pilot and human audit before an approximately 80,000-image corpus. Any hosted deployment, credential integration, scheduling, resource sizing, or rollback process requires a separate design and documentation effort.
