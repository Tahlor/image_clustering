"""Run the grouped reviewed-real occlusion evaluation with an explicit audit boundary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import time
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from image_clustering.clustering import (
    ClusterConfig,
    ClusteringResult,
    cluster_images,
    write_result,
)
from image_clustering.evaluation.reviewed_groups import (
    compare_runs,
    evaluate_predictions,
    fit_real_calibration,
    prepare_dataset,
)

TUNING_SPLITS = frozenset({"development", "selection"})
ALL_SPLITS = frozenset({*TUNING_SPLITS, "locked_audit"})
PHASES = ("baseline", "calibrate", "locked-audit")
FREEZE_SCHEMA = "reviewed-real-frozen-system-v1"
AUDIT_SCHEMA = "reviewed-real-locked-audit-execution-v1"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_evaluation_inputs(path: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups[row["original_cluster_id"]].append(row)
    if len(groups) != 200:
        raise ValueError(f"expected 200 reviewed groups, found {len(groups)}")
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["sequence_index"]))
    return dict(sorted(groups.items()))


def _load_split_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            group_id = str(row.get("original_cluster_id") or "")
            split = str(row.get("split") or "")
            if not group_id or split not in ALL_SPLITS:
                raise ValueError(
                    f"invalid split assignment at {path}:{line_number}"
                )
            if group_id in assignments:
                raise ValueError(f"duplicate split assignment: {group_id}")
            assignments[group_id] = split
    if len(assignments) != 200:
        raise ValueError(f"expected 200 split assignments, found {len(assignments)}")
    return assignments


def _groups_for_splits(
    groups: Mapping[str, list[dict[str, str]]],
    split_assignments: Mapping[str, str],
    include_splits: Collection[str],
) -> dict[str, list[dict[str, str]]]:
    selected = frozenset(include_splits)
    unknown = selected - ALL_SPLITS
    if not selected or unknown:
        raise ValueError(f"invalid requested split set: {sorted(selected)}")
    output = {
        group_id: rows
        for group_id, rows in groups.items()
        if split_assignments[group_id] in selected
    }
    if not output:
        raise ValueError(f"no groups found for splits: {sorted(selected)}")
    return output


def _cluster_reviewed_groups(
    groups: dict[str, list[dict[str, str]]],
    *,
    config: ClusterConfig,
    cache_dir: Path,
) -> ClusteringResult:
    images = []
    clusters = []
    comparisons = []
    for original_cluster_id, rows in groups.items():
        paths = [Path(row["staged_path"]) for row in rows]
        result = cluster_images(
            paths,
            sequence_id=original_cluster_id,
            config=config,
            cache_dir=cache_dir,
            show_progress=False,
        )
        images.extend(result.images)
        comparisons.extend(result.comparisons)
        clusters.extend(
            replace(
                cluster,
                cluster_id=f"{original_cluster_id}:{cluster.cluster_id}",
            )
            for cluster in result.clusters
        )
    return ClusteringResult(
        config_fingerprint=config.fingerprint(),
        input_root=Path(next(iter(groups.values()))[0]["staged_path"]).parents[1],
        grouping_mode="reviewed_original_cluster",
        group_manifest=None,
        images=tuple(images),
        clusters=tuple(clusters),
        comparisons=tuple(comparisons),
    )


def _stable_result_payload(result: ClusteringResult) -> dict[str, Any]:
    payload = result.to_dict()
    payload.pop("input_root", None)
    payload.pop("group_manifest", None)
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"missing required {label}: {resolved}")
    return resolved


def _validate_code_commit(value: str | None) -> str:
    normalized = str(value or "").lower()
    if not COMMIT_RE.fullmatch(normalized):
        raise ValueError(
            "--code-commit must be the exact 40-character frozen commit SHA"
        )
    return normalized


def _phase_file_map(
    args: argparse.Namespace,
    output_root: Path,
    subtype_path: Path,
) -> dict[str, Path]:
    if args.config is None:
        raise ValueError("--config is required for every governed phase")
    return {
        "assignments_csv": _require_file(
            args.assignments_csv, "assignments CSV"
        ),
        "assignments_jsonl": _require_file(
            args.assignments_jsonl, "assignments JSONL"
        ),
        "config": _require_file(args.config, "detector config"),
        "subtypes": _require_file(subtype_path, "completed subtype sidecar"),
        "baseline_clustering": _require_file(
            output_root / "baseline_clustering" / "clustering.json",
            "baseline clustering",
        ),
        "calibrator": _require_file(
            output_root / "real_probability_calibrator.json",
            "real probability calibrator",
        ),
    }


def _write_freeze_receipt(
    path: Path,
    *,
    args: argparse.Namespace,
    output_root: Path,
    subtype_path: Path,
    config: ClusterConfig,
) -> dict[str, Any]:
    code_commit = _validate_code_commit(args.code_commit)
    files = _phase_file_map(args, output_root, subtype_path)
    payload = {
        "schema_version": FREEZE_SCHEMA,
        "code_commit": code_commit,
        "config_fingerprint": config.fingerprint(),
        "evaluated_splits_before_freeze": sorted(TUNING_SPLITS),
        "locked_audit_included_before_freeze": False,
        "locked_audit_used_for_fit": False,
        "graph_edge_policy": (
            "unchanged; probabilities are review-only and cannot create edges"
        ),
        "input_hashes": {
            name: {
                "path": str(file_path),
                "sha256": _sha256(file_path),
            }
            for name, file_path in sorted(files.items())
        },
    }
    _write_json(path, payload)
    return payload


def _verify_freeze_receipt(
    path: Path,
    *,
    args: argparse.Namespace,
    output_root: Path,
    subtype_path: Path,
    config: ClusterConfig,
) -> dict[str, Any]:
    receipt_path = _require_file(path, "frozen-system receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != FREEZE_SCHEMA:
        raise ValueError("unsupported frozen-system receipt schema")
    code_commit = _validate_code_commit(args.code_commit)
    if receipt.get("code_commit") != code_commit:
        raise ValueError("current code commit differs from the frozen receipt")
    if receipt.get("config_fingerprint") != config.fingerprint():
        raise ValueError("current detector config differs from the frozen receipt")
    if receipt.get("locked_audit_included_before_freeze") is not False:
        raise ValueError("frozen receipt indicates pre-freeze locked-audit exposure")
    if receipt.get("locked_audit_used_for_fit") is not False:
        raise ValueError("frozen receipt indicates locked-audit calibration use")

    files = _phase_file_map(args, output_root, subtype_path)
    expected = receipt.get("input_hashes")
    if not isinstance(expected, dict) or set(expected) != set(files):
        raise ValueError("frozen-system input inventory differs from current inputs")
    for name, file_path in sorted(files.items()):
        if expected[name].get("sha256") != _sha256(file_path):
            raise ValueError(f"frozen-system input hash changed: {name}")
    return receipt


def _run_cold_warm(
    groups: dict[str, list[dict[str, str]]],
    *,
    config: ClusterConfig,
    cache_dir: Path,
    result_dir: Path,
    runtime_path: Path,
    preserve_cache: bool,
    evaluated_splits: Collection[str],
) -> tuple[ClusteringResult, dict[str, Any]]:
    if not preserve_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

    cold_start = time.perf_counter()
    cold_result = _cluster_reviewed_groups(
        groups,
        config=config,
        cache_dir=cache_dir,
    )
    cold_seconds = time.perf_counter() - cold_start
    write_result(cold_result, result_dir, config=config)

    warm_start = time.perf_counter()
    warm_result = _cluster_reviewed_groups(
        groups,
        config=config,
        cache_dir=cache_dir,
    )
    warm_seconds = time.perf_counter() - warm_start
    warm_exact = _stable_result_payload(cold_result) == _stable_result_payload(
        warm_result
    )
    if not warm_exact:
        raise ValueError("warm exact-cache replay changed the clustering result")

    runtime = {
        "cold_seconds": cold_seconds,
        "warm_seconds": warm_seconds,
        "warm_speedup": cold_seconds / warm_seconds if warm_seconds else None,
        "warm_replay_exact": warm_exact,
        "image_count": len(cold_result.images),
        "comparison_count": len(cold_result.comparisons),
        "component_count": len(cold_result.clusters),
        "original_cluster_count": len(groups),
        "evaluated_splits": sorted(evaluated_splits),
    }
    _write_json(runtime_path, runtime)
    return cold_result, runtime


def _write_final_report(
    path: Path,
    *,
    baseline: dict[str, Any],
    calibrated: dict[str, Any],
    runtime: dict[str, Any],
) -> None:
    locked = calibrated["by_split"]["locked_audit"]
    path.write_text(
        "\n".join(
            [
                "# Reviewed real occlusion evaluation",
                "",
                "## Promotion gates",
                "",
                f"- Frozen baseline passed: {baseline['promotion_gates']['passed']}",
                "- Frozen calibrated run passed: "
                f"{calibrated['promotion_gates']['passed']}",
                "- Probability calibration is review-only; graph edges are unchanged.",
                "",
                "## Locked real audit",
                "",
                f"- Pair count: {locked['pair_count']}",
                f"- Group count: {locked['group_count']}",
                "- Same-document connected recall: "
                f"{locked['same_document_recall_connected']}",
                f"- Candidate recall: {locked['candidate_recall']}",
                f"- Candidate precision: {locked['candidate_precision']}",
                f"- Negative false-link rate: {locked['negative_false_link_rate']}",
                "- Contaminated components: "
                f"{locked['contaminated_component_count']}",
                "",
                "## Runtime",
                "",
                f"- Cold run seconds: {runtime['cold_seconds']:.3f}",
                f"- Warm run seconds: {runtime['warm_seconds']:.3f}",
                f"- Exact warm replay: {runtime['warm_replay_exact']}",
                "",
                "See the JSON/CSV outputs for every pair, group, failure, "
                "calibration bin, and review operating point.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--assignments-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subtypes", type=Path)
    parser.add_argument("--preserve-cache", action="store_true")
    parser.add_argument("--phase", choices=PHASES, default="baseline")
    parser.add_argument(
        "--code-commit",
        help=(
            "Exact 40-character commit SHA. Required when freezing calibration "
            "or executing the locked audit."
        ),
    )
    return parser.parse_args(argv)


def _prepare(
    args: argparse.Namespace,
    output_root: Path,
) -> tuple[
    Path,
    dict[str, list[dict[str, str]]],
    dict[str, str],
    ClusterConfig,
    Path,
]:
    prepared_dir = output_root / "prepared"
    stage_root = output_root / "evaluation_input"
    prepare_dataset(
        args.assignments_csv,
        prepared_dir,
        jsonl_path=args.assignments_jsonl,
        package_root=args.package_root,
        stage_root=stage_root,
    )
    groups = _load_evaluation_inputs(
        prepared_dir / "evaluation_input_manifest.csv"
    )
    split_assignments = _load_split_assignments(
        prepared_dir / "split_manifest.jsonl"
    )
    config = ClusterConfig.from_json(args.config)
    subtype_path = args.subtypes or (
        prepared_dir / "accepted_group_occlusion_subtypes.csv"
    )
    return prepared_dir, groups, split_assignments, config, subtype_path


def _run_baseline(
    args: argparse.Namespace,
    *,
    output_root: Path,
    prepared_dir: Path,
    groups: dict[str, list[dict[str, str]]],
    split_assignments: dict[str, str],
    config: ClusterConfig,
    subtype_path: Path,
) -> int:
    tuning_groups = _groups_for_splits(
        groups,
        split_assignments,
        TUNING_SPLITS,
    )
    baseline_clustering = output_root / "baseline_clustering"
    _, runtime = _run_cold_warm(
        tuning_groups,
        config=config,
        cache_dir=output_root / ".exact_cache" / "baseline",
        result_dir=baseline_clustering,
        runtime_path=output_root / "baseline_runtime_report.json",
        preserve_cache=args.preserve_cache,
        evaluated_splits=TUNING_SPLITS,
    )
    baseline_dir = output_root / "baseline_evaluation"
    baseline = evaluate_predictions(
        prepared_dir,
        baseline_clustering / "clustering.json",
        baseline_dir,
        subtype_path=subtype_path,
        run_label="current-master-baseline-tuning-splits",
        include_splits=TUNING_SPLITS,
    )
    receipt = {
        "phase": "baseline",
        "evaluated_splits": sorted(TUNING_SPLITS),
        "locked_audit_included": False,
        "calibration_fitted": False,
        "runtime": runtime,
        "promotion_gates": baseline["promotion_gates"],
    }
    _write_json(output_root / "baseline_phase_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if baseline["promotion_gates"]["passed"] else 2


def _run_calibration(
    args: argparse.Namespace,
    *,
    output_root: Path,
    prepared_dir: Path,
    config: ClusterConfig,
    subtype_path: Path,
) -> int:
    if args.subtypes is None:
        raise ValueError(
            "--subtypes must identify the completed accepted-group sidecar "
            "before calibration is frozen"
        )
    _validate_code_commit(args.code_commit)
    baseline_clustering = _require_file(
        output_root / "baseline_clustering" / "clustering.json",
        "baseline clustering",
    )
    baseline_metrics = _require_file(
        output_root / "baseline_evaluation" / "metrics.json",
        "baseline metrics",
    )
    baseline_pairs = _require_file(
        output_root / "baseline_evaluation" / "pair_results.jsonl",
        "baseline pair results",
    )
    calibrator_path = output_root / "real_probability_calibrator.json"
    fit_real_calibration(
        prepared_dir,
        baseline_pairs,
        calibrator_path,
        subtype_path=subtype_path,
    )
    calibrated_dir = output_root / "calibrated_tuning_evaluation"
    calibrated = evaluate_predictions(
        prepared_dir,
        baseline_clustering,
        calibrated_dir,
        subtype_path=subtype_path,
        calibrator_path=calibrator_path,
        run_label="reviewed-real-calibrated-tuning-splits",
        include_splits=TUNING_SPLITS,
    )
    compare_runs(
        baseline_metrics,
        calibrated_dir / "metrics.json",
        output_root / "tuning_before_after_report.json",
    )
    freeze_path = output_root / "frozen_system_receipt.json"
    freeze = _write_freeze_receipt(
        freeze_path,
        args=args,
        output_root=output_root,
        subtype_path=subtype_path,
        config=config,
    )
    receipt = {
        "phase": "calibrate",
        "evaluated_splits": sorted(TUNING_SPLITS),
        "locked_audit_included": False,
        "locked_audit_used_for_fit": False,
        "freeze_receipt": str(freeze_path.resolve()),
        "freeze_receipt_sha256": _sha256(freeze_path),
        "code_commit": freeze["code_commit"],
        "promotion_gates": calibrated["promotion_gates"],
    }
    _write_json(output_root / "calibration_phase_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if calibrated["promotion_gates"]["passed"] else 2


def _run_locked_audit(
    args: argparse.Namespace,
    *,
    output_root: Path,
    prepared_dir: Path,
    groups: dict[str, list[dict[str, str]]],
    config: ClusterConfig,
    subtype_path: Path,
) -> int:
    if args.subtypes is None:
        raise ValueError("--subtypes is required for the frozen locked audit")
    audit_receipt_path = output_root / "locked_audit_execution_receipt.json"
    if audit_receipt_path.exists():
        raise ValueError(
            "locked audit already has an execution receipt; refusing to "
            "silently evaluate it again"
        )
    freeze_path = output_root / "frozen_system_receipt.json"
    freeze = _verify_freeze_receipt(
        freeze_path,
        args=args,
        output_root=output_root,
        subtype_path=subtype_path,
        config=config,
    )
    full_clustering = output_root / "frozen_full_clustering"
    _, runtime = _run_cold_warm(
        groups,
        config=config,
        cache_dir=output_root / ".exact_cache" / "frozen_full",
        result_dir=full_clustering,
        runtime_path=output_root / "frozen_full_runtime_report.json",
        preserve_cache=args.preserve_cache,
        evaluated_splits=ALL_SPLITS,
    )
    baseline_dir = output_root / "frozen_full_baseline_evaluation"
    baseline = evaluate_predictions(
        prepared_dir,
        full_clustering / "clustering.json",
        baseline_dir,
        subtype_path=subtype_path,
        run_label="frozen-full-baseline",
        include_splits=ALL_SPLITS,
    )
    calibrator_path = output_root / "real_probability_calibrator.json"
    final_dir = output_root / "final_evaluation"
    calibrated = evaluate_predictions(
        prepared_dir,
        full_clustering / "clustering.json",
        final_dir,
        subtype_path=subtype_path,
        calibrator_path=calibrator_path,
        run_label="frozen-reviewed-real-calibrated",
        include_splits=ALL_SPLITS,
    )
    compare_runs(
        baseline_dir / "metrics.json",
        final_dir / "metrics.json",
        output_root / "before_after_report.json",
    )
    _write_final_report(
        output_root / "FINAL_REPORT.md",
        baseline=baseline,
        calibrated=calibrated,
        runtime=runtime,
    )
    receipt = {
        "schema_version": AUDIT_SCHEMA,
        "phase": "locked-audit",
        "code_commit": freeze["code_commit"],
        "frozen_receipt": str(freeze_path.resolve()),
        "frozen_receipt_sha256": _sha256(freeze_path),
        "evaluated_splits": sorted(ALL_SPLITS),
        "locked_audit_included": True,
        "locked_audit_used_for_fit": False,
        "runtime": runtime,
        "baseline_promotion_gates": baseline["promotion_gates"],
        "final_promotion_gates": calibrated["promotion_gates"],
    }
    _write_json(audit_receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if calibrated["promotion_gates"]["passed"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    (
        prepared_dir,
        groups,
        split_assignments,
        config,
        subtype_path,
    ) = _prepare(args, output_root)
    if args.phase == "baseline":
        return _run_baseline(
            args,
            output_root=output_root,
            prepared_dir=prepared_dir,
            groups=groups,
            split_assignments=split_assignments,
            config=config,
            subtype_path=subtype_path,
        )
    if args.phase == "calibrate":
        return _run_calibration(
            args,
            output_root=output_root,
            prepared_dir=prepared_dir,
            config=config,
            subtype_path=subtype_path,
        )
    return _run_locked_audit(
        args,
        output_root=output_root,
        prepared_dir=prepared_dir,
        groups=groups,
        config=config,
        subtype_path=subtype_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
