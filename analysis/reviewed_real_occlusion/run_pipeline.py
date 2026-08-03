"""Run the grouped reviewed-real occlusion evaluation end to end."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from collections.abc import Sequence
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
                f"- Baseline passed: {baseline['promotion_gates']['passed']}",
                f"- Final passed: {calibrated['promotion_gates']['passed']}",
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
    parser.add_argument("--config", type=Path)
    parser.add_argument("--subtypes", type=Path)
    parser.add_argument("--preserve-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    prepared_dir = output_root / "prepared"
    stage_root = output_root / "evaluation_input"
    cache_dir = output_root / ".exact_cache"
    if not args.preserve_cache and cache_dir.exists():
        shutil.rmtree(cache_dir)

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
    config = ClusterConfig.from_json(args.config)

    cold_start = time.perf_counter()
    cold_result = _cluster_reviewed_groups(
        groups,
        config=config,
        cache_dir=cache_dir,
    )
    cold_seconds = time.perf_counter() - cold_start
    baseline_clustering = output_root / "baseline_clustering"
    write_result(cold_result, baseline_clustering, config=config)

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
    }
    (output_root / "runtime_report.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    subtype_path = args.subtypes or (
        prepared_dir / "accepted_group_occlusion_subtypes.csv"
    )
    baseline_dir = output_root / "baseline_evaluation"
    baseline = evaluate_predictions(
        prepared_dir,
        baseline_clustering / "clustering.json",
        baseline_dir,
        subtype_path=subtype_path,
        run_label="current-master-baseline",
    )
    calibrator_path = output_root / "real_probability_calibrator.json"
    fit_real_calibration(
        prepared_dir,
        baseline_dir / "pair_results.jsonl",
        calibrator_path,
        subtype_path=subtype_path,
    )
    final_dir = output_root / "final_evaluation"
    calibrated = evaluate_predictions(
        prepared_dir,
        baseline_clustering / "clustering.json",
        final_dir,
        subtype_path=subtype_path,
        calibrator_path=calibrator_path,
        run_label="reviewed-real-calibrated",
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
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "baseline_promotion_gates": baseline["promotion_gates"],
                "final_promotion_gates": calibrated["promotion_gates"],
                "runtime": runtime,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if calibrated["promotion_gates"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
