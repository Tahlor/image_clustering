"""Aggregate comparison of two saved clustering runs.

The comparison deliberately treats a cluster as its exact set of stable image IDs.
Cluster IDs are run-local and may be renumbered between runs, so comparing those
IDs directly would report false differences.
"""

# Generated HTML strings intentionally remain readable as prose.
# ruff: noqa: E501

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any

from image_clustering.clustering.models import ClusteringResult, ImageCluster
from image_clustering.clustering.serialization import load_result

ClusterSignature = frozenset[str]


def resolve_clustering_manifest(path: Path) -> Path:
    """Resolve a run root or manifest path to a canonical clustering manifest."""
    candidate = path.expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "clustering" / "clustering.json"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Clustering manifest not found: {candidate}. "
            "Pass a result root containing clustering/clustering.json or that file directly."
        )
    return candidate


def load_comparison_run(path: Path) -> tuple[Path, ClusteringResult]:
    """Load and validate one saved clustering run for comparison."""
    manifest = resolve_clustering_manifest(path)
    result = load_result(manifest)
    _validate_result(result, manifest)
    return manifest, result


def _validate_result(result: ClusteringResult, manifest: Path) -> None:
    image_ids = [image.image_id for image in result.images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError(f"Duplicate image IDs in clustering manifest: {manifest}")

    cluster_ids = [cluster.cluster_id for cluster in result.clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError(f"Duplicate cluster IDs in clustering manifest: {manifest}")

    known_image_ids = set(image_ids)
    member_ids: list[str] = []
    signatures: set[ClusterSignature] = set()
    for cluster in result.clusters:
        if not cluster.image_ids:
            raise ValueError(
                f"Empty cluster {cluster.cluster_id!r} in clustering manifest: {manifest}"
            )
        if len(cluster.image_ids) != len(set(cluster.image_ids)):
            raise ValueError(
                f"Duplicate members in cluster {cluster.cluster_id!r}: {manifest}"
            )
        unknown = set(cluster.image_ids) - known_image_ids
        if unknown:
            raise ValueError(
                f"Cluster {cluster.cluster_id!r} references unknown image IDs: "
                f"{sorted(unknown)[:3]}"
            )
        signature = frozenset(cluster.image_ids)
        if signature in signatures:
            raise ValueError(
                f"Duplicate cluster membership set in clustering manifest: {manifest}"
            )
        signatures.add(signature)
        member_ids.extend(cluster.image_ids)

    member_counts = Counter(member_ids)
    if set(member_counts) != known_image_ids or any(
        count != 1 for count in member_counts.values()
    ):
        raise ValueError(
            "Clustering manifest must partition every image into exactly one cluster: "
            f"{manifest}"
        )


def _cluster_map(result: ClusteringResult) -> dict[ClusterSignature, ImageCluster]:
    return {
        frozenset(cluster.image_ids): cluster for cluster in result.clusters
    }


def _run_root(manifest: Path) -> Path:
    return (
        manifest.parent.parent
        if manifest.parent.name.lower() == "clustering"
        else manifest.parent
    )


def _run_summary(manifest: Path, result: ClusteringResult, label: str) -> dict[str, Any]:
    size_counts = Counter(len(cluster.image_ids) for cluster in result.clusters)
    return {
        "label": label,
        "manifest": str(manifest),
        "output_root": str(_run_root(manifest)),
        "input_root": str(result.input_root) if result.input_root else None,
        "config_fingerprint": result.config_fingerprint,
        "grouping_mode": result.grouping_mode,
        "group_manifest": str(result.group_manifest) if result.group_manifest else None,
        "image_count": len(result.images),
        "cluster_count": len(result.clusters),
        "sequence_count": len({image.sequence_id for image in result.images}),
        "cluster_size_distribution": {
            str(size): count for size, count in sorted(size_counts.items())
        },
    }


def _best_overlap(
    signature: ClusterSignature,
    other_signatures: dict[ClusterSignature, ImageCluster],
) -> dict[str, Any] | None:
    candidates: list[tuple[int, float, str, ImageCluster]] = []
    for other_signature, other_cluster in other_signatures.items():
        intersection_count = len(signature & other_signature)
        if intersection_count == 0:
            continue
        union_count = len(signature | other_signature)
        candidates.append(
            (
                intersection_count,
                intersection_count / union_count,
                other_cluster.cluster_id,
                other_cluster,
            )
        )
    if not candidates:
        return None
    intersection_count, jaccard, _, other_cluster = max(
        candidates,
        key=lambda candidate: (
            candidate[0],
            candidate[1],
            candidate[2],
        ),
    )
    return {
        "cluster_id": other_cluster.cluster_id,
        "image_count": len(other_cluster.image_ids),
        "overlap_image_count": intersection_count,
        "jaccard": round(jaccard, 6),
    }


def _unmatched_cluster_rows(
    signatures: set[ClusterSignature],
    clusters: dict[ClusterSignature, ImageCluster],
    other_clusters: dict[ClusterSignature, ImageCluster],
) -> list[dict[str, Any]]:
    rows = []
    for signature in signatures:
        cluster = clusters[signature]
        overlaps = [
            other_signature
            for other_signature in other_clusters
            if signature & other_signature
        ]
        rows.append(
            {
                "cluster_id": cluster.cluster_id,
                "sequence_id": cluster.sequence_id,
                "representative_image_id": cluster.representative_image_id,
                "image_count": len(cluster.image_ids),
                "image_ids": list(cluster.image_ids),
                "overlapping_cluster_count": len(overlaps),
                "best_overlap": _best_overlap(signature, other_clusters),
            }
        )
    return sorted(
        rows,
        key=lambda row: (-row["image_count"], row["cluster_id"]),
    )


def compare_result_paths(
    left_path: Path,
    right_path: Path,
    *,
    left_label: str | None = None,
    right_label: str | None = None,
) -> dict[str, Any]:
    """Compare two saved clustering result roots or manifest files."""
    left_manifest, left_result = load_comparison_run(left_path)
    right_manifest, right_result = load_comparison_run(right_path)
    left_summary = _run_summary(
        left_manifest,
        left_result,
        left_label or left_manifest.parent.parent.name,
    )
    right_summary = _run_summary(
        right_manifest,
        right_result,
        right_label or right_manifest.parent.parent.name,
    )
    left_clusters = _cluster_map(left_result)
    right_clusters = _cluster_map(right_result)
    left_signatures = set(left_clusters)
    right_signatures = set(right_clusters)
    shared_signatures = left_signatures & right_signatures
    left_only_signatures = left_signatures - right_signatures
    right_only_signatures = right_signatures - left_signatures

    left_images = {image.image_id for image in left_result.images}
    right_images = {image.image_id for image in right_result.images}
    common_images = left_images & right_images
    left_only_images = left_images - right_images
    right_only_images = right_images - left_images
    exact_match_image_count = sum(len(signature) for signature in shared_signatures)
    changed_membership_count = len(common_images) - exact_match_image_count
    union_image_count = len(left_images | right_images)

    exact_matches = sorted(
        (
            {
                "left_cluster_id": left_clusters[signature].cluster_id,
                "right_cluster_id": right_clusters[signature].cluster_id,
                "image_count": len(signature),
            }
            for signature in shared_signatures
        ),
        key=lambda row: (row["left_cluster_id"], row["right_cluster_id"]),
    )
    comparison = {
        "schema_version": 1,
        "comparison_basis": (
            "Exact cluster membership sets of stable image IDs. Cluster IDs are "
            "run-local and are not used as identity."
        ),
        "left": left_summary,
        "right": right_summary,
        "images": {
            "left_count": len(left_images),
            "right_count": len(right_images),
            "common_count": len(common_images),
            "left_only_count": len(left_only_images),
            "right_only_count": len(right_only_images),
            "union_count": union_image_count,
            "exact_match_cluster_image_count": exact_match_image_count,
            "changed_membership_count": changed_membership_count,
            "common_coverage": round(
                len(common_images) / union_image_count if union_image_count else 1.0,
                6,
            ),
        },
        "clusters": {
            "left_count": len(left_clusters),
            "right_count": len(right_clusters),
            "exact_match_count": len(shared_signatures),
            "left_only_count": len(left_only_signatures),
            "right_only_count": len(right_only_signatures),
            "left_only_with_overlap_count": sum(
                bool(signature & right_images) for signature in left_only_signatures
            ),
            "right_only_with_overlap_count": sum(
                bool(signature & left_images) for signature in right_only_signatures
            ),
        },
        "exact_matches": exact_matches,
        "left_only_clusters": _unmatched_cluster_rows(
            left_only_signatures, left_clusters, right_clusters
        ),
        "right_only_clusters": _unmatched_cluster_rows(
            right_only_signatures, right_clusters, left_clusters
        ),
    }
    # The overlap counts above are cluster-to-cluster counts, not image-set counts.
    # Keep the calculation explicit so the aggregate page can explain the distinction.
    comparison["clusters"]["left_only_with_overlap_count"] = sum(
        bool(_best_overlap(signature, right_clusters))
        for signature in left_only_signatures
    )
    comparison["clusters"]["right_only_with_overlap_count"] = sum(
        bool(_best_overlap(signature, left_clusters))
        for signature in right_only_signatures
    )
    return comparison


def _number(value: int | float) -> str:
    return f"{value:,}"


def _metric_card(label: str, value: str, detail: str = "") -> str:
    detail_markup = f"<p>{html.escape(detail)}</p>" if detail else ""
    return (
        "<article class='metric-card'>"
        f"<h2>{html.escape(label)}</h2><strong>{html.escape(value)}</strong>"
        f"{detail_markup}</article>"
    )


def _cluster_table(rows: list[dict[str, Any]], side_label: str) -> str:
    if not rows:
        return "<p>No clusters are unique to this run by exact membership.</p>"
    table_rows = []
    for row in rows:
        best = row["best_overlap"]
        if best is None:
            overlap = "none"
            overlap_count = "0"
            jaccard = "—"
        else:
            overlap = best["cluster_id"]
            overlap_count = _number(best["overlap_image_count"])
            jaccard = f"{best['jaccard']:.3f}"
        members = "<br>".join(html.escape(image_id) for image_id in row["image_ids"])
        table_rows.append(
            "<tr data-search='"
            + html.escape(
                " ".join(
                    [
                        side_label,
                        row["cluster_id"],
                        row["sequence_id"],
                        *row["image_ids"],
                    ]
                ),
                quote=True,
            )
            + "'>"
            f"<td><code>{html.escape(row['cluster_id'])}</code></td>"
            f"<td data-sort='{row['image_count']}'>{_number(row['image_count'])}</td>"
            f"<td>{html.escape(overlap)}</td>"
            f"<td>{overlap_count}</td><td>{jaccard}</td>"
            f"<td>{_number(row['overlapping_cluster_count'])}</td>"
            f"<td><code>{html.escape(row['sequence_id'])}</code></td>"
            f"<td><details><summary>members</summary><small>{members}</small></details></td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table class='cluster-table'><thead><tr>"
        f"<th>{html.escape(side_label)} cluster</th><th>Images</th>"
        "<th>Best overlap in other run</th><th>Overlap images</th><th>Jaccard</th>"
        "<th>Overlapping clusters</th><th>Sequence</th><th>Members</th>"
        "</tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></div>"
    )


def _size_table(left: dict[str, Any], right: dict[str, Any], left_label: str, right_label: str) -> str:
    sizes = sorted(
        {
            int(size)
            for distribution in (
                left["cluster_size_distribution"],
                right["cluster_size_distribution"],
            )
            for size in distribution
        }
    )
    rows = []
    for size in sizes:
        rows.append(
            "<tr>"
            f"<td>{_number(size)}</td>"
            f"<td>{_number(left['cluster_size_distribution'].get(str(size), 0))}</td>"
            f"<td>{_number(right['cluster_size_distribution'].get(str(size), 0))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Cluster size</th>"
        f"<th>{html.escape(left_label)}</th><th>{html.escape(right_label)}</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def write_comparison_page(comparison: dict[str, Any], output_path: Path) -> None:
    """Write a self-contained static aggregate comparison page."""
    output_path = output_path.expanduser().resolve()
    left = comparison["left"]
    right = comparison["right"]
    images = comparison["images"]
    clusters = comparison["clusters"]
    left_label = left["label"]
    right_label = right["label"]
    body = (
        "<nav class='page-nav'><a href='index.html' title='Return to the cluster review'>"
        "Cluster review</a></nav>"
        "<h1>Compare clustering runs</h1>"
        f"<p class='lede'>{html.escape(comparison['comparison_basis'])}</p>"
        "<section class='run-grid'>"
        f"<article class='run-card'><h2>{html.escape(left_label)}</h2>"
        f"<p><strong>Manifest:</strong> <code>{html.escape(left['manifest'])}</code></p>"
        f"<p><strong>Config:</strong> <code>{html.escape(left['config_fingerprint'])}</code>; "
        f"{_number(left['image_count'])} images, {_number(left['cluster_count'])} clusters, "
        f"{_number(left['sequence_count'])} sequences.</p></article>"
        f"<article class='run-card'><h2>{html.escape(right_label)}</h2>"
        f"<p><strong>Manifest:</strong> <code>{html.escape(right['manifest'])}</code></p>"
        f"<p><strong>Config:</strong> <code>{html.escape(right['config_fingerprint'])}</code>; "
        f"{_number(right['image_count'])} images, {_number(right['cluster_count'])} clusters, "
        f"{_number(right['sequence_count'])} sequences.</p></article></section>"
        "<h2>Aggregate summary</h2><section class='metric-grid'>"
        + _metric_card("Exact shared clusters", _number(clusters["exact_match_count"]), "Same image membership set in both runs")
        + _metric_card(f"Only in {left_label}", _number(clusters["left_only_count"]), "No exact membership match in the other run")
        + _metric_card(f"Only in {right_label}", _number(clusters["right_only_count"]), "No exact membership match in the other run")
        + _metric_card("Changed cluster membership", _number(images["changed_membership_count"]), "Common images belonging to non-matching clusters")
        + _metric_card("Common images", _number(images["common_count"]), f"{images['common_coverage']:.1%} of the union")
        + _metric_card(f"Images only in {left_label}", _number(images["left_only_count"]))
        + _metric_card(f"Images only in {right_label}", _number(images["right_only_count"]))
        + _metric_card("Exact-match cluster images", _number(images["exact_match_cluster_image_count"]), "Images contained in exact shared clusters")
        + "</section>"
        "<section class='warning'><strong>How to read this page:</strong> "
        "“Only in” means the complete membership set does not occur in the other run. "
        "A cluster can therefore appear in both only when all of its stable image IDs match; "
        "the overlap columns below show likely split/merge or membership changes.</section>"
        "<h2>Cluster-size distribution</h2>"
        + _size_table(left, right, left_label, right_label)
        + "<h2>Clusters unique to a run</h2>"
        "<label class='filter-label' for='cluster-filter'>Filter cluster rows by ID, sequence, or image ID</label>"
        "<input id='cluster-filter' class='filter' type='search' placeholder='e.g. cluster_00042 or image ID' "
        "title='Filter the unique-cluster tables'>"
        f"<details open><summary>{html.escape(left_label)} only ({_number(clusters['left_only_count'])})</summary>"
        f"{_cluster_table(comparison['left_only_clusters'], left_label)}</details>"
        f"<details open><summary>{html.escape(right_label)} only ({_number(clusters['right_only_count'])})</summary>"
        f"{_cluster_table(comparison['right_only_clusters'], right_label)}</details>"
        "<h2>Overlap summary</h2>"
        "<table><thead><tr><th>Measure</th><th>Value</th></tr></thead><tbody>"
        f"<tr><td>{html.escape(left_label)}-only clusters with any shared image</td><td>{_number(clusters['left_only_with_overlap_count'])}</td></tr>"
        f"<tr><td>{html.escape(right_label)}-only clusters with any shared image</td><td>{_number(clusters['right_only_with_overlap_count'])}</td></tr>"
        f"<tr><td>{html.escape(left_label)} images</td><td>{_number(images['left_count'])}</td></tr>"
        f"<tr><td>{html.escape(right_label)} images</td><td>{_number(images['right_count'])}</td></tr>"
        f"<tr><td>Union of image IDs</td><td>{_number(images['union_count'])}</td></tr>"
        "</tbody></table>"
        "<details><summary>Machine-readable comparison JSON</summary>"
        f"<pre>{html.escape(json.dumps(comparison, indent=2, sort_keys=True))}</pre></details>"
        "<script>"
        "const filter = document.querySelector('#cluster-filter');"
        "filter.addEventListener('input', () => {"
        "  const query = filter.value.trim().toLowerCase();"
        "  for (const row of document.querySelectorAll('.cluster-table tbody tr')) {"
        "    row.hidden = query !== '' && !(row.dataset.search || '').toLowerCase().includes(query);"
        "  }"
        "});"
        "</script>"
    )
    document = f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Compare clustering runs</title>
<style>
body{{font-family:system-ui,sans-serif;margin:1.5rem;background:#f7f7f7;color:#222;line-height:1.45}}
a{{color:#145a9c}}code{{overflow-wrap:anywhere}}h1{{margin-bottom:.3rem}}
.lede{{color:#46566a}}.page-nav{{margin-bottom:1rem}}.run-grid,.metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem}}
.run-card,.metric-card{{background:#fff;border:1px solid #bbc5d1;border-radius:.35rem;padding:1rem}}
.metric-card h2{{font-size:.95rem;margin:0 0 .35rem;color:#46566a}}.metric-card strong{{font-size:1.8rem}}
.metric-card p{{font-size:.85rem;color:#46566a;margin:.35rem 0 0}}.warning{{padding:.8rem;margin:1rem 0;background:#fff3cd;border:1px solid #e0b400}}
table{{border-collapse:collapse;background:#fff;width:100%;margin:1rem 0}}th,td{{border:1px solid #ccd4df;padding:.45rem;text-align:left;vertical-align:top}}th{{background:#e9eef5}}
.table-wrap{{overflow:auto;max-height:70vh;margin-top:.75rem}}.table-wrap table{{margin-top:0;min-width:900px}}
.filter-label{{display:block;font-weight:600;margin-top:.75rem}}.filter{{font:inherit;padding:.45rem;min-width:min(100%,28rem);margin:.35rem 0 1rem}}
.cluster-table tbody tr[hidden]{{display:none}}details{{margin:1rem 0}}summary{{cursor:pointer;font-weight:700;padding:.35rem 0}}
pre{{white-space:pre-wrap;max-height:32rem;overflow:auto;background:#111;color:#eee;padding:1rem}}
</style></head><body>{body}</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")


def default_comparison_output(left_path: Path) -> Path:
    """Return the conventional comparison page path inside the left run."""
    manifest = resolve_clustering_manifest(left_path)
    return _run_root(manifest) / "review" / "compare.html"
