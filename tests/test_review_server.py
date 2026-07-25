import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from image_clustering import write_result
from image_clustering.clustering.models import (
    ClusteringResult,
    ImageCluster,
    ImageItem,
    PairComparison,
)
from image_clustering.review.dataset import build_review_dataset
from image_clustering.review.decisions import empty_state, mark_irregular_cluster
from image_clustering.review.exports import write_review_exports
from image_clustering.review.server import build_server
from image_clustering.review.store import DecisionStore


def build_run(tmp_path: Path) -> tuple[Path, Path]:
    input_root = tmp_path / "input"
    folder = input_root / "sequence-a"
    folder.mkdir(parents=True)
    paths = []
    for index in range(2):
        path = folder / f"page-{index:03d}.png"
        cv2.imwrite(str(path), np.full((80, 100, 3), 200, dtype=np.uint8))
        paths.append(path)
    images = tuple(
        ImageItem(f"sequence-a/{path.name}", path, "sequence-a", index)
        for index, path in enumerate(paths)
    )
    result = ClusteringResult(
        config_fingerprint="fingerprint",
        input_root=input_root,
        images=images,
        clusters=(
            ImageCluster(
                "cluster_00001",
                "sequence-a",
                tuple(image.image_id for image in images),
                images[0].image_id,
            ),
        ),
        comparisons=(
            PairComparison(
                images[0].image_id,
                images[1].image_id,
                "sequence-a",
                1,
                True,
                0.62,
                "test",
            ),
        ),
    )
    output_root = tmp_path / "output"
    write_result(result, output_root / "clustering")
    inventory = {
        "images": [
            {
                "absolute_source_path": str(path.resolve()),
                "width": 100,
                "height": 80,
            }
            for path in paths
        ]
    }
    inventory_path = output_root / "inventory" / "dataset_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    crops = output_root / "reports" / "crops_for_recognizer.jsonl"
    crops.parent.mkdir(parents=True, exist_ok=True)
    crops.write_text(
        json.dumps(
            {
                "submission_id": "submission_00001",
                "cluster_id": "cluster_00001",
                "source_image_path": str(paths[0].resolve()),
                "crop_path": None,
                "bbox": [5, 6, 60, 70],
                "kind": "base_page",
                "completeness": "complete",
                "confidence": 0.8,
                "review_required": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return input_root, output_root


class RunningServer:
    def __init__(self, output_root: Path) -> None:
        self.dataset = build_review_dataset(output_root)
        self.store = DecisionStore.for_output_root(
            self.dataset.output_root, self.dataset.provenance
        )
        self.server = build_server(self.dataset, self.store, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def get(self, path: str) -> tuple[int, bytes]:
        with urllib.request.urlopen(f"{self.base}{path}") as response:
            return response.status, response.read()

    def post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())


def test_server_serves_app_dataset_and_previews(tmp_path: Path) -> None:
    _, output_root = build_run(tmp_path)
    running = RunningServer(output_root)
    try:
        status, body = running.get("/")
        assert status == 200
        page = body.decode("utf-8")
        assert "Cluster review" in page
        assert "Mark remaining clusters OK" in page
        assert "Mark this grouping irregular" in page
        for key in ("j", "k", "a", "d", "i", "e", "b", "x"):
            assert f"<kbd>{key}</kbd>" in page

        status, body = running.get("/api/dataset")
        dataset = json.loads(body)
        assert dataset["defaults"] == {
            "minimum_cluster_size": 2,
            "sort": "confidence-asc",
        }
        cluster = dataset["clusters"][0]
        assert cluster["image_count"] == 2
        assert cluster["images"][0]["boxes"][0]["bbox"] == [5, 6, 60, 70]

        source = cluster["images"][0]["source_path"]
        status, body = running.get(
            f"/preview?size=thumbnail&path={urllib.request.quote(source)}"
        )
        assert status == 200
        assert body[:2] == b"\xff\xd8"
    finally:
        running.close()


def test_membership_change_autosaves_and_dissolves_a_pair(tmp_path: Path) -> None:
    _, output_root = build_run(tmp_path)
    running = RunningServer(output_root)
    try:
        dataset = json.loads(running.get("/api/dataset")[1])
        cluster_id = dataset["clusters"][0]["cluster_id"]
        image_id = dataset["clusters"][0]["images"][0]["image_id"]

        payload = running.post(
            f"/api/clusters/{cluster_id}/images/{urllib.request.quote(image_id)}/membership",
            {"included": False},
        )

        assert payload["cluster"]["status"] == "dissolved"
        assert payload["progress"]["reviewed_cluster_count"] == 1
        saved = json.loads(Path(payload["decisions_path"]).read_text(encoding="utf-8"))
        assert saved["clusters"][cluster_id]["dissolved"] is True
    finally:
        running.close()


def test_irregular_action_persists_and_counts_as_reviewed(tmp_path: Path) -> None:
    _, output_root = build_run(tmp_path)
    running = RunningServer(output_root)
    try:
        payload = running.post(
            "/api/clusters/cluster_00001",
            {"action": "irregular"},
        )

        assert payload["cluster"]["status"] == "irregular"
        assert payload["cluster"]["dissolved"] is False
        assert payload["progress"]["irregular_cluster_count"] == 1
        saved = json.loads(Path(payload["decisions_path"]).read_text(encoding="utf-8"))
        assert saved["clusters"]["cluster_00001"]["status"] == "irregular"
    finally:
        running.close()


def test_box_edit_persists_and_rejects_out_of_bounds_boxes(tmp_path: Path) -> None:
    _, output_root = build_run(tmp_path)
    running = RunningServer(output_root)
    try:
        dataset = json.loads(running.get("/api/dataset")[1])
        cluster_id = dataset["clusters"][0]["cluster_id"]
        image_id = urllib.request.quote(dataset["clusters"][0]["images"][0]["image_id"])

        payload = running.post(
            f"/api/clusters/{cluster_id}/images/{image_id}/boxes",
            {"boxes": [{"bbox": [1, 2, 40, 50], "kind": "reviewer"}]},
        )
        assert payload["image"]["bbox_status"] == "edited"
        assert payload["image"]["boxes"][0]["bbox"] == [1, 2, 40, 50]

        try:
            running.post(
                f"/api/clusters/{cluster_id}/images/{image_id}/boxes",
                {"boxes": [{"bbox": [0, 0, 500, 50]}]},
            )
            raise AssertionError("expected an error for an out-of-bounds box")
        except urllib.error.HTTPError as error:
            assert error.code == 400
    finally:
        running.close()


def test_preview_rejects_paths_outside_the_run(tmp_path: Path) -> None:
    _, output_root = build_run(tmp_path)
    outside = tmp_path / "outside.png"
    cv2.imwrite(str(outside), np.zeros((10, 10, 3), dtype=np.uint8))
    running = RunningServer(output_root)
    try:
        running.get(f"/preview?path={urllib.request.quote(str(outside))}")
        raise AssertionError("expected a rejection for a path outside the run")
    except urllib.error.HTTPError as error:
        assert error.code == 403
    finally:
        running.close()


def test_export_keeps_every_image_after_a_dissolve(tmp_path: Path) -> None:
    _, output_root = build_run(tmp_path)
    dataset = build_review_dataset(output_root)
    state = empty_state(dataset.provenance)
    cluster = dataset.clusters[0]
    from image_clustering.review.decisions import set_membership

    set_membership(
        state, dataset, cluster.cluster_id, cluster.images[0].image_id, included=False
    )

    manifest = write_review_exports(state, dataset)

    clusters_path = Path(manifest["artifacts"]["clusters_reviewed"])
    records = [
        json.loads(line)
        for line in clusters_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert manifest["final_image_count"] == 2
    assert all(record["image_count"] == 1 for record in records)
    assert {record["origin_cluster_id"] for record in records} == {cluster.cluster_id}
    boxes = [
        json.loads(line)
        for line in Path(manifest["artifacts"]["crops_reviewed"])
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(boxes) == 2
    assert boxes[0]["boxes"][0]["bbox"] == [5, 6, 60, 70]
    assert Path(manifest["artifacts"]["review_summary"]).is_file()


def test_export_excludes_irregular_clusters_from_training_and_validation(
    tmp_path: Path,
) -> None:
    _, output_root = build_run(tmp_path)
    dataset = build_review_dataset(output_root)
    state = empty_state(dataset.provenance)
    cluster = dataset.clusters[0]

    mark_irregular_cluster(state, dataset, cluster.cluster_id)
    manifest = write_review_exports(state, dataset)

    clusters_path = Path(manifest["artifacts"]["clusters_reviewed"])
    boxes_path = Path(manifest["artifacts"]["crops_reviewed"])
    exclusions_path = Path(
        manifest["artifacts"]["excluded_from_training_validation"]
    )
    assert clusters_path.read_text(encoding="utf-8") == ""
    assert boxes_path.read_text(encoding="utf-8") == ""
    exclusions = [
        json.loads(line)
        for line in exclusions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(exclusions) == 1
    assert exclusions[0]["cluster_id"] == cluster.cluster_id
    assert exclusions[0]["image_count"] == cluster.image_count
    assert manifest["final_image_count"] == 0
    assert manifest["excluded_from_training_validation"] == {
        "cluster_count": 1,
        "image_count": cluster.image_count,
    }


def test_app_exposes_both_view_modes_and_edit_affordances(tmp_path: Path) -> None:
    _, output_root = build_run(tmp_path)
    running = RunningServer(output_root)
    try:
        page = running.get("/")[1].decode("utf-8")
        assert "Side by side" in page
        assert "id='view-mode'" in page
        assert "side by side ↔ full screen" in page
        assert "focus-mode" in page
        assert "id='mark-irregular'" in page
        assert "value='irregular'" in page
        assert "exclude from training/validation" in page
        assert "clusterAction('irregular')" in page
        assert "id='draft-banner'" in page
        assert ".draft-banner[hidden]{display:flex !important" in page
        assert "height:3.4rem;overflow:auto" in page
        assert "large || isFocused ? 'edit' : 'thumbnail'" not in page
        assert "previewUrl(image, 'edit')" in page
        assert "image-canvas" in page
        assert ".focus-mode .image-canvas{width:fit-content;max-width:100%}" in page
        assert "renderBoxes(canvas, image, boxes, editable)" in page
        assert "renderBoxes(canvas, image, draft.boxes, true)" in page
        assert "Save boxes" in page
        assert "Releasing the mouse saves automatically." in page
        assert "id='undo'" in page
        for key in ("v", "i", "Tab", "Enter", "Esc", "Ctrl", "Z"):
            assert f"<kbd>{key}</kbd>" in page
        for edge in ("nw", "ne", "sw", "se", "n]", "s]", "w]", "e]"):
            assert edge in page
        assert "data-edge=" in page
    finally:
        running.close()


def test_restore_endpoint_undoes_a_saved_change(tmp_path: Path) -> None:
    _, output_root = build_run(tmp_path)
    running = RunningServer(output_root)
    try:
        dataset = json.loads(running.get("/api/dataset")[1])
        cluster_id = dataset["clusters"][0]["cluster_id"]
        image_id = dataset["clusters"][0]["images"][0]["image_id"]
        snapshot = json.loads(running.get("/api/decisions")[1])["clusters"].get(
            cluster_id,
            {
                "status": "unreviewed",
                "dissolved": False,
                "excluded_image_ids": [],
                "images": {},
            },
        )

        dissolved = running.post(
            f"/api/clusters/{cluster_id}/images/{urllib.request.quote(image_id)}/membership",
            {"included": False},
        )
        assert dissolved["cluster"]["status"] == "dissolved"

        restored = running.post(
            f"/api/clusters/{cluster_id}/restore", {"cluster": snapshot}
        )

        assert restored["cluster"]["status"] == "unreviewed"
        assert restored["cluster"]["dissolved"] is False
        assert restored["progress"]["reviewed_cluster_count"] == 0
        saved = json.loads(Path(restored["decisions_path"]).read_text(encoding="utf-8"))
        assert saved["clusters"][cluster_id]["excluded_image_ids"] == []
    finally:
        running.close()


def test_restore_endpoint_rejects_invalid_geometry(tmp_path: Path) -> None:
    _, output_root = build_run(tmp_path)
    running = RunningServer(output_root)
    try:
        dataset = json.loads(running.get("/api/dataset")[1])
        cluster_id = dataset["clusters"][0]["cluster_id"]
        image_id = dataset["clusters"][0]["images"][0]["image_id"]
        running.post(
            f"/api/clusters/{cluster_id}/restore",
            {
                "cluster": {
                    "status": "edited",
                    "images": {image_id: {"boxes": [{"bbox": [0, 0, 4000, 10]}]}},
                }
            },
        )
        raise AssertionError("expected a rejection for an out-of-bounds restore")
    except urllib.error.HTTPError as error:
        assert error.code == 400
    finally:
        running.close()


def test_percent_encoded_image_ids_are_accepted(tmp_path: Path) -> None:
    """The browser encodes slashes in image ids as %2F; the server must decode them."""
    _, output_root = build_run(tmp_path)
    running = RunningServer(output_root)
    try:
        dataset = json.loads(running.get("/api/dataset")[1])
        cluster_id = dataset["clusters"][0]["cluster_id"]
        image_id = dataset["clusters"][0]["images"][0]["image_id"]
        assert "/" in image_id
        encoded = urllib.parse.quote(image_id, safe="")
        assert "%2F" in encoded

        boxes = running.post(
            f"/api/clusters/{cluster_id}/images/{encoded}/boxes",
            {"boxes": [{"bbox": [2, 3, 44, 55], "kind": "reviewer"}]},
        )
        assert boxes["image"]["boxes"][0]["bbox"] == [2, 3, 44, 55]

        status = running.post(
            f"/api/clusters/{cluster_id}/images/{encoded}/bbox-status",
            {"approved": True},
        )
        assert status["image"]["bbox_status"] == "approved"

        membership = running.post(
            f"/api/clusters/{cluster_id}/images/{encoded}/membership",
            {"included": False},
        )
        assert membership["cluster"]["status"] == "dissolved"
    finally:
        running.close()


def test_nested_image_ids_survive_encoding(tmp_path: Path) -> None:
    """Deeply nested source folders produce multi-slash image ids."""
    input_root = tmp_path / "input"
    folder = input_root / "triplet" / "54ecf012-c0ff" / "143"
    folder.mkdir(parents=True)
    paths = []
    for index in range(2):
        path = folder / f"i407166{index}-00385.png"
        cv2.imwrite(str(path), np.full((60, 90, 3), 180, dtype=np.uint8))
        paths.append(path)
    sequence_id = "triplet/54ecf012-c0ff/143"
    images = tuple(
        ImageItem(f"{sequence_id}/{path.name}", path, sequence_id, index)
        for index, path in enumerate(paths)
    )
    result = ClusteringResult(
        config_fingerprint="fingerprint",
        input_root=input_root,
        images=images,
        clusters=(
            ImageCluster(
                "cluster_00400",
                sequence_id,
                tuple(image.image_id for image in images),
                images[0].image_id,
            ),
        ),
        comparisons=(
            PairComparison(
                images[0].image_id, images[1].image_id, sequence_id, 1, True, 0.5, "t"
            ),
        ),
    )
    output_root = tmp_path / "output"
    write_result(result, output_root / "clustering")
    inventory_path = output_root / "inventory" / "dataset_inventory.json"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "absolute_source_path": str(path.resolve()),
                        "width": 90,
                        "height": 60,
                    }
                    for path in paths
                ]
            }
        ),
        encoding="utf-8",
    )

    running = RunningServer(output_root)
    try:
        encoded = urllib.parse.quote(images[0].image_id, safe="")
        assert encoded.count("%2F") == 3
        payload = running.post(
            f"/api/clusters/cluster_00400/images/{encoded}/boxes",
            {"boxes": [{"bbox": [1, 1, 80, 50]}]},
        )
        assert payload["image"]["boxes"][0]["bbox"] == [1, 1, 80, 50]
    finally:
        running.close()


def test_server_refuses_to_share_a_bound_port(tmp_path: Path) -> None:
    """A stale server must be a visible error, not a silent shadow bind."""
    _, output_root = build_run(tmp_path)
    running = RunningServer(output_root)
    try:
        port = running.server.server_address[1]
        dataset = build_review_dataset(output_root)
        store = DecisionStore.for_output_root(output_root, dataset.provenance)
        try:
            build_server(dataset, store, port=port)
            raise AssertionError("expected a bind failure on an occupied port")
        except OSError as error:
            assert "Another review server" in str(error)
    finally:
        running.close()
