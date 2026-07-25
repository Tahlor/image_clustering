"""Local review server: serves the labeling app and autosaves every decision.

The server binds to the loopback interface and has no authentication, because it
is a single-user local tool that reads one evaluation directory. Every state
change is written to disk before the response returns, so a browser crash or a
closed laptop cannot lose reviewer work.
"""

from __future__ import annotations

import json
import logging
import re
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from image_clustering.review.app import review_app_html
from image_clustering.review.dataset import ReviewDataset, dataset_payload
from image_clustering.review.decisions import (
    approve_bboxes,
    approve_cluster,
    dissolve_cluster,
    mark_irregular_cluster,
    mark_remaining_bboxes_ok,
    mark_remaining_clusters_ok,
    progress,
    reopen_cluster,
    restore_cluster,
    set_boxes,
    set_membership,
)
from image_clustering.review.exports import write_review_exports
from image_clustering.review.previews import (
    EDIT_MAX_DIMENSION,
    THUMBNAIL_MAX_DIMENSION,
    ensure_preview,
)
from image_clustering.review.store import DecisionStore

LOGGER = logging.getLogger(__name__)

CLUSTER_DECISION_PATTERN = re.compile(r"^/api/clusters/(?P<cluster_id>[^/]+)$")
RESTORE_PATTERN = re.compile(r"^/api/clusters/(?P<cluster_id>[^/]+)/restore$")
MEMBERSHIP_PATTERN = re.compile(
    r"^/api/clusters/(?P<cluster_id>[^/]+)/images/(?P<image_id>.+)/membership$"
)
BOXES_PATTERN = re.compile(
    r"^/api/clusters/(?P<cluster_id>[^/]+)/images/(?P<image_id>.+)/boxes$"
)
BBOX_STATUS_PATTERN = re.compile(
    r"^/api/clusters/(?P<cluster_id>[^/]+)/images/(?P<image_id>.+)/bbox-status$"
)


def path_parameter(value: str) -> str:
    """Decode one URL path parameter.

    Image identifiers contain slashes, and the client sends them percent-encoded,
    so path parameters must be decoded before they are looked up.
    """
    return unquote(value)


class ReviewServer(ThreadingHTTPServer):
    """Loopback server that refuses to share an already-bound port.

    The stdlib default allows a second socket to bind the same address, which on
    Windows silently leaves an older server answering every request. Binding
    exclusively makes a stale server an immediate, visible error instead.
    """

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        super().server_bind()


class ReviewService:
    """Decision operations shared by the HTTP layer and tests."""

    def __init__(self, dataset: ReviewDataset, store: DecisionStore) -> None:
        self.dataset = dataset
        self.store = store
        self._allowed_sources = {
            str(path.resolve()) for path in dataset.source_paths
        }

    @property
    def state(self) -> dict[str, Any]:
        """Return the live decision document."""
        return self.store.state

    def snapshot(self) -> dict[str, Any]:
        """Return decisions plus progress for the client."""
        return {
            "schema_version": 1,
            "clusters": self.state.get("clusters", {}),
            "progress": progress(self.state, self.dataset).to_dict(),
            "decisions_path": str(self.store.path),
        }

    def _saved(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.store.save()
        return {
            **payload,
            "progress": progress(self.state, self.dataset).to_dict(),
            "decisions_path": str(self.store.path),
        }

    def set_membership(
        self,
        cluster_id: str,
        image_id: str,
        included: bool,
    ) -> dict[str, Any]:
        """Include or exclude one member and persist immediately."""
        cluster = set_membership(
            self.state, self.dataset, cluster_id, image_id, included
        )
        return self._saved({"cluster": cluster})

    def approve_cluster(self, cluster_id: str) -> dict[str, Any]:
        """Confirm a grouping as correct."""
        return self._saved(
            {"cluster": approve_cluster(self.state, self.dataset, cluster_id)}
        )

    def dissolve_cluster(self, cluster_id: str) -> dict[str, Any]:
        """Reject a whole grouping."""
        return self._saved(
            {"cluster": dissolve_cluster(self.state, self.dataset, cluster_id)}
        )

    def mark_irregular(self, cluster_id: str) -> dict[str, Any]:
        """Classify a grouping as irregular and exclude it from exports."""
        return self._saved(
            {"cluster": mark_irregular_cluster(self.state, self.dataset, cluster_id)}
        )

    def reopen_cluster(self, cluster_id: str) -> dict[str, Any]:
        """Undo a cluster decision back to unreviewed."""
        return self._saved(
            {"cluster": reopen_cluster(self.state, self.dataset, cluster_id)}
        )

    def restore_cluster(
        self,
        cluster_id: str,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        """Restore a previous decision snapshot for undo."""
        return self._saved(
            {"cluster": restore_cluster(self.state, self.dataset, cluster_id, record)}
        )

    def set_boxes(
        self,
        cluster_id: str,
        image_id: str,
        boxes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Replace reviewer boxes for one image."""
        record = set_boxes(self.state, self.dataset, cluster_id, image_id, boxes)
        return self._saved({"image": record})

    def set_bbox_status(
        self,
        cluster_id: str,
        image_id: str,
        approved: bool,
    ) -> dict[str, Any]:
        """Approve or unapprove the box set for one image."""
        record = approve_bboxes(
            self.state, self.dataset, cluster_id, image_id, approved
        )
        return self._saved({"image": record})

    def mark_remaining_ok(
        self,
        scope: str,
        cluster_ids: list[str] | None,
    ) -> dict[str, Any]:
        """Approve everything still unreviewed in the requested scope."""
        if scope == "bboxes":
            changed = mark_remaining_bboxes_ok(self.state, self.dataset, cluster_ids)
        else:
            changed = mark_remaining_clusters_ok(self.state, self.dataset, cluster_ids)
        return self._saved({"scope": scope, "changed_count": len(changed)})

    def export(self) -> dict[str, Any]:
        """Write corrected manifests for the current decisions."""
        self.store.save()
        return write_review_exports(self.state, self.dataset)

    def import_decisions(self, decisions: dict[str, Any]) -> dict[str, Any]:
        """Replace all decisions from an exported document."""
        self.store.replace(decisions)
        return self.snapshot()

    def preview_path(self, source: str, size: str) -> Path | None:
        """Return a cached clean preview for an allowed source image."""
        resolved = str(Path(source).resolve())
        if resolved not in self._allowed_sources:
            raise PermissionError(f"Source image is not part of this run: {source}")
        max_dimension = (
            THUMBNAIL_MAX_DIMENSION if size == "thumbnail" else EDIT_MAX_DIMENSION
        )
        return ensure_preview(
            self.dataset.output_root, Path(resolved), max_dimension=max_dimension
        )


class ReviewRequestHandler(BaseHTTPRequestHandler):
    """Minimal JSON/HTML request handler for the review app."""

    server_version = "ImageClusteringReview/1.0"
    service: ReviewService

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.debug("%s %s", self.address_string(), format % args)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                self._send_bytes(
                    review_app_html().encode("utf-8"), "text/html; charset=utf-8"
                )
                return
            if parsed.path == "/api/dataset":
                self._send_json(dataset_payload(self.service.dataset))
                return
            if parsed.path == "/api/decisions":
                self._send_json(self.service.snapshot())
                return
            if parsed.path == "/preview":
                query = parse_qs(parsed.query)
                source = (query.get("path") or [""])[0]
                size = (query.get("size") or ["edit"])[0]
                preview = self.service.preview_path(source, size)
                if preview is None:
                    self._send_json({"error": "preview unavailable"}, status=404)
                    return
                self._send_bytes(preview.read_bytes(), "image/jpeg")
                return
            self._send_json({"error": "not found"}, status=404)
        except PermissionError as error:
            self._send_json({"error": str(error)}, status=403)
        except (KeyError, FileNotFoundError) as error:
            self._send_json({"error": str(error)}, status=404)
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.exception("Unhandled GET error")
            self._send_json({"error": str(error)}, status=500)

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        try:
            payload = self._read_json()
            membership = MEMBERSHIP_PATTERN.match(parsed.path)
            boxes = BOXES_PATTERN.match(parsed.path)
            bbox_status = BBOX_STATUS_PATTERN.match(parsed.path)
            restore = RESTORE_PATTERN.match(parsed.path)
            decision = CLUSTER_DECISION_PATTERN.match(parsed.path)
            if restore:
                self._send_json(
                    self.service.restore_cluster(
                        path_parameter(restore["cluster_id"]),
                        dict(payload.get("cluster") or {}),
                    )
                )
                return
            if membership:
                self._send_json(
                    self.service.set_membership(
                        path_parameter(membership["cluster_id"]),
                        path_parameter(membership["image_id"]),
                        bool(payload.get("included")),
                    )
                )
                return
            if boxes:
                self._send_json(
                    self.service.set_boxes(
                        path_parameter(boxes["cluster_id"]),
                        path_parameter(boxes["image_id"]),
                        list(payload.get("boxes") or []),
                    )
                )
                return
            if bbox_status:
                self._send_json(
                    self.service.set_bbox_status(
                        path_parameter(bbox_status["cluster_id"]),
                        path_parameter(bbox_status["image_id"]),
                        bool(payload.get("approved", True)),
                    )
                )
                return
            if decision:
                action = str(payload.get("action") or "")
                cluster_id = path_parameter(decision["cluster_id"])
                if action == "approve":
                    self._send_json(self.service.approve_cluster(cluster_id))
                    return
                if action == "dissolve":
                    self._send_json(self.service.dissolve_cluster(cluster_id))
                    return
                if action == "irregular":
                    self._send_json(self.service.mark_irregular(cluster_id))
                    return
                if action == "reopen":
                    self._send_json(self.service.reopen_cluster(cluster_id))
                    return
                self._send_json({"error": f"unknown action: {action}"}, status=400)
                return
            if parsed.path == "/api/mark-remaining-ok":
                self._send_json(
                    self.service.mark_remaining_ok(
                        str(payload.get("scope") or "clusters"),
                        payload.get("cluster_ids"),
                    )
                )
                return
            if parsed.path == "/api/export":
                self._send_json(self.service.export())
                return
            if parsed.path == "/api/import":
                self._send_json(
                    self.service.import_decisions(dict(payload.get("clusters") or {}))
                )
                return
            self._send_json({"error": "not found"}, status=404)
        except (KeyError, FileNotFoundError) as error:
            self._send_json({"error": str(error)}, status=404)
        except ValueError as error:
            self._send_json({"error": str(error)}, status=400)
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.exception("Unhandled POST error")
            self._send_json({"error": str(error)}, status=500)


def build_server(
    dataset: ReviewDataset,
    store: DecisionStore,
    host: str = "127.0.0.1",
    port: int = 8756,
) -> ThreadingHTTPServer:
    """Create the loopback review server bound to an available port."""
    service = ReviewService(dataset, store)
    handler = type(
        "BoundReviewRequestHandler",
        (ReviewRequestHandler,),
        {"service": service},
    )
    try:
        return ReviewServer((host, port), handler)
    except OSError as error:
        raise OSError(
            f"Could not bind {host}:{port} ({error}). Another review server is "
            "probably still running; stop it, or pass --port 0 for a free port."
        ) from error


def serve(
    output_root: Path,
    host: str = "127.0.0.1",
    port: int = 8756,
) -> None:
    """Run the review server until interrupted."""
    from image_clustering.review.dataset import build_review_dataset

    dataset = build_review_dataset(Path(output_root))
    store = DecisionStore.for_output_root(dataset.output_root, dataset.provenance)
    store.save()
    server = build_server(dataset, store, host=host, port=port)
    address = f"http://{host}:{server.server_address[1]}/"
    LOGGER.info("Review %d clusters at %s", len(dataset.clusters), address)
    LOGGER.info("Decisions autosave to %s", store.path)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        LOGGER.info("Stopping; decisions are saved at %s", store.path)
    finally:
        server.server_close()
