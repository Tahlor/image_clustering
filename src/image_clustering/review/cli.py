"""Command-line interface for the local cluster and crop review server."""

from __future__ import annotations

import argparse
import logging
import shlex
import webbrowser
from collections.abc import Sequence
from pathlib import Path

from image_clustering.review.dataset import build_review_dataset
from image_clustering.review.server import build_server
from image_clustering.review.store import DecisionStore

LOGGER = logging.getLogger(__name__)


def parser(args: str | Sequence[str] | None = None) -> argparse.Namespace:
    """Parse review-server arguments.

    Args:
        args: Optional shell-style argument string or token sequence.

    Returns:
        Parsed arguments.
    """
    command = argparse.ArgumentParser(
        description="Review cluster membership and proposed crop boxes locally."
    )
    command.add_argument("--output_dir", type=Path, required=True)
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8756)
    command.add_argument("--no_browser", action="store_true")
    if isinstance(args, str):
        return command.parse_args(shlex.split(args))
    if args is not None:
        return command.parse_args(list(args))
    return command.parse_args()


def run(args: str | Sequence[str] | None = None) -> int:
    """Run the review server until interrupted."""
    options = parser(args)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    dataset = build_review_dataset(options.output_dir.resolve())
    store = DecisionStore.for_output_root(dataset.output_root, dataset.provenance)
    store.save()
    server = build_server(dataset, store, host=options.host, port=options.port)
    url = f"http://{options.host}:{server.server_address[1]}/"
    reviewable = sum(1 for cluster in dataset.clusters if cluster.image_count >= 2)
    LOGGER.info(
        "Review %d clusters (%d with 2+ captures) at %s",
        len(dataset.clusters),
        reviewable,
        url,
    )
    LOGGER.info("Decisions autosave to %s", store.path)
    if not options.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        LOGGER.info("Stopped. Decisions are saved at %s", store.path)
    finally:
        server.server_close()
    return 0


def main() -> None:
    """Console-script entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
