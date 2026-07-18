"""Command-line interface for document-view clustering."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from image_clustering.clustering import (
    ClusterConfig,
    cluster_directory,
    write_result,
)


def parse_args(arg_str: str | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        arg_str: Optional shell-style argument string for programmatic use.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Cluster nearby document images that show the same physical "
            "document scene under changing occlusions."
        )
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no_cache", action="store_true")
    if arg_str:
        return parser.parse_args(shlex.split(arg_str))
    return parser.parse_args()


def main(arg_str: str | None = None) -> int:
    """Run directory clustering and write manifests."""
    args = parse_args(arg_str=arg_str)
    config = ClusterConfig.from_json(path=args.config)
    cache_dir = None if args.no_cache else args.output_dir / ".feature_cache"
    result = cluster_directory(
        input_dir=args.input_dir,
        config=config,
        cache_dir=cache_dir,
        show_progress=True,
    )
    write_result(result=result, output_dir=args.output_dir, config=config)
    accepted = sum(
        comparison.same_document for comparison in result.comparisons
    )
    print(
        f"Wrote {len(result.clusters)} clusters and {accepted} accepted "
        f"pair registrations to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
