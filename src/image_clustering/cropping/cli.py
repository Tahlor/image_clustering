"""Command-line interface for sequence-level crop recovery."""

from __future__ import annotations

import argparse
import logging
import shlex
from pathlib import Path
from typing import Sequence

from image_clustering import (
    ClusterConfig,
    cluster_directory,
    load_result,
    write_result,
)

from .api import crop_clustering_result
from .config import load_default_config


def parser(args: str | Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Optional shell-style argument string or token sequence.

    Returns:
        Parsed arguments.
    """
    command = argparse.ArgumentParser(
        description=(
            "Recover each unique complete page or foreground sheet once from "
            "clusters of related archival captures."
        )
    )
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_dir", type=Path)
    source.add_argument("--clustering_json", type=Path)
    command.add_argument("--output_dir", type=Path, required=True)
    command.add_argument("--cluster_config", type=Path)
    command.add_argument("--crop_config", type=Path)
    command.add_argument("--cache_dir", type=Path)
    command.add_argument("--show_progress", action="store_true")
    command.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    if isinstance(args, str):
        return command.parse_args(shlex.split(args))
    if args is not None:
        return command.parse_args(list(args))
    return command.parse_args()


def run(argv: str | Sequence[str] | None = None) -> int:
    """Run clustering when needed, then recover sequence-level submissions."""
    args = parser(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    crop_config = load_default_config(args.crop_config)
    if args.clustering_json is not None:
        clustering = load_result(path=args.clustering_json)
    else:
        cluster_config = ClusterConfig.from_json(args.cluster_config)
        clustering = cluster_directory(
            input_dir=args.input_dir,
            config=cluster_config,
            cache_dir=args.cache_dir,
            show_progress=args.show_progress,
        )
        write_result(
            result=clustering,
            output_dir=args.output_dir / "clustering",
        )
    crop_clustering_result(
        clustering=clustering,
        output_dir=args.output_dir,
        config=crop_config,
        show_progress=args.show_progress,
    )
    return 0


def main() -> None:
    """CLI entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
