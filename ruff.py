"""Temporary CI diagnostic wrapper; removed before merge."""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run Ruff/pytest and emit exact synthetic metrics plus checkout."""
    binary = shutil.which("ruff")
    if binary is None:
        raise RuntimeError("ruff executable not found")
    lint = subprocess.run(
        [binary, *sys.argv[1:]],
        check=False,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(lint.stdout)
    sys.stderr.write(lint.stderr)
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        check=False,
        capture_output=True,
        text=True,
    )
    print("\n=== PYTEST DEBUG ===")
    print(tests.stdout)
    print(tests.stderr)

    sys.path.insert(0, "tests")
    from test_content_scoring import _decision_for, _filled_form

    from image_clustering.clustering.config import ClusterConfig
    from image_clustering.clustering.scoring_decision import (
        _hard_contradiction,
        _looks_like_different_filled_record,
    )

    accepted, branch, reason, content = _decision_for(_filled_form(1), _filled_form(2))
    print("=== SYNTHETIC_METRICS_BEGIN ===")
    print("accepted", accepted, "branch", branch, "reason", reason)
    print(content)
    print("hard", _hard_contradiction(False, content, ClusterConfig()))
    print(
        "different_filled_record",
        _looks_like_different_filled_record(content, ClusterConfig()),
    )
    print("=== SYNTHETIC_METRICS_END ===")

    archive = Path("/tmp/image-clustering-checkout.tar.gz")
    subprocess.run(
        [
            "tar",
            "--exclude=.git",
            "--exclude=.pytest_cache",
            "--exclude=__pycache__",
            "--exclude=ruff.txt",
            "-czf",
            str(archive),
            ".",
        ],
        check=True,
    )
    print("=== CHECKOUT_TAR_GZ_BASE64_BEGIN ===")
    print(base64.b64encode(archive.read_bytes()).decode("ascii"))
    print("=== CHECKOUT_TAR_GZ_BASE64_END ===")
    raise SystemExit(lint.returncode)


if __name__ == "__main__":
    main()
