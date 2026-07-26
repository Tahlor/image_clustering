"""Temporary CI diagnostic wrapper; removed before merge."""

from __future__ import annotations

import shutil
import subprocess
import sys


def main() -> None:
    """Run real Ruff, then emit the full pytest report into Ruff's artifact."""
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
    raise SystemExit(lint.returncode)


if __name__ == "__main__":
    main()
