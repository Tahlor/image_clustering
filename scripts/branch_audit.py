"""Inventory all remote branches and record their exact tree differences."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path


def parser(args: str | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Optional shell-style argument string.

    Returns:
        Parsed arguments.
    """
    command = argparse.ArgumentParser()
    command.add_argument("--output_dir", type=Path, required=True)
    if args:
        return command.parse_args(shlex.split(args))
    return command.parse_args()


def git(*args: str, check: bool = True) -> str:
    """Run Git and return stripped stdout."""
    result = subprocess.run(
        ["git", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def succeeds(*args: str) -> bool:
    """Return whether a Git command exits successfully."""
    return (
        subprocess.run(
            ["git", *args],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def audit(output_dir: Path) -> dict[str, object]:
    """Audit every origin branch against origin/master.

    Args:
        output_dir: Destination for JSON, summaries, and patches.

    Returns:
        The complete audit payload.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    patches_dir = output_dir / "patches"
    patches_dir.mkdir(parents=True, exist_ok=True)
    git("fetch", "--prune", "origin", "+refs/heads/*:refs/remotes/origin/*")

    branches: list[dict[str, object]] = []
    refs = git(
        "for-each-ref",
        "--format=%(refname:strip=3)",
        "refs/remotes/origin",
    ).splitlines()
    for branch in sorted(ref for ref in refs if ref and ref != "HEAD"):
        ref = f"origin/{branch}"
        head = git("rev-parse", ref)
        if branch == "master":
            branches.append({"name": branch, "head": head, "default": True})
            continue

        merge_base = git("merge-base", "origin/master", ref)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "__", branch)
        direct_patch = git("diff", "--binary", "origin/master", ref)
        merge_base_patch = git("diff", "--binary", f"{merge_base}..{ref}")
        (patches_dir / f"{safe_name}__tree-vs-master.patch").write_text(
            direct_patch,
            encoding="utf-8",
        )
        (patches_dir / f"{safe_name}__since-merge-base.patch").write_text(
            merge_base_patch,
            encoding="utf-8",
        )
        branches.append(
            {
                "name": branch,
                "head": head,
                "default": False,
                "merge_base": merge_base,
                "head_is_ancestor_of_master": succeeds(
                    "merge-base", "--is-ancestor", ref, "origin/master"
                ),
                "master_is_ancestor_of_head": succeeds(
                    "merge-base", "--is-ancestor", "origin/master", ref
                ),
                "tree_equal_to_master": not bool(direct_patch),
                "ahead_of_master": int(
                    git("rev-list", "--count", f"origin/master..{ref}") or 0
                ),
                "behind_master": int(
                    git("rev-list", "--count", f"{ref}..origin/master") or 0
                ),
                "unique_commits_vs_master": git(
                    "log",
                    "--format=%H%x09%aI%x09%an%x09%s",
                    f"origin/master..{ref}",
                ).splitlines(),
                "commits_since_merge_base": git(
                    "log",
                    "--format=%H%x09%aI%x09%an%x09%s",
                    f"{merge_base}..{ref}",
                ).splitlines(),
                "tree_diff_name_status": git(
                    "diff", "--name-status", "origin/master", ref
                ).splitlines(),
                "merge_base_diff_name_status": git(
                    "diff", "--name-status", f"{merge_base}..{ref}"
                ).splitlines(),
                "tree_diff_stat": git(
                    "diff", "--stat", "origin/master", ref
                ).splitlines(),
            }
        )

    payload: dict[str, object] = {
        "master": git("rev-parse", "origin/master"),
        "branches": branches,
    }
    (output_dir / "inventory.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [f"master: {payload['master']}", ""]
    for branch in branches:
        lines.append(f"BRANCH {branch['name']} @ {branch['head']}")
        if branch.get("default"):
            lines.append("  default branch")
        else:
            lines.append(
                "  "
                f"ahead={branch['ahead_of_master']} "
                f"behind={branch['behind_master']} "
                f"tree_equal={branch['tree_equal_to_master']} "
                f"head_ancestor={branch['head_is_ancestor_of_master']} "
                f"master_ancestor={branch['master_is_ancestor_of_head']}"
            )
            lines.append("  commits unique versus master:")
            lines.extend(
                f"    {commit}"
                for commit in branch["unique_commits_vs_master"]
            )
            lines.append("  tree diff versus master:")
            lines.extend(f"    {item}" for item in branch["tree_diff_name_status"])
        lines.append("")
    summary = "\n".join(lines)
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return payload


def main() -> int:
    """Run the branch audit."""
    args = parser()
    audit(output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
