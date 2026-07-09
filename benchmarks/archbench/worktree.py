"""Disposable git worktrees for ARCH-bench.

The harness applies flaw/fix patches and runs smoke training — real
subprocesses — but ONLY inside a throwaway worktree created under the system
temp dir. It NEVER touches the live checkout. The live-repo guard is
enforced here and asserted by tests/test_archbench_live_repo_guard.py:
- worktrees are created under tempfile.mkdtemp(), never inside the repo;
- every path operation is confined to the worktree;
- cleanup removes the worktree via `git worktree remove` + rmtree.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class LiveRepoError(RuntimeError):
    """An operation would touch the live checkout instead of a worktree."""


def repo_root(start: str | Path | None = None) -> Path:
    """The live repository's top level (git rev-parse), resolved."""
    out = subprocess.run(
        ["git", "-C", str(start or Path.cwd()), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, check=True)
    return Path(out.stdout.strip()).resolve()


def _assert_disposable(path: Path, live_root: Path) -> None:
    """A worktree path must be outside the live checkout AND under a temp
    dir — otherwise refuse before any git/subprocess touches it."""
    path = path.resolve()
    if path == live_root or live_root in path.parents:
        raise LiveRepoError(
            f"refusing to use {path} — it is inside the live checkout {live_root}")
    tmp_root = Path(tempfile.gettempdir()).resolve()
    if tmp_root not in path.parents and path != tmp_root:
        raise LiveRepoError(
            f"refusing to use {path} — worktrees must live under {tmp_root}")


class Worktree:
    """A disposable git worktree at ``base`` ref. Context-managed: created
    under the temp dir on enter, removed on exit."""

    def __init__(self, base: str = "HEAD", live_root: Path | None = None) -> None:
        self.base = base
        self.live_root = (live_root or repo_root()).resolve()
        self._parent = Path(tempfile.mkdtemp(prefix="archbench-"))
        self.path = (self._parent / "wt").resolve()
        _assert_disposable(self.path, self.live_root)

    def __enter__(self) -> "Worktree":
        self._git(["worktree", "add", "--detach", str(self.path), self.base],
                  cwd=self.live_root)
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def _git(self, args: list[str], cwd: Path | None = None) -> str:
        out = subprocess.run(["git", *args], cwd=str(cwd or self.path),
                             capture_output=True, text=True, check=True)
        return out.stdout

    def apply_diff(self, diff_text: str) -> None:
        """Apply a unified diff INSIDE the worktree (never the live repo)."""
        if not diff_text.strip():
            return
        _assert_disposable(self.path, self.live_root)  # re-check before writing
        patch = self.path / ".archbench.patch"
        patch.write_text(diff_text, encoding="utf-8")
        try:
            self._git(["apply", "--whitespace=nowarn", str(patch)])
        finally:
            patch.unlink(missing_ok=True)

    def apply_subs(self, subs: list[dict[str, str]]) -> str:
        """Apply find/replace ops inside the worktree; return a synthesized
        unified diff of what changed (so the record is still a diff). Each
        op: {file, find, replace}; ``find`` must occur exactly once."""
        import difflib

        _assert_disposable(self.path, self.live_root)
        diff_chunks: list[str] = []
        for op in subs:
            rel = op["file"]
            target = (self.path / rel).resolve()
            if self.path not in target.parents:
                raise LiveRepoError(f"sub targets {target} outside the worktree")
            before = target.read_text(encoding="utf-8")
            count = before.count(op["find"])
            if count != 1:
                raise ValueError(
                    f"{rel}: find text occurs {count}x (need exactly 1): "
                    f"{op['find']!r}")
            after = before.replace(op["find"], op["replace"])
            target.write_text(after, encoding="utf-8")
            diff_chunks.append("".join(difflib.unified_diff(
                before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile=f"a/{rel}", tofile=f"b/{rel}")))
        return "\n".join(diff_chunks)

    def cleanup(self) -> None:
        try:
            self._git(["worktree", "remove", "--force", str(self.path)],
                      cwd=self.live_root)
        except (subprocess.CalledProcessError, OSError):
            pass
        shutil.rmtree(self._parent, ignore_errors=True)
