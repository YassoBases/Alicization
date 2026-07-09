"""Stage-D6 STRUCTURAL GUARANTEE: ARCH-bench never touches the live checkout.
Worktrees live under the temp dir; any path at/inside the live repo, or
outside the temp root, is refused before git/subprocess runs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmarks.archbench.worktree import (
    LiveRepoError,
    Worktree,
    _assert_disposable,
    repo_root,
)

ROOT = Path(__file__).resolve().parent.parent


def test_repo_root_is_the_live_checkout() -> None:
    assert repo_root(ROOT) == ROOT.resolve()


def test_assert_disposable_refuses_the_live_repo() -> None:
    live = ROOT.resolve()
    with pytest.raises(LiveRepoError):
        _assert_disposable(live, live)                 # the repo itself
    with pytest.raises(LiveRepoError):
        _assert_disposable(live / "runs" / "x", live)  # inside the repo
    with pytest.raises(LiveRepoError):
        _assert_disposable(Path.home() / "elsewhere", live)  # not under temp


def test_assert_disposable_allows_temp_paths() -> None:
    live = ROOT.resolve()
    tmp = Path(tempfile.gettempdir()).resolve() / "archbench-xyz" / "wt"
    _assert_disposable(tmp, live)  # no raise


def test_worktree_path_is_under_tempdir_not_the_repo() -> None:
    # Construction alone (no git) must already place the path under temp and
    # pass the guard; it must not be inside the live checkout.
    wt = Worktree(base="HEAD", live_root=ROOT.resolve())
    try:
        assert Path(tempfile.gettempdir()).resolve() in wt.path.parents
        assert ROOT.resolve() not in wt.path.parents
        assert wt.path != ROOT.resolve()
    finally:
        wt.cleanup()


def test_apply_subs_refuses_paths_outside_the_worktree(tmp_path: Path) -> None:
    # A sub that tries to escape the worktree is refused (the guard re-checks).
    wt = Worktree.__new__(Worktree)
    wt.path = (tmp_path / "wt").resolve()
    wt.path.mkdir()
    wt.live_root = ROOT.resolve()
    wt._parent = tmp_path
    with pytest.raises((LiveRepoError, ValueError)):
        wt.apply_subs([{"file": "../escape.py", "find": "a", "replace": "b"}])
