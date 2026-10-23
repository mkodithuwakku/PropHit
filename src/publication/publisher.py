"""Git commit & push of the built site to GitHub Pages (Milestone 6).

Publishes the static bundle to the ``gh-pages`` branch via a temporary git
worktree so the main working tree is untouched. ``dry_run`` returns the
exact command plan without side effects — used by tests and by the daily
runner's verification mode.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


class GitPublisher:
    def __init__(
        self,
        repo_path: "str | Path" = ".",
        branch: str = "gh-pages",
        remote: str = "origin",
    ):
        self.repo_path = str(repo_path)
        self.branch = branch
        self.remote = remote

    def _git(
        self, *args: str, cwd: Optional[str] = None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=cwd or self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

    def publish(
        self, site_dir: "str | Path", message: str, dry_run: bool = True
    ) -> dict:
        """Publish ``site_dir`` contents to the pages branch.

        Returns a summary dict; with ``dry_run=True`` no git state changes.
        """
        site_dir = Path(site_dir)
        commands = [
            f"git worktree add --detach <tmp> {self.branch} " f"(or orphan if missing)",
            f"rsync -a --delete {site_dir}/ <tmp>/",
            "git add -A",
            f'git commit -m "{message}"',
            f"git push {self.remote} {self.branch}",
            "git worktree remove <tmp>",
        ]
        if dry_run:
            return {
                "dry_run": True,
                "branch": self.branch,
                "remote": self.remote,
                "commands": commands,
            }

        tmp = Path(tempfile.mkdtemp(prefix="prophit-ghpages-"))
        try:
            try:
                self._git("worktree", "add", "--detach", str(tmp), self.branch)
            except subprocess.CalledProcessError:
                self._git("worktree", "add", "--detach", str(tmp))
                self._git("checkout", "--orphan", self.branch, cwd=str(tmp))
                self._git("rm", "-rf", ".", cwd=str(tmp))

            for item in tmp.iterdir():
                if item.name == ".git":
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            shutil.copytree(site_dir, tmp, dirs_exist_ok=True)

            self._git("add", "-A", cwd=str(tmp))
            try:
                self._git("commit", "-m", message, cwd=str(tmp))
            except subprocess.CalledProcessError:
                return {"published": False, "reason": "nothing to commit"}
            self._git("push", self.remote, self.branch, cwd=str(tmp))
            return {"published": True, "branch": self.branch}
        finally:
            try:
                self._git("worktree", "remove", "--force", str(tmp))
            except subprocess.CalledProcessError:
                shutil.rmtree(tmp, ignore_errors=True)
