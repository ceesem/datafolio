"""Reproducibility context capture for snapshots.

Git state, Python environment, and execution info recorded by
``create_snapshot``. Captured from the *current working directory* (the
running code), never the bundle directory. Environment variables are never
captured.
"""

from typing import Any, Dict, Optional


class ContextCaptureMixin:
    """Context-capture methods of :class:`~datafolio.folio.DataFolio`."""

    def _sanitize_git_remote_url(self, url: str) -> Optional[str]:
        """Remove credentials from git remote URLs.

        Handles various git URL formats and removes embedded credentials
        (tokens, username:password) from HTTP(S) URLs while preserving the
        repository information.

        Args:
            url: Git remote URL (potentially with embedded credentials)

        Returns:
            Sanitized URL with credentials removed, or None if sanitization fails

        Examples:
            >>> _sanitize_git_remote_url('https://token@github.com/user/repo.git')
            'https://github.com/user/repo.git'

            >>> _sanitize_git_remote_url('https://user:pass@gitlab.com/repo.git')
            'https://gitlab.com/repo.git'

            >>> _sanitize_git_remote_url('git@github.com:user/repo.git')
            'git@github.com:user/repo.git'  # SSH format preserved (no credentials)

        Security:
            This prevents credential leakage when snapshots containing git
            information are shared with collaborators or made public.
        """
        from urllib.parse import urlparse, urlunparse

        if not url:
            return None

        # Handle SSH format (git@host:path) - safe to keep as-is
        # SSH URLs don't contain credentials, they use SSH keys
        if url.startswith("git@") or url.startswith("ssh://"):
            return url

        # Handle git:// protocol - no credentials possible
        if url.startswith("git://"):
            return url

        # Handle file paths - local repositories
        if url.startswith("/") or url.startswith("file://"):
            return url

        # Handle HTTP(S) URLs - need to strip credentials if present
        try:
            parsed = urlparse(url)

            # If it's http/https and has userinfo (credentials before @)
            if parsed.scheme in ("http", "https"):
                # Check if there's an @ in the netloc (indicates userinfo)
                if "@" in parsed.netloc:
                    # Extract just the host:port part (everything after @)
                    host_with_port = parsed.netloc.split("@")[-1]

                    # Rebuild URL without credentials
                    clean_url = urlunparse(
                        (
                            parsed.scheme,
                            host_with_port,  # Just host:port, no userinfo
                            parsed.path,
                            parsed.params,
                            parsed.query,
                            parsed.fragment,
                        )
                    )
                    return clean_url

                # No credentials present - return as-is
                return url

            # Other schemes - return as-is
            return url

        except Exception:
            # If parsing fails, safer to return None than risk leaking
            return None

    def _capture_git_info(self) -> Optional[Dict[str, Any]]:
        """Capture current git repository state.

        Captures commit hash, branch name, dirty status, and remote URL.
        Remote URLs are automatically sanitized to remove embedded credentials
        (tokens, passwords) for security.

        Returns:
            Git info dict with:
            - commit: Full commit hash
            - commit_short: Short (7-char) commit hash
            - branch: Current branch name
            - dirty: Whether there are uncommitted changes (bool)
            - remote: Repository URL (sanitized, credentials removed)
            Or None if not a git repository

        The state captured is that of the repository containing the *current
        working directory* — i.e. the code that is running — not the bundle
        directory (which is often outside the code repo, or in the cloud).

        Security:
            - Git remote URLs like "https://token@github.com/repo.git" are
              automatically cleaned to "https://github.com/repo.git"
            - Uncommitted file list is NOT captured to avoid exposing sensitive
              filenames (.env, secrets.yaml, etc.)
        """
        import subprocess
        from pathlib import Path

        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return None

            # Get commit hash
            commit_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            commit = (
                commit_result.stdout.strip() if commit_result.returncode == 0 else ""
            )
            commit_short = commit[:7] if commit else ""

            # Get branch name
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            branch = (
                branch_result.stdout.strip() if branch_result.returncode == 0 else ""
            )

            # Get remote URL
            remote_result = subprocess.run(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            remote = (
                remote_result.stdout.strip() if remote_result.returncode == 0 else None
            )

            # Check for uncommitted changes (dirty flag only, no file list for security)
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path.cwd(),
                capture_output=True,
                text=True,
                timeout=5,
            )
            dirty = (
                bool(status_result.stdout.strip())
                if status_result.returncode == 0
                else False
            )

            git_info: Dict[str, Any] = {
                "commit": commit,
                "commit_short": commit_short,
                "branch": branch,
                "dirty": dirty,
            }

            # Sanitize remote URL to remove any embedded credentials
            if remote:
                sanitized_remote = self._sanitize_git_remote_url(remote)
                if sanitized_remote:  # Only include if sanitization succeeded
                    git_info["remote"] = sanitized_remote

            return git_info

        except Exception:
            # Git not available or error occurred
            return None

    def _capture_environment_info(self) -> Dict[str, Any]:
        """Capture Python environment information.

        Returns:
            Environment info dict with Python version, platform, packages
        """
        import platform
        import sys
        from pathlib import Path

        env_info: Dict[str, Any] = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }

        # Try to capture uv.lock hash if it exists
        try:
            import hashlib

            uv_lock = Path.cwd() / "uv.lock"
            if uv_lock.exists():
                with open(uv_lock, "rb") as f:
                    lock_hash = hashlib.md5(f.read()).hexdigest()
                env_info["uv_lock_hash"] = lock_hash
        except Exception:
            pass

        # Try to capture requirements
        try:
            requirements_file = Path.cwd() / "requirements.txt"
            if requirements_file.exists():
                env_info["requirements"] = requirements_file.read_text()
        except Exception:
            pass

        return env_info

    def _capture_execution_info(self) -> Dict[str, Any]:
        """Capture execution context for reproducibility.

        Returns:
            Execution info dict with entry point and working directory
        """
        import sys
        from pathlib import Path

        exec_info: Dict[str, Any] = {
            "working_dir": str(Path.cwd()),
        }

        # Try to capture command line that was run
        if sys.argv:
            exec_info["entry_point"] = " ".join(sys.argv)

        return exec_info
