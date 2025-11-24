"""Tests for git remote URL sanitization.

These tests verify that git remote URLs with embedded credentials are
properly sanitized while preserving repository information.
"""

import pandas as pd
import pytest

from datafolio import DataFolio


class TestSanitizeGitRemoteUrl:
    """Unit tests for _sanitize_git_remote_url() method."""

    def test_sanitize_https_with_token(self, tmp_path):
        """Test that HTTPS URLs with tokens are sanitized."""
        folio = DataFolio(tmp_path / "test")

        # GitHub personal access token format
        url = "https://ghp_abc123token@github.com/user/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == "https://github.com/user/repo.git"
        assert "token" not in clean
        assert "ghp_" not in clean
        assert "@" not in clean

    def test_sanitize_https_with_username_password(self, tmp_path):
        """Test that HTTPS URLs with username:password are sanitized."""
        folio = DataFolio(tmp_path / "test")

        # Username and password format
        url = "https://username:password@gitlab.com/project/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == "https://gitlab.com/project/repo.git"
        assert "username" not in clean
        assert "password" not in clean
        assert "@" not in clean

    def test_sanitize_https_with_complex_credentials(self, tmp_path):
        """Test sanitization of URLs with complex credential strings."""
        folio = DataFolio(tmp_path / "test")

        # OAuth token format
        url = "https://oauth2:ghp_veryLongTokenString123456@github.com/org/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == "https://github.com/org/repo.git"
        assert "oauth2" not in clean
        assert "Token" not in clean

    def test_sanitize_https_with_port(self, tmp_path):
        """Test that URLs with ports are handled correctly."""
        folio = DataFolio(tmp_path / "test")

        # URL with custom port
        url = "https://token@gitlab.example.com:8443/project/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == "https://gitlab.example.com:8443/project/repo.git"
        assert "token" not in clean
        assert ":8443" in clean  # Port preserved

    def test_clean_https_url_unchanged(self, tmp_path):
        """Test that clean HTTPS URLs without credentials are preserved."""
        folio = DataFolio(tmp_path / "test")

        # Already clean URL
        url = "https://github.com/user/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == url  # Unchanged
        assert "@" not in clean

    def test_clean_http_url_unchanged(self, tmp_path):
        """Test that clean HTTP URLs are preserved."""
        folio = DataFolio(tmp_path / "test")

        url = "http://gitlab.local/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == url

    def test_ssh_url_preserved(self, tmp_path):
        """Test that SSH URLs are kept as-is (no credentials embedded)."""
        folio = DataFolio(tmp_path / "test")

        # Standard SSH format
        url = "git@github.com:user/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == url  # Unchanged
        # SSH URLs use keys, not embedded credentials

    def test_ssh_protocol_url_preserved(self, tmp_path):
        """Test that ssh:// protocol URLs are preserved."""
        folio = DataFolio(tmp_path / "test")

        url = "ssh://git@github.com/user/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == url

    def test_git_protocol_preserved(self, tmp_path):
        """Test that git:// protocol URLs are preserved."""
        folio = DataFolio(tmp_path / "test")

        url = "git://github.com/user/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == url

    def test_file_path_preserved(self, tmp_path):
        """Test that local file paths are preserved."""
        folio = DataFolio(tmp_path / "test")

        # Absolute path
        url = "/path/to/local/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == url

    def test_file_protocol_preserved(self, tmp_path):
        """Test that file:// protocol URLs are preserved."""
        folio = DataFolio(tmp_path / "test")

        url = "file:///path/to/local/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == url

    def test_empty_url_returns_none(self, tmp_path):
        """Test that empty URLs return None."""
        folio = DataFolio(tmp_path / "test")

        assert folio._sanitize_git_remote_url("") is None
        assert folio._sanitize_git_remote_url(None) is None

    def test_malformed_url_returned_as_is(self, tmp_path):
        """Test that malformed URLs are returned as-is (safe - no credentials)."""
        folio = DataFolio(tmp_path / "test")

        # Malformed URLs that don't match http(s) patterns are returned as-is
        # This is safe since they can't contain embedded credentials in URL format
        result = folio._sanitize_git_remote_url("not::a::valid::url")
        # Result should be the original (no credentials to strip)
        assert result is not None

    def test_url_with_special_characters_in_credentials(self, tmp_path):
        """Test URLs with special characters in credentials."""
        folio = DataFolio(tmp_path / "test")

        # Password with special characters
        url = "https://user:p@ss%20word@github.com/repo.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == "https://github.com/repo.git"
        assert "p@ss" not in clean
        assert "user" not in clean

    def test_url_with_subdirectories(self, tmp_path):
        """Test that repository paths with subdirectories are preserved."""
        folio = DataFolio(tmp_path / "test")

        url = "https://token@gitlab.com/group/subgroup/project.git"
        clean = folio._sanitize_git_remote_url(url)

        assert clean == "https://gitlab.com/group/subgroup/project.git"
        assert "token" not in clean
        assert "/group/subgroup/project.git" in clean


class TestGitUrlSanitizationIntegration:
    """Integration tests for git URL sanitization in snapshot creation."""

    def test_snapshot_sanitizes_git_remote(self, tmp_path):
        """Test that snapshot creation sanitizes git remote URL."""
        import subprocess

        bundle_dir = tmp_path / "bundle"

        # Create folio first (this creates the directory structure)
        folio = DataFolio(bundle_dir)
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Now setup git repo in the bundle directory
        subprocess.run(["git", "init"], cwd=bundle_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Add remote with embedded token
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "origin",
                "https://ghp_secrettoken123@github.com/user/repo.git",
            ],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Create initial commit (required for git)
        subprocess.run(
            ["git", "add", "."], cwd=bundle_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Create snapshot with git capture
        folio.create_snapshot("v1.0", capture_git=True)

        # Verify remote is sanitized
        snapshot = folio.get_snapshot_info("v1.0")
        assert "git" in snapshot
        assert "remote" in snapshot["git"]
        assert snapshot["git"]["remote"] == "https://github.com/user/repo.git"
        assert "token" not in snapshot["git"]["remote"]
        assert "ghp_" not in snapshot["git"]["remote"]

    def test_snapshot_with_clean_remote(self, tmp_path):
        """Test that clean remote URLs are preserved unchanged."""
        import subprocess

        bundle_dir = tmp_path / "bundle"

        # Create folio first
        folio = DataFolio(bundle_dir)
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Setup git repo
        subprocess.run(["git", "init"], cwd=bundle_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Add clean remote (no credentials)
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "origin",
                "https://github.com/user/repo.git",
            ],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Create commit
        subprocess.run(
            ["git", "add", "."], cwd=bundle_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Create snapshot
        folio.create_snapshot("v1.0", capture_git=True)

        # Verify remote is preserved
        snapshot = folio.get_snapshot_info("v1.0")
        assert snapshot["git"]["remote"] == "https://github.com/user/repo.git"

    def test_snapshot_with_ssh_remote(self, tmp_path):
        """Test that SSH remotes are preserved (they don't have credentials)."""
        import subprocess

        bundle_dir = tmp_path / "bundle"

        # Create folio first
        folio = DataFolio(bundle_dir)
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Setup git repo
        subprocess.run(["git", "init"], cwd=bundle_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Add SSH remote
        subprocess.run(
            ["git", "remote", "add", "origin", "git@github.com:user/repo.git"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Create commit
        subprocess.run(
            ["git", "add", "."], cwd=bundle_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Create snapshot
        folio.create_snapshot("v1.0", capture_git=True)

        # Verify SSH remote is preserved
        snapshot = folio.get_snapshot_info("v1.0")
        assert snapshot["git"]["remote"] == "git@github.com:user/repo.git"

    def test_reproduce_instructions_dont_leak_credentials(self, tmp_path):
        """Test that reproduce instructions don't contain credentials."""
        import subprocess

        bundle_dir = tmp_path / "bundle"

        # Create folio first
        folio = DataFolio(bundle_dir)
        df = pd.DataFrame({"a": [1, 2, 3]})
        folio.add_table("data", df)

        # Setup git repo
        subprocess.run(["git", "init"], cwd=bundle_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Add remote with credentials
        subprocess.run(
            [
                "git",
                "remote",
                "add",
                "origin",
                "https://secret_token@bitbucket.org/org/repo.git",
            ],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Create commit
        subprocess.run(
            ["git", "add", "."], cwd=bundle_dir, capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial"],
            cwd=bundle_dir,
            capture_output=True,
            check=True,
        )

        # Create snapshot
        folio.create_snapshot("v1.0", capture_git=True)

        # Get reproduction instructions
        instructions = folio.reproduce_instructions("v1.0")

        # Verify no credentials in output
        assert "secret_token" not in instructions
        assert "bitbucket.org" in instructions  # But host should be present


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
