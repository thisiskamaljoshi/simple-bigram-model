from pathlib import Path


def test_changelog_structure_and_standards() -> None:
    changelog_path = Path("CHANGELOG.md")
    assert changelog_path.exists(), "CHANGELOG.md must exist in repository root"

    content = changelog_path.read_text(encoding="utf-8")

    # Keep a Changelog & SemVer standards
    assert "Keep a Changelog" in content
    assert "Semantic Versioning" in content

    # Version headers and categories
    assert "## [Unreleased]" in content
    assert "## [1.0.0] - " in content
    assert "### Added" in content
    assert "### Fixed" in content

    # Key v1.0.0 deliverables documented
    assert "Maximum Likelihood Estimation" in content
    assert "Tokenizer" in content or "tokenizer" in content
    assert "Sentence boundary" in content or "boundary" in content
    assert "Temperature" in content
    assert "Perplexity" in content or "perplexity" in content
    assert "demo.ipynb" in content
    assert "simple-bigram" in content


def test_security_policy_structure() -> None:
    security_path = Path("SECURITY.md")
    assert security_path.exists(), "SECURITY.md must exist in repository root"

    content = security_path.read_text(encoding="utf-8")

    # Supported versions table
    assert "Supported Versions" in content
    assert "1.0.x" in content

    # Reporting process & guidelines
    assert "Reporting a Vulnerability" in content
    assert "Private Vulnerability Reporting" in content or "private reporting" in content
    assert "Response Timeline" in content
    assert "Security Scope" in content or "Threat Model" in content


def test_version_consistency() -> None:
    pyproject_path = Path("pyproject.toml")
    changelog_path = Path("CHANGELOG.md")

    pyproject_content = pyproject_path.read_text(encoding="utf-8")
    changelog_content = changelog_path.read_text(encoding="utf-8")

    # Verify version 1.0.0 is declared in pyproject.toml and documented in changelog
    assert 'version = "1.0.0"' in pyproject_content
    assert "[1.0.0]" in changelog_content
