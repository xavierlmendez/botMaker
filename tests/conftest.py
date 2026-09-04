"""Fixtures shared by the whole suite."""

from pathlib import Path

import pytest

# tests/conftest.py -> repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repository_root() -> Path:
    """The repository root, so tests reach committed data without depending on the working dir."""
    return REPOSITORY_ROOT


@pytest.fixture
def uci_data_dir(repository_root: Path) -> Path:
    """The committed UCI datasets; see `data/uci/README.md` for provenance."""
    return repository_root / "data" / "uci"
