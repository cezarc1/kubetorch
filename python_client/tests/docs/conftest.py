import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.level("unit"))


@pytest.fixture(scope="session", autouse=True)
def test_hash_and_teardown():
    """Docs tooling tests never create cluster resources."""

    yield "docs"
