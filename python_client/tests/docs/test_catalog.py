from pathlib import Path

import pytest
import yaml

from scripts.docs.catalog import CatalogError, load_catalog


REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = REPO_ROOT / "python_client/kubetorch/docs/_data/catalog.yaml"


def test_repository_catalog_covers_recovered_routes_and_tutorials():
    catalog = load_catalog(CATALOG_PATH, repo_root=REPO_ROOT)

    assert len(catalog.routes) == 41
    assert len(catalog.tutorials) == 33
    assert {tutorial.validation.state for tutorial in catalog.tutorials} == {
        "adapted",
        "reference",
    }
    assert all((REPO_ROOT / tutorial.source).is_file() for tutorial in catalog.tutorials)


def test_catalog_rejects_duplicate_tutorial_ids(tmp_path):
    data = yaml.safe_load(CATALOG_PATH.read_text())
    data["tutorials"].append(data["tutorials"][0])
    duplicate_catalog = tmp_path / "catalog.yaml"
    duplicate_catalog.write_text(yaml.safe_dump(data, sort_keys=False))

    with pytest.raises(CatalogError, match="duplicate tutorial id"):
        load_catalog(duplicate_catalog, repo_root=REPO_ROOT)


def test_catalog_rejects_missing_example_source(tmp_path):
    data = yaml.safe_load(CATALOG_PATH.read_text())
    data["tutorials"][0]["source"] = "examples/tutorials/missing.py"
    missing_source_catalog = tmp_path / "catalog.yaml"
    missing_source_catalog.write_text(yaml.safe_dump(data, sort_keys=False))

    with pytest.raises(CatalogError, match="source does not exist"):
        load_catalog(missing_source_catalog, repo_root=REPO_ROOT)

