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
    assert all(tutorial.video_id is None for tutorial in catalog.tutorials)
    assert all((REPO_ROOT / tutorial.source).is_file() for tutorial in catalog.tutorials)


def test_every_literate_example_is_cataloged():
    catalog = load_catalog(CATALOG_PATH, repo_root=REPO_ROOT)
    examples_root = REPO_ROOT / "examples/tutorials"
    literate_sources = {
        str(path.relative_to(REPO_ROOT)) for path in examples_root.rglob("*.py") if path.read_text().startswith("# # ")
    }

    assert {tutorial.source for tutorial in catalog.tutorials} == literate_sources


def test_every_route_replacement_is_a_real_sphinx_document():
    catalog = load_catalog(CATALOG_PATH, repo_root=REPO_ROOT)
    docs_root = REPO_ROOT / "python_client/kubetorch/docs"
    docnames = {
        str(path.relative_to(docs_root).with_suffix(""))
        for suffix in ("*.md", "*.rst")
        for path in docs_root.rglob(suffix)
        if path.name != "README.md"
    }

    assert {route.replacement for route in catalog.routes} <= docnames


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


def test_catalog_rejects_tutorial_slug_traversal(tmp_path):
    data = yaml.safe_load(CATALOG_PATH.read_text())
    data["tutorials"][0]["slug"] = "tutorials/../../outside"
    invalid_catalog = tmp_path / "catalog.yaml"
    invalid_catalog.write_text(yaml.safe_dump(data, sort_keys=False))

    with pytest.raises(CatalogError, match="invalid tutorial slug"):
        load_catalog(invalid_catalog, repo_root=REPO_ROOT)


@pytest.mark.parametrize("missing", ["date", "hardware", "evidence"])
def test_validated_tutorials_require_evidence_metadata(tmp_path, missing):
    data = yaml.safe_load(CATALOG_PATH.read_text())
    validation = data["tutorials"][0]["validation"]
    validation.update(
        {
            "state": "validated",
            "date": "2026-07-11",
            "hardware": "cezar-4090-cluster",
            "evidence": "validation-evidence/mnist.json",
        }
    )
    del validation[missing]
    invalid_catalog = tmp_path / "catalog.yaml"
    invalid_catalog.write_text(yaml.safe_dump(data, sort_keys=False))

    with pytest.raises(CatalogError, match=f"validated tutorial.*{missing}"):
        load_catalog(invalid_catalog, repo_root=REPO_ROOT)
