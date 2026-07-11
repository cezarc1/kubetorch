import ast
from pathlib import Path

from scripts.docs.catalog import load_catalog


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "python_client/kubetorch/docs"


def _redirect_keys() -> set[str]:
    tree = ast.parse((DOCS_ROOT / "conf.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "redirects" for target in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError("conf.py does not define redirects")


def test_redirects_never_overwrite_real_document_sources():
    docnames = {
        str(path.relative_to(DOCS_ROOT).with_suffix(""))
        for suffix in ("*.md", "*.rst")
        for path in DOCS_ROOT.rglob(suffix)
    }

    assert _redirect_keys().isdisjoint(docnames)


def test_every_legacy_route_is_a_document_or_redirect():
    catalog = load_catalog(DOCS_ROOT / "_data/catalog.yaml", repo_root=REPO_ROOT)
    docnames = {
        str(path.relative_to(DOCS_ROOT).with_suffix(""))
        for suffix in ("*.md", "*.rst")
        for path in DOCS_ROOT.rglob(suffix)
        if path.name != "README.md"
    }
    legacy_docnames = {route.upstream.removeprefix("/kubetorch/") for route in catalog.routes}

    assert legacy_docnames <= docnames | _redirect_keys()
