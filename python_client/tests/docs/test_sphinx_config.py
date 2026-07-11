import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "python_client/kubetorch/docs"


def _redirect_keys() -> set[str]:
    tree = ast.parse((DOCS_ROOT / "conf.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "redirects"
            for target in node.targets
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
