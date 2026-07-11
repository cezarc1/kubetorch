"""Check imported tutorials for removed Kubetorch APIs and stale services."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples/tutorials"
TEXT_SUFFIXES = {".md", ".py", ".txt", ".yaml", ".yml"}
STALE_REFERENCES = (
    "https://www.run.house",
    "ghcr.io/run-house",
    "github.com/run-house/kubetorch-examples",
)


@dataclass(frozen=True)
class CompatibilityIssue:
    path: Path
    message: str


def collect_kt_attributes(source: str) -> set[str]:
    """Collect top-level attributes used through ``import kubetorch as kt``."""

    tree = ast.parse(source)
    aliases = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "kubetorch"
    }
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    }


def find_compatibility_issues(
    paths: list[Path], *, public_names: set[str]
) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []
    for path in paths:
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = path.read_text(errors="replace")
        if path.suffix == ".py":
            try:
                attributes = collect_kt_attributes(text)
            except SyntaxError as error:
                issues.append(CompatibilityIssue(path, f"invalid Python: {error}"))
            else:
                for attribute in sorted(attributes - public_names):
                    issues.append(
                        CompatibilityIssue(path, f"unknown public API: kt.{attribute}")
                    )

        attribution_only = (
            path.name == "README.md"
            and "original" in text.lower()
            and "github.com/run-house/kubetorch-examples" in text
        )
        for reference in STALE_REFERENCES:
            if reference in text and not (
                attribution_only
                and reference == "github.com/run-house/kubetorch-examples"
            ):
                issues.append(
                    CompatibilityIssue(path, f"stale service reference: {reference}")
                )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=[EXAMPLES_ROOT])
    args = parser.parse_args()

    import kubetorch

    files: list[Path] = []
    for path in args.paths:
        files.extend(path.rglob("*") if path.is_dir() else [path])
    issues = find_compatibility_issues(files, public_names=set(dir(kubetorch)))
    for issue in issues:
        try:
            display = issue.path.relative_to(REPO_ROOT)
        except ValueError:
            display = issue.path
        print(f"{display}: {issue.message}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
