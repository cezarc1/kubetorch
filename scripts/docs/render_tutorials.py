"""Render literate Python examples into checked-in MyST tutorial pages."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.docs.catalog import Tutorial, load_catalog


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "python_client/kubetorch/docs/_data/catalog.yaml"
DOCS_ROOT = REPO_ROOT / "python_client/kubetorch/docs"
DOC_BLOCK = re.compile(r"^# (#+)(?:\s+(.*))?$")
YOUTUBE_MARKER = re.compile(r"^::youtube\[")


def _strip_comment(line: str) -> str:
    if line == "#":
        return ""
    if line.startswith("# "):
        return line[2:]
    return line[1:] if line.startswith("#") else line


def _code_block(lines: list[str]) -> str:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    return "```python\n" + "\n".join(lines) + "\n```\n"


def _status_panel(tutorial: Tutorial) -> str:
    state = tutorial.validation.state.capitalize()
    requirements = ", ".join(f"`{item}`" for item in tutorial.hardware)
    source_url = (
        "https://github.com/cezarc1/kubetorch/blob/main/" + tutorial.source
    )
    detail = {
        "validated": "Executed against the recorded fork version and hardware.",
        "adapted": "Updated for the current fork and checked statically; cluster execution is not yet recorded.",
        "reference": "Requires external infrastructure or serves as supporting reference code.",
    }[tutorial.validation.state]
    return (
        f"```{{admonition}} {state}\n"
        f":class: kt-status kt-status-{tutorial.validation.state}\n\n"
        f"**{state}** for Kubetorch `{tutorial.validation.fork_version}`. {detail}\n\n"
        f"Requires: {requirements}. [View source]({source_url}).\n"
        "```\n"
    )


def render_literate_source(source: str, tutorial: Tutorial) -> str:
    """Render Markdown comment sections and intervening Python code."""

    lines = source.splitlines()
    parts = [f"# {tutorial.title}\n", _status_panel(tutorial)]
    if tutorial.video_id:
        parts.append(
            f"```{{youtube}} {tutorial.video_id}\n"
            f":title: {tutorial.title}\n"
            "```\n"
        )

    code: list[str] = []
    first_heading = True
    index = 0
    while index < len(lines):
        match = DOC_BLOCK.match(lines[index])
        if not match:
            code.append(lines[index])
            index += 1
            continue

        rendered_code = _code_block(code)
        if rendered_code:
            parts.append(rendered_code)
        code = []

        depth = len(match.group(1))
        heading = match.group(2) or ""
        block: list[str] = []
        index += 1
        while index < len(lines) and (
            lines[index] == "#" or lines[index].startswith("# ")
        ):
            text = _strip_comment(lines[index])
            if not YOUTUBE_MARKER.match(text):
                block.append(text)
            index += 1

        if first_heading and depth == 1:
            first_heading = False
        elif heading:
            parts.append(f"{'#' * depth} {heading}\n")
        if block and any(line.strip() for line in block):
            parts.append("\n".join(block).strip() + "\n")

    rendered_code = _code_block(code)
    if rendered_code:
        parts.append(rendered_code)

    return "\n".join(part.rstrip() for part in parts if part).rstrip() + "\n"


def build_outputs(catalog_path: Path = CATALOG_PATH) -> dict[Path, str]:
    catalog = load_catalog(catalog_path, repo_root=REPO_ROOT)
    outputs: dict[Path, str] = {}
    for tutorial in catalog.tutorials:
        source = (REPO_ROOT / tutorial.source).read_text()
        outputs[Path(f"{tutorial.slug}.md")] = render_literate_source(source, tutorial)
    return outputs


def sync_outputs(
    outputs: dict[Path, str], destination: Path, *, check: bool
) -> list[Path]:
    """Write outputs, or return the files that differ in check mode."""

    drift: list[Path] = []
    expected = {destination / relative for relative in outputs}
    for relative, content in outputs.items():
        path = destination / relative
        if not path.exists() or path.read_text() != content:
            if check:
                drift.append(path)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)

    generated_root = destination / "tutorials"
    if generated_root.exists():
        for path in generated_root.rglob("*.md"):
            if path.name != "index.md" and path not in expected:
                if check:
                    drift.append(path)
                else:
                    path.unlink()
    return sorted(drift)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="write generated tutorials")
    mode.add_argument("--check", action="store_true", help="fail if generated tutorials drift")
    args = parser.parse_args()

    drift = sync_outputs(build_outputs(), DOCS_ROOT, check=args.check)
    if drift:
        print("Generated tutorials are out of date:")
        for path in drift:
            print(f"- {path.relative_to(REPO_ROOT)}")
        print("Run: python scripts/docs/render_tutorials.py --write")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
