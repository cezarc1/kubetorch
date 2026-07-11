"""Recover a pinned Runhouse documentation page as reviewable Markdown."""

from __future__ import annotations

import argparse
import html as html_module
from pathlib import Path
from urllib.request import urlopen

from bs4 import BeautifulSoup, NavigableString, Tag


def wayback_url(path: str, timestamp: str) -> str:
    return f"https://web.archive.org/web/{timestamp}id_/" f"https://www.run.house{path}"


def _inline(node: Tag | NavigableString) -> str:
    if isinstance(node, NavigableString):
        return str(node)
    text = "".join(_inline(child) for child in node.children)
    if node.name == "a":
        return f"[{text.strip()}]({node.get('href', '')})"
    if node.name == "code":
        return f"`{text}`"
    if node.name in {"strong", "b"}:
        return f"**{text}**"
    if node.name in {"em", "i"}:
        return f"*{text}*"
    if node.name == "br":
        return "\n"
    return text


def _blocks(container: Tag) -> list[str]:
    blocks: list[str] = []
    for node in container.find_all(recursive=False):
        if node.name in {"script", "style", "nav", "footer"}:
            continue
        if node.name and node.name.startswith("h") and node.name[1:].isdigit():
            blocks.append(f"{'#' * int(node.name[1:])} {_inline(node).strip()}")
        elif node.name == "p":
            blocks.append(_inline(node).strip())
        elif node.name == "a":
            blocks.append(_inline(node).strip())
        elif node.name == "pre":
            code = node.get_text().strip("\n")
            blocks.append(f"```\n{code}\n```")
        elif node.name in {"ul", "ol"}:
            marker = "-" if node.name == "ul" else "1."
            blocks.append(
                "\n".join(
                    f"{marker} {_inline(item).strip()}"
                    for item in node.find_all("li", recursive=False)
                )
            )
        elif node.name == "blockquote":
            blocks.append(
                "\n".join(f"> {line}" for line in _inline(node).strip().splitlines())
            )
        elif isinstance(node, Tag):
            nested = _blocks(node)
            if nested:
                blocks.extend(nested)
    return [block for block in blocks if block]


def extract_main_markdown(html: str) -> str:
    """Extract the article body while discarding the archived site shell."""

    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("main") or soup.find("article")
    if container is None:
        raise ValueError("archived page has no <main> or <article> content")
    markdown = "\n\n".join(_blocks(container)).strip()
    return html_module.unescape(markdown) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", help="original path, for example /kubetorch/guides/summary"
    )
    parser.add_argument("timestamp", help="14-digit Wayback timestamp")
    parser.add_argument("output", type=Path, help="Markdown output path")
    args = parser.parse_args()

    with urlopen(wayback_url(args.path, args.timestamp), timeout=60) as response:
        markdown = extract_main_markdown(response.read().decode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
