import subprocess
import sys
from pathlib import Path

import pytest

from scripts.docs.catalog import Tutorial, Validation
from scripts.docs.render_tutorials import build_outputs, GENERATED_MARKER, render_literate_source, sync_outputs


REPO_ROOT = Path(__file__).resolve().parents[3]


def tutorial(**overrides):
    values = {
        "id": "demo",
        "title": "A Demo",
        "category": "Training",
        "source": "examples/tutorials/demo.py",
        "slug": "tutorials/training/demo",
        "hardware": ["CPU"],
        "video_id": None,
        "validation": Validation(state="adapted", fork_version="0.5.2"),
    }
    values.update(overrides)
    return Tutorial(**values)


def test_render_literate_source_interleaves_markdown_and_code():
    source = """# # Source title
# Intro paragraph.
import kubetorch as kt

# This remains a code comment.
compute = kt.Compute(cpus=\"1\")

# ## Dispatch the function
# Send normal Python to the cluster.
remote = kt.fn(lambda: \"ok\").to(compute)
"""

    rendered = render_literate_source(source, tutorial())

    assert rendered.startswith(f"{GENERATED_MARKER}\n# A Demo\n")
    assert "**Adapted**" in rendered
    assert "Requires: `CPU`" in rendered
    assert "# Intro paragraph." not in rendered
    assert "Intro paragraph." in rendered
    assert "```python\nimport kubetorch as kt" in rendered
    assert "# This remains a code comment." in rendered
    assert "## Dispatch the function" in rendered
    assert "remote = kt.fn" in rendered


def test_render_literate_source_discards_legacy_video_marker():
    source = """# # Video demo
# ::youtube[Old marker]{url=\"https://www.youtube.com/watch?v=abc123\"}
print(\"hello\")
"""

    rendered = render_literate_source(source, tutorial(video_id="abc123"))

    assert "::youtube" not in rendered
    assert "```{youtube}" not in rendered
    assert "abc123" not in rendered


def test_sync_outputs_check_mode_detects_drift(tmp_path):
    destination = tmp_path / "generated"
    destination.mkdir()
    output = destination / "tutorials/demo.md"
    output.parent.mkdir()
    output.write_text("stale\n")

    assert sync_outputs({Path("tutorials/demo.md"): "fresh\n"}, destination, check=True) == [output]
    assert output.read_text() == "stale\n"

    assert sync_outputs({Path("tutorials/demo.md"): "fresh\n"}, destination, check=False) == []
    assert output.read_text() == "fresh\n"


def test_sync_outputs_preserves_hand_authored_indexes(tmp_path):
    destination = tmp_path / "docs"
    index = destination / "tutorials/training/index.md"
    index.parent.mkdir(parents=True)
    index.write_text("# Training\n")

    assert sync_outputs({}, destination, check=True) == []
    assert sync_outputs({}, destination, check=False) == []
    assert index.read_text() == "# Training\n"


def test_sync_outputs_only_deletes_owned_generated_pages(tmp_path):
    destination = tmp_path / "docs"
    generated = destination / "tutorials/generated.md"
    hand_authored = destination / "tutorials/notes.md"
    generated.parent.mkdir(parents=True)
    generated.write_text(f"{GENERATED_MARKER}\n# Old generated page\n")
    hand_authored.write_text("# Maintainer notes\n")

    assert sync_outputs({}, destination, check=False) == []
    assert not generated.exists()
    assert hand_authored.read_text() == "# Maintainer notes\n"


def test_sync_outputs_rejects_paths_outside_tutorials(tmp_path):
    destination = tmp_path / "docs"

    with pytest.raises(ValueError, match="outside generated tutorial root"):
        sync_outputs({Path("tutorials/../../escape.md"): "bad\n"}, destination, check=False)


def test_source_inventory_links_every_imported_file():
    inventory = build_outputs()[Path("tutorials/source-inventory.md")]
    examples_root = REPO_ROOT / "examples/tutorials"
    imported = {
        str(path.relative_to(REPO_ROOT))
        for path in examples_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }

    assert imported
    assert all(path in inventory for path in imported)


def test_renderer_cli_runs_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/docs/render_tutorials.py", "--check"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "ModuleNotFoundError" not in result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
