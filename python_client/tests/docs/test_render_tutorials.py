from pathlib import Path
import subprocess
import sys

from scripts.docs.catalog import Tutorial, Validation
from scripts.docs.render_tutorials import render_literate_source, sync_outputs


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

    assert rendered.startswith("# A Demo\n")
    assert "**Adapted**" in rendered
    assert "Requires: `CPU`" in rendered
    assert "# Intro paragraph." not in rendered
    assert "Intro paragraph." in rendered
    assert "```python\nimport kubetorch as kt" in rendered
    assert "# This remains a code comment." in rendered
    assert "## Dispatch the function" in rendered
    assert "remote = kt.fn" in rendered


def test_render_literate_source_uses_privacy_enhanced_video_directive():
    source = """# # Video demo
# ::youtube[Old marker]{url=\"https://www.youtube.com/watch?v=abc123\"}
print(\"hello\")
"""

    rendered = render_literate_source(source, tutorial(video_id="abc123"))

    assert "::youtube" not in rendered
    assert "```{youtube} abc123" in rendered
    assert "youtube-nocookie.com" not in rendered


def test_sync_outputs_check_mode_detects_drift(tmp_path):
    destination = tmp_path / "generated"
    destination.mkdir()
    output = destination / "demo.md"
    output.write_text("stale\n")

    assert sync_outputs({Path("demo.md"): "fresh\n"}, destination, check=True) == [output]
    assert output.read_text() == "stale\n"

    assert sync_outputs({Path("demo.md"): "fresh\n"}, destination, check=False) == []
    assert output.read_text() == "fresh\n"


def test_renderer_cli_runs_from_repository_root():
    result = subprocess.run(
        [sys.executable, "scripts/docs/render_tutorials.py", "--check"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert "ModuleNotFoundError" not in result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
