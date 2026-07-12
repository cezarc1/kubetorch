from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_ROOT = REPO_ROOT / "python_client/kubetorch/docs"
EXAMPLES_ROOT = REPO_ROOT / "examples/tutorials"
RECOVERY = DOCS_ROOT / "_data/video_recovery.yaml"

VIDEO_IDS = {
    "CH0mMcR5hZ8",
    "k1olO4P_1WY",
    "5nRRaxZJnUg",
    "sqNYnowFufY",
    "yB29sojkAiU",
    "-oz49qt_uSM",
    "VCWpCM2m1Hw",
    "8slAR7459X4",
    "9vQww8bhCzY",
    "yJ3b6Gps9qI",
}


def test_video_recovery_records_every_reviewed_source():
    data = yaml.safe_load(RECOVERY.read_text())
    videos = data["videos"]

    assert len(videos) == len(VIDEO_IDS)
    assert {video["id"] for video in videos} == VIDEO_IDS
    assert len({video["id"] for video in videos}) == len(videos)

    for video in videos:
        assert set(video) == {
            "id",
            "title",
            "source",
            "target",
            "disposition",
            "context",
        }
        assert video["disposition"] in {"context-added", "no-new-context"}
        assert video["context"].strip()
        if video["source"] is not None:
            assert (REPO_ROOT / video["source"]).is_file()
        if video["target"] is not None:
            assert (DOCS_ROOT / f"{video['target']}.md").is_file()


def test_maintained_sources_have_no_video_presentation_artifacts():
    patterns = (
        "youtube.com",
        "youtu.be",
        "youtube-nocookie.com",
        "```{youtube}",
        "::youtube[",
        "_ext.youtube",
        ".kt-video",
    )
    suffixes = {".css", ".html", ".md", ".py", ".rst", ".yaml"}
    offenders = {}

    for root in (DOCS_ROOT, EXAMPLES_ROOT):
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            source = path.read_text()
            matches = [pattern for pattern in patterns if pattern in source]
            if matches:
                offenders[str(path.relative_to(REPO_ROOT))] = matches

    assert offenders == {}
