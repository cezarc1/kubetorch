import pytest

from kubetorch.docs._ext.youtube import youtube_html


def test_youtube_html_uses_privacy_enhanced_embed_and_fallback_link():
    rendered = youtube_html("abc_123-X", "Training <demo>")

    assert "https://www.youtube-nocookie.com/embed/abc_123-X" in rendered
    assert "https://www.youtube.com/watch?v=abc_123-X" in rendered
    assert "Training &lt;demo&gt;" in rendered
    assert 'loading="lazy"' in rendered


@pytest.mark.parametrize("video_id", ["", "has spaces", "../escape", "a" * 65])
def test_youtube_html_rejects_invalid_video_ids(video_id):
    with pytest.raises(ValueError, match="invalid YouTube video id"):
        youtube_html(video_id, "Demo")
