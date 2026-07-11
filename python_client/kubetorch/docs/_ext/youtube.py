"""Sphinx directive for privacy-enhanced YouTube embeds."""

from __future__ import annotations

import html
import re

from docutils import nodes
from docutils.parsers.rst import Directive, directives


VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def youtube_html(video_id: str, title: str) -> str:
    """Return a responsive, privacy-enhanced embed with a normal link fallback."""

    if not VIDEO_ID.fullmatch(video_id):
        raise ValueError(f"invalid YouTube video id: {video_id!r}")
    safe_title = html.escape(title, quote=True)
    return f"""<div class="kt-video">
  <iframe
    src="https://www.youtube-nocookie.com/embed/{video_id}"
    title="{safe_title}"
    loading="lazy"
    referrerpolicy="strict-origin-when-cross-origin"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen></iframe>
  <p><a href="https://www.youtube.com/watch?v={video_id}">Watch {safe_title} on YouTube</a></p>
</div>"""


class YouTubeDirective(Directive):
    required_arguments = 1
    final_argument_whitespace = False
    has_content = False
    option_spec = {"title": directives.unchanged}

    def run(self):
        video_id = self.arguments[0]
        title = self.options.get("title", "Kubetorch tutorial video")
        try:
            markup = youtube_html(video_id, title)
        except ValueError as error:
            raise self.error(str(error)) from error
        return [nodes.raw("", markup, format="html")]


def setup(app):
    app.add_directive("youtube", YouTubeDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
