from scripts.docs.recover_runhouse import extract_main_markdown, wayback_url


def test_wayback_url_pins_timestamp_and_uses_identity_replay():
    assert wayback_url("/kubetorch/guides/summary", "20251010000215") == (
        "https://web.archive.org/web/20251010000215id_/"
        "https://www.run.house/kubetorch/guides/summary"
    )


def test_extract_main_markdown_drops_site_shell_and_preserves_content():
    html = """
    <html><body>
      <nav>Runhouse navigation</nav>
      <main>
        <h1>Developer Workflow</h1>
        <p>Define compute and dispatch Python.</p>
        <pre><code>compute = kt.Compute(cpus=\"1\")</code></pre>
        <a href=\"/kubetorch/concepts\">Concepts</a>
      </main>
      <footer>All Rights Reserved</footer>
    </body></html>
    """

    markdown = extract_main_markdown(html)

    assert markdown.startswith("# Developer Workflow")
    assert "Define compute" in markdown
    assert "```\ncompute = kt.Compute" in markdown
    assert "[Concepts](/kubetorch/concepts)" in markdown
    assert "Runhouse navigation" not in markdown
    assert "All Rights Reserved" not in markdown
