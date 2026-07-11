from pathlib import Path

from scripts.docs.check_example_compat import collect_kt_attributes, find_compatibility_issues


def test_collect_kt_attributes_finds_public_module_calls():
    source = """import kubetorch as kt
compute = kt.Compute(cpus=\"1\")
remote = kt.fn(lambda: 1).to(compute)
"""

    assert collect_kt_attributes(source) == {"Compute", "fn"}


def test_compatibility_check_reports_removed_api_and_stale_links(tmp_path):
    example = tmp_path / "legacy.py"
    example.write_text(
        "import kubetorch as kt\n"
        "remote = kt.function(lambda: 1)\n"
        "docs = 'https://www.run.house/kubetorch/introduction'\n"
    )

    issues = find_compatibility_issues([example], public_names={"Compute", "fn"})

    assert any("kt.function" in issue.message for issue in issues)
    assert any("www.run.house" in issue.message for issue in issues)


def test_compatibility_check_allows_attribution_links(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        "Originally published at https://github.com/run-house/kubetorch-examples.\n"
    )

    assert find_compatibility_issues([readme], public_names=set()) == []

