# Documentation Sitemap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and publish a canonical XML sitemap for every Sphinx documentation build.

**Architecture:** The `sphinx-sitemap` extension consumes Sphinx's public `html_baseurl` and discovered HTML outputs. The normal Pages artifact therefore gains `sitemap.xml` without a separate crawler or committed generated file.

**Tech Stack:** Sphinx 6.2.1, sphinx-sitemap, GitHub Pages

## Global Constraints

- Public URLs must begin with `https://cezarc1.github.io/kubetorch/`.
- The sitemap must be generated, not committed.
- Do not add a documentation test suite.
- Preserve the current Pages workflow and Google integrations.

---

### Task 1: Enable sitemap generation

**Files:**
- Modify: `python_client/kubetorch/docs/requirements.txt`
- Modify: `python_client/kubetorch/docs/conf.py`

**Interfaces:**
- Consumes: Sphinx HTML document outputs and `html_baseurl`
- Produces: `_build/html/sitemap.xml` and canonical HTML links

- [ ] **Step 1: Record the failing baseline**

Build the complete documentation and confirm `sitemap.xml` is absent.

- [ ] **Step 2: Add the extension dependency and configuration**

Pin a Sphinx-6-compatible `sphinx-sitemap` release in the documentation
requirements, add `sphinx_sitemap` to `extensions`, set `html_baseurl` to the
GitHub Pages project URL, and set `sitemap_url_scheme = "{link}"` so Sphinx's
default language does not introduce an `/en/` path.

- [ ] **Step 3: Validate the generated artifact**

Run a clean warning-as-error build. Parse `sitemap.xml` and confirm that every
`loc` begins with the configured base URL. Confirm representative homepage,
tutorial, and API URLs are present and that no `_build` URLs appear.

### Task 2: Repository verification and publication

**Files:**
- Verify: `.github/workflows/build_docs.yaml`

**Interfaces:**
- Consumes: the existing Pages artifact path
- Produces: a reviewable branch whose normal deploy includes `sitemap.xml`

- [ ] **Step 1: Run existing checks**

Run tutorial rendering drift checks, example compatibility checks, example
compilation, the complete warning-as-error Sphinx build, and all pre-commit
hooks.

- [ ] **Step 2: Commit and publish for review**

Commit only the sitemap design, plan, dependency, and Sphinx configuration.
Push `codex/docs-sitemap` and open a draft pull request targeting `main`.
