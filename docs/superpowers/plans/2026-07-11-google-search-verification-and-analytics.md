# Google Search Verification and Analytics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Inject Search Console verification and optional GA4 markup into the deployed Sphinx site without committing either configured value.

**Architecture:** Sphinx reads optional build-time environment variables into its template context. The existing layout override conditionally renders Google markup, while the Pages workflow supplies production values from GitHub repository configuration.

**Tech Stack:** Sphinx, Jinja templates, GitHub Actions, GitHub Pages, GA4

## Global Constraints

- Do not commit the Google verification token or verification HTML file.
- Do not commit the GA4 measurement ID.
- Builds must succeed when either value is absent.
- Do not add a documentation test suite.

---

### Task 1: Conditional Google markup

**Files:**
- Modify: `python_client/kubetorch/docs/conf.py`
- Modify: `python_client/kubetorch/docs/_templates/layout.html`

**Interfaces:**
- Consumes: optional `GOOGLE_SITE_VERIFICATION` and `GOOGLE_ANALYTICS_ID` environment variables
- Produces: `google_site_verification` and `google_analytics_id` Sphinx template-context values

- [ ] **Step 1: Establish the absent-value baseline**

Run a clean Sphinx build without either environment variable and search the
generated homepage for `google-site-verification`, `googletagmanager`, and
`gtag(`. Expected: Sphinx succeeds and the search returns no matches.

- [ ] **Step 2: Add the environment-backed template context**

Extend `html_context` in `conf.py` with stripped values from
`GOOGLE_SITE_VERIFICATION` and `GOOGLE_ANALYTICS_ID`, defaulting to empty
strings.

- [ ] **Step 3: Add conditional head markup**

Extend `_templates/layout.html` with an `extrahead` block. Emit the Search
Console meta tag only when `google_site_verification` is non-empty. Emit the
async GA4 loader and initialization snippet only when `google_analytics_id` is
non-empty.

- [ ] **Step 4: Verify absent and present builds**

Build without variables and confirm no Google markup is generated. Build with
sample values and confirm the generated homepage contains each sample exactly
where expected. Both builds must complete without warnings.

### Task 2: GitHub Pages configuration

**Files:**
- Modify: `.github/workflows/build_docs.yaml`

**Interfaces:**
- Consumes: `secrets.GOOGLE_SITE_VERIFICATION` and `vars.GOOGLE_ANALYTICS_ID`
- Produces: build-step environment variables consumed by Sphinx

- [ ] **Step 1: Pass repository configuration to Sphinx**

Add a step-level `env` mapping to `Build docs with warnings as errors`, mapping
the repository secret and variable to the names consumed by `conf.py`.

- [ ] **Step 2: Verify repository checks**

Run tutorial drift validation, example compatibility validation, example
compilation, and a warning-as-error Sphinx build. Expected: all commands exit
zero.

- [ ] **Step 3: Configure the repository secret**

Set `GOOGLE_SITE_VERIFICATION` using authenticated GitHub CLI input without
printing its value. Do not set `GOOGLE_ANALYTICS_ID` until a GA4 measurement ID
is available.

- [ ] **Step 4: Commit and publish for review**

Commit only the design, plan, Sphinx configuration, template, and Pages workflow
changes. Push `codex/google-search-integration` and open a ready pull request.
