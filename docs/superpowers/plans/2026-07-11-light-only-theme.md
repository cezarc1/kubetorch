# Kubetorch Light-Only Theme Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Kubetorch Sphinx site render only its existing light theme and remove the color-mode control.

**Architecture:** Set the inherited PyData theme's initial mode through Sphinx context, then use project-owned template overrides to normalize origin-scoped theme storage before the upstream bootstrap and to render no switcher button. The upstream packages and existing light-theme CSS remain unchanged.

**Tech Stack:** Sphinx 6.2.1, Sphinx Book Theme 1.0.1, Jinja templates, pytest, in-app browser verification.

## Global Constraints

- Every visit renders light regardless of operating-system preference or previously saved theme choice.
- The color-mode control is not rendered or keyboard reachable.
- Existing light CSS, homepage layout, print behavior, and responsive navigation remain unchanged.
- Build with warnings treated as errors.
- Refresh only the local preview; do not push, open a pull request, or deploy Pages.

---

### Task 1: Force light initialization and remove the switcher

**Files:**
- Modify: `python_client/tests/docs/test_sphinx_config.py`
- Modify: `python_client/kubetorch/docs/conf.py`
- Create: `python_client/kubetorch/docs/_templates/layout.html`
- Create: `python_client/kubetorch/docs/_templates/theme-switcher.html`

**Interfaces:**
- Consumes: Sphinx's existing `templates_path = ["_templates"]` and PyData's `default_mode` template context.
- Produces: generated HTML initialized with `data-theme="light"`, saved light preferences, and no `.theme-switch-button`.

- [ ] **Step 1: Write failing configuration and template tests**

Add to `python_client/tests/docs/test_sphinx_config.py`:

```python
def test_documentation_forces_light_mode_before_theme_bootstrap():
    conf = (DOCS_ROOT / "conf.py").read_text()
    layout = (DOCS_ROOT / "_templates/layout.html").read_text()

    assert '"default_mode": "light"' in conf
    assert 'localStorage.setItem("mode", "light")' in layout
    assert 'localStorage.setItem("theme", "light")' in layout
    assert 'document.documentElement.dataset.theme = "light"' in layout


def test_documentation_does_not_render_a_theme_switcher():
    switcher = DOCS_ROOT / "_templates/theme-switcher.html"

    assert switcher.is_file()
    assert "theme-switch-button" not in switcher.read_text()
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q \
  python_client/tests/docs/test_sphinx_config.py --level unit
```

Expected: failures because the project templates and light context do not exist.

- [ ] **Step 3: Add the light default to Sphinx configuration**

Add beside `templates_path` in `python_client/kubetorch/docs/conf.py`:

```python
html_context = {"default_mode": "light"}
```

- [ ] **Step 4: Add the pre-bootstrap layout override**

Create `python_client/kubetorch/docs/_templates/layout.html`:

```jinja
{% extends "!layout.html" %}

{% block css %}
  <script data-cfasync="false">
    localStorage.setItem("mode", "light");
    localStorage.setItem("theme", "light");
    document.documentElement.dataset.mode = "light";
    document.documentElement.dataset.theme = "light";
  </script>
  {{ super() }}
{% endblock css %}
```

This executes before the inherited PyData mini-script, so saved dark or auto values cannot affect first paint.

- [ ] **Step 5: Override the switcher with an intentionally empty template**

Create `python_client/kubetorch/docs/_templates/theme-switcher.html`:

```jinja
{# Kubetorch intentionally provides a light-only documentation theme. #}
```

- [ ] **Step 6: Run the focused tests and verify they pass**

Run the Step 2 command.

Expected: all `test_sphinx_config.py` tests pass.

- [ ] **Step 7: Commit the light-only implementation**

```bash
git add \
  python_client/kubetorch/docs/conf.py \
  python_client/kubetorch/docs/_templates/layout.html \
  python_client/kubetorch/docs/_templates/theme-switcher.html \
  python_client/tests/docs/test_sphinx_config.py
git commit -m "docs: remove dark theme"
```

### Task 2: Prove generated and browser behavior

**Files:**
- Test: `python_client/tests/docs/test_sphinx_config.py`
- Generated locally: `/tmp/kubetorch-docs-oled-preview`

**Interfaces:**
- Consumes: the light-only templates from Task 1 and the existing local HTTP server on `127.0.0.1:8765`.
- Produces: a clickable local preview that cannot enter dark mode.

- [ ] **Step 1: Run all documentation checks**

```bash
.venv/bin/python -m pytest -q python_client/tests/docs --level unit
.venv/bin/python scripts/docs/render_tutorials.py --check
.venv/bin/python scripts/docs/check_example_compat.py
```

Expected: all commands exit successfully.

- [ ] **Step 2: Rebuild the served Sphinx output with warnings as errors**

```bash
.venv/bin/python -m sphinx -W --keep-going -b html \
  python_client/kubetorch/docs \
  /tmp/kubetorch-docs-oled-preview
```

Expected: `build succeeded` with no warnings.

- [ ] **Step 3: Verify the generated HTML contract**

```bash
rg 'data-theme="light"' /tmp/kubetorch-docs-oled-preview/index.html
! rg 'theme-switch-button|Color mode' /tmp/kubetorch-docs-oled-preview/index.html
```

Expected: the light theme attribute is present and no switcher markup is found.

- [ ] **Step 4: Verify saved dark preferences cannot reactivate dark mode**

In the local browser at `http://127.0.0.1:8765/`, set origin storage to
`mode=dark` and `theme=dark`, reload, and inspect:

```javascript
({
  mode: document.documentElement.dataset.mode,
  theme: document.documentElement.dataset.theme,
  storedMode: localStorage.getItem("mode"),
  storedTheme: localStorage.getItem("theme"),
  switchers: document.querySelectorAll(".theme-switch-button").length,
  horizontalOverflow:
    document.documentElement.scrollWidth > document.documentElement.clientWidth,
})
```

Expected: both document and stored values are `light`, `switchers` is `0`, and `horizontalOverflow` is `false`.

- [ ] **Step 5: Inspect representative pages**

Check the homepage, tutorial index, MNIST, basic GRPO, and Python API pages at desktop and mobile widths. Expected: the existing light appearance is preserved and navigation remains usable.

- [ ] **Step 6: Report the local preview for user review**

Keep the server running at `http://127.0.0.1:8765/`. Do not publish. Ask the user to approve or request further changes.
