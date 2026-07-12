# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "Kubetorch"
copyright = "Runhouse Inc"
author = "the Runhouse team and Kubetorch fork maintainers"

# The full version, including alpha/beta/rc tags
import kubetorch

release = kubetorch.__version__


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_reredirects",
    "_ext.json_globaltoc",
]

autodoc_typehints_format = "short"
autodoc_default_flags = ["members", "show-inheritance"]
autodoc_member_order = "bysource"

templates_path = ["_templates"]
html_context = {"default_mode": "light"}
exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]
myst_enable_extensions = ["colon_fence", "deflist"]

markdown_http_base = "/docs/guide"
markdown_anchor_sections = True

if tags.has("json"):
    # Force simpler output format (helps CLI output)
    autodoc_typehints = "signature"  # "description"
    napoleon_use_param = True
    napoleon_use_rtype = True

    html_link_suffix = ""
    json_baseurl = "docs/"

# -- Options for HTML output -------------------------------------------------

if not tags.has("json"):
    html_theme = "sphinx_book_theme"

html_title = "Kubetorch"
html_theme_options = {
    "repository_url": "https://github.com/cezarc1/kubetorch",
    "repository_branch": "main",
    "path_to_docs": "python_client/kubetorch/docs",
    "home_page_in_toc": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "use_fullscreen_button": False,
    "footer_content_items": [],
    "show_navbar_depth": 2,
}
html_static_path = ["_static"]
html_css_files = ["css/kubetorch.css"]
html_favicon = "_static/favicon.svg"

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# Preserve useful old paths while presenting one maintained documentation set.
redirects = {
    "advanced-installation": "start/installation.html",
    "advanced-installation/v0.1.16": "../start/installation.html",
    "api/compute": "python/compute.html",
    "api/distribute": "../guides/distributed.html",
    "api/image": "python/image.html",
    "api-reference/python": "../api/python.html",
    "api-reference/python/app": "../../api/python/app.html",
    "api-reference/python/cls": "../../api/python/cls.html",
    "api-reference/python/compute": "../../api/python/compute.html",
    "api-reference/python/config": "../../api/python/config.html",
    "api-reference/python/data_store": "../../api/python/data_store.html",
    "api-reference/python/fn": "../../api/python/fn.html",
    "api-reference/python/image": "../../api/python/image.html",
    "api-reference/python/secret": "../../api/python/secret.html",
    "api-reference/python/volumes": "../../api/python/volumes.html",
    "api-reference/python/workload_configs": "../../api/python/workload_configs.html",
    "concepts": "concepts/overview.html",
    "concepts/distributed": "../guides/distributed.html",
    "concepts/monitoring-and-observability": "../guides/monitoring.html",
    "concepts/summary": "overview.html",
    "guides/summary": "../start/workflow.html",
    "hello-world": "start/quickstart.html",
    "installation": "start/installation.html",
    "introduction": "start/introduction.html",
    "kubernetes-install": "start/installation.html",
    "serverless": "start/installation.html",
}

# -- Disable "View Source" links and code display ----------------------------

html_show_sourcelink = False  # hides "View Source" link
html_copy_source = False  # prevents .html files from containing source code
