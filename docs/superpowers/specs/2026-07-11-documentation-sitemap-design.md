# Documentation Sitemap Design

## Goal

Publish a generated XML sitemap at
`https://cezarc1.github.io/kubetorch/sitemap.xml` with canonical URLs for the
Kubetorch GitHub Pages project site.

## Design

Use the `sphinx-sitemap` extension during the existing Sphinx HTML build. Set
Sphinx's `html_baseurl` to `https://cezarc1.github.io/kubetorch/` and configure
the extension to build URLs directly from each output link. This makes Sphinx
the single source of truth for documentation routes and also adds canonical
links to rendered pages.

The sitemap is a generated Pages artifact and is not committed. Redirect-only
compatibility pages may appear because they are real HTML outputs; source pages,
build internals, and non-HTML assets must not appear.

## Validation

Build the complete documentation with warnings treated as errors. Confirm that
`sitemap.xml` exists, parses as XML, includes the homepage and representative
tutorial/API routes under the `/kubetorch/` prefix, and contains no URLs outside
that public base. Keep the existing repository checks and do not add a separate
documentation test suite.
