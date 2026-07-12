# Google Search Verification and Analytics Design

## Goal

Verify the published Kubetorch documentation with Google Search Console without
committing the verification token, and prepare the site for optional Google
Analytics 4 tracking without committing the measurement ID.

## Design

The documentation build reads two optional environment variables:

- `GOOGLE_SITE_VERIFICATION` contains the Search Console verification token.
- `GOOGLE_ANALYTICS_ID` contains a GA4 measurement ID such as `G-XXXXXXXXXX`.

Sphinx passes these values to the existing `layout.html` override. The template
adds the verification meta tag and GA4 scripts only when their corresponding
values are present. Local builds and pull-request builds therefore remain free
of production verification and analytics markup.

The GitHub Pages workflow supplies `GOOGLE_SITE_VERIFICATION` from a repository
secret and `GOOGLE_ANALYTICS_ID` from a repository variable. Neither value is
stored in Git. Their appearance in deployed HTML is expected: Google must read
the verification token, and browsers must receive the GA4 measurement ID.

## Deployment Behavior

Only the documentation build step receives the GitHub secret and variable. A
missing value is valid and simply disables that integration. Fork pull requests
continue building successfully because their unavailable secrets resolve to an
empty value.

Search Console verification remains independent of Analytics. Removing or
changing the GA4 property will not remove the verification tag.

## Validation

Build the documentation once with neither environment variable and confirm the
generated homepage contains no Google verification or analytics markup. Build
again with harmless sample values and confirm the expected meta tag and GA4
loader appear. Keep the existing warning-as-error Sphinx build as the primary
documentation quality gate; do not add a documentation test suite.
