# Kubetorch Light-Only Documentation Theme Design

## Objective

Remove dark mode from the Kubetorch documentation. Every visit renders the
existing light theme regardless of operating-system preference or a previously
saved theme choice. The color-mode control is not rendered.

The first delivery remains a local preview. Publishing requires a separate user
decision after visual review.

## Implementation

Use the pinned Sphinx theme's template extension points rather than patching its
installed package or duplicating dark-theme styles:

1. Set the PyData theme context's `default_mode` to `light`. This gives the HTML
   document a light theme before JavaScript runs and provides a light fallback
   when JavaScript is unavailable.
2. Add a project `layout.html` override that runs before the inherited theme
   bootstrap. It replaces any origin-scoped saved `mode` and `theme` values with
   `light`, then lets the pinned theme initialize normally.
3. Add an empty project `theme-switcher.html` override. Sphinx Book Theme keeps
   its standard article controls but emits no color-mode button.

The existing Kubetorch stylesheet, homepage design, documentation structure,
syntax highlighting, and light-theme colors remain unchanged.

## Behavior and accessibility

- A first visit in an operating system configured for dark mode renders light.
- A returning browser with `mode=dark`, `theme=dark`, or `mode=auto` saved for
  the local or published origin is reset to light before first paint.
- JavaScript-disabled rendering remains light through the document's initial
  theme attribute.
- The inaccessible dark state is not exposed through a hidden or keyboard-
  reachable control.
- Print styling and responsive navigation remain owned by the upstream theme.

## Verification

- Add configuration tests for a light default and the project template
  overrides.
- Build all Sphinx pages with warnings treated as errors.
- Assert generated HTML contains the light default and contains no
  `theme-switch-button` or `Color mode` control.
- In the local browser, preseed dark saved preferences, reload, and verify the
  computed theme is light.
- Inspect the homepage, tutorial cards, MNIST code, GRPO callouts, API pages,
  navigation, and footer at desktop and mobile widths.
- Confirm the existing light layout has no horizontal overflow.

No branch push, pull request, or GitHub Pages deployment occurs until the user
approves the local result.
