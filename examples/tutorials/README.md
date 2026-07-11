# Kubetorch tutorial catalog

This directory preserves and maintains the public examples originally published
in [`run-house/kubetorch-examples`](https://github.com/run-house/kubetorch-examples).
The initial import is pinned to commit
`6f1f962937758a61b9c16b946d4482bc862c2210`.

The examples were originally written by the Runhouse team. They are maintained
here as part of the `cezarc1/kubetorch` fork and may be adapted as the fork's API,
images, and operational model evolve. The repository-level Apache 2.0 license
and attribution apply to this maintained copy under the authorization recorded
for the documentation recovery project.

## Documentation source

Files beginning with Markdown-style comments such as `# # Title` and
`# ## Section` are rendered into the Sphinx tutorial catalog. Metadata and
validation status live in
`python_client/kubetorch/docs/_data/catalog.yaml`.

Generated tutorial pages must be refreshed after changing a literate example:

```bash
python scripts/docs/render_tutorials.py --write
python scripts/docs/render_tutorials.py --check
```

Validation labels are intentionally narrow:

- **Validated** means the example ran against the recorded fork version and
  hardware profile.
- **Adapted** means it uses the current public API and passes repository checks,
  but has not completed a recorded cluster run.
- **Reference** means it depends on external infrastructure or serves as
  supporting code rather than a standalone smoke test.
