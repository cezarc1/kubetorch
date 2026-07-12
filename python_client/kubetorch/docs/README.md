# Kubetorch documentation source

This directory is the complete source for
[`cezarc1.github.io/kubetorch`](https://cezarc1.github.io/kubetorch/). The build
does not fetch Runhouse pages, tutorial source, or video files.

## Build locally

From the repository root:

```bash
uv venv
source .venv/bin/activate
uv pip install -e "./python_client[client,docs]"
uv pip install -r python_client/kubetorch/docs/requirements.txt

python scripts/docs/render_tutorials.py --check
python scripts/docs/check_example_compat.py
python -m compileall -q examples/tutorials
python -m sphinx -W --keep-going -b html \
  python_client/kubetorch/docs \
  python_client/kubetorch/docs/_build/html
```

## Content model

- Maintained guides and API references live in this directory.
- Imported source lives under `examples/tutorials/`.
- `_data/catalog.yaml` records the pinned upstream source, recoverable Runhouse
  routes, tutorial metadata, requirements, and validation status.
- `scripts/docs/render_tutorials.py` converts literate Python comments into the
  checked-in tutorial pages.
- `scripts/docs/recover_runhouse.py` can recover a pinned Wayback page for
  editorial review. Recovery is never part of the normal site build.

## Update a tutorial

Edit the example source first. Then regenerate and validate:

```bash
python scripts/docs/render_tutorials.py --write
python scripts/docs/render_tutorials.py --check
```

Do not mark an example **Validated** without recording a real fork version,
date, hardware profile, and evidence reference in the catalog.

The manual `Documentation Tutorial Smoke` workflow targets a self-hosted runner
labelled `kubetorch`. It never runs on pushes or pull requests. The runner must
already have cluster credentials and the selected example's local driver
dependencies. Each invocation uploads a JSON evidence artifact whether the
tutorial passes or fails.
