# Contributing to Kubetorch
Please file an [issue](https://github.com/cezarc1/kubetorch/issues) if you encounter a bug.

If you would like to submit a bug-fix or improve an existing feature, please submit a pull request following the
process outlined below.

## Development Process
If you want to modify code, please follow the instructions for creating a Pull Request.

1. Fork the Github repository, and then clone the forked repo to local.
```
git clone git@github.com:<your-gh-username>/kubetorch.git
cd kubetorch
git remote add upstream https://github.com/cezarc1/kubetorch.git
```

2. Create a new branch for your development changes:
```
git checkout -b branch-name
```

3. Install Kubetorch
```
cd python_client
pip install -e .
```

4. Develop your features

5. Download and run pre-commit to automatically format your code using black and ruff.

```
pip install pre-commit
pre-commit run --files [FILES [FILES ...]]
```

6. Add, commit, and push your changes. Create a "Pull Request" on GitHub to submit the changes for review.

```
git push -u origin branch-name
```

## Testing
To run tests, install the dev dependencies:
```
cd python_client
pip install -e ".[dev]"
```

## Documentation
Docs source code is located in `python_client/kubetorch/docs/`. To build and review docs locally:

```
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

Open `python_client/kubetorch/docs/_build/html/index.html` to review the site.
When a literate example changes, run
`python scripts/docs/render_tutorials.py --write` and commit the regenerated
MyST page with the source change.

### Examples
Example code for this fork lives under `examples/` in this repository. Please follow the process above to create pull requests.
