# Start Workflow Artifact Walkthrough Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an executable, end-to-end artifact registration walkthrough to the Start Here developer workflow page.

**Architecture:** The MyST page owns the example source so readers and tests exercise the same code. A focused pytest extracts that code block, runs it with a fake Kubetorch module, and verifies file creation plus API call ordering without requiring Kubernetes; Sphinx then verifies the rendered documentation structure.

**Tech Stack:** MyST Markdown, Python 3, pytest, Sphinx

## Global Constraints

- Preserve the user's unrelated uncommitted `python_client/kubetorch/docs/index.md` edit.
- Use the `KT_RUN_ID` and `KT_NAMESPACE` environment variables injected by `kt run`.
- Upload the namespace-relative key with `kt.put` before registering its full `kt://` URI with `kt.artifact`.
- Use an explicit `YOUR_WORKLOAD_IMAGE` substitution rather than claiming a workload image is publicly available.
- Do not launch a Kubernetes Job as part of the ordinary documentation test suite.

---

### Task 1: Executable artifact workflow

**Files:**
- Create: `python_client/tests/docs/test_start_workflow_docs.py`
- Modify: `python_client/kubetorch/docs/start/workflow.md`

**Interfaces:**
- Consumes: `kt.put(key=..., src=..., namespace=..., force=True)`, `kt.artifact(name=..., uri=..., kind=..., metadata=...)`, and `kt.note(body)`.
- Produces: a `record_result.py` documentation block and a focused executable documentation test.

- [ ] **Step 1: Write the failing documentation test**

Create a test that extracts the Python fence beginning with `# record_result.py`, compiles it, executes it with `KT_RUN_ID=demo-run` and `KT_NAMESPACE=kubetorch`, and records calls through a fake `kubetorch` module. Assert that:

```python
assert [event[0] for event in events] == ["put", "artifact", "note"]
assert events[0][1]["key"] == "runs/demo-run/artifacts/metrics.json"
assert events[1][1]["uri"] == "kt://kubetorch/runs/demo-run/artifacts/metrics.json"
assert json.loads((tmp_path / "metrics.json").read_text()) == {
    "accuracy": 0.98,
    "epochs": 3,
}
```

Also assert the page contains `kt run`, `kt runs show RUN_ID`, `kt runs logs RUN_ID`, `kt runs artifact list RUN_ID`, `kt get`, and `kt runs delete RUN_ID --dry-run`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q --level unit python_client/tests/docs/test_start_workflow_docs.py
```

Expected: FAIL because the page does not yet contain the identified Python block or end-to-end commands.

- [ ] **Step 3: Add the minimal walkthrough**

Extend `start/workflow.md` with a section that creates `record_result.py` using this data flow:

```python
run_id = os.environ["KT_RUN_ID"]
namespace = os.getenv("KT_NAMESPACE", "kubetorch")
artifact_key = f"runs/{run_id}/artifacts/metrics.json"
artifact_uri = f"kt://{namespace}/{artifact_key}"
kt.put(key=artifact_key, src=metrics_path, namespace=namespace, force=True)
kt.artifact(
    name="metrics",
    uri=artifact_uri,
    kind="kt-data-store",
    metadata={"content_type": "application/json"},
)
kt.note("Recorded the metrics artifact.")
```

Add launch, inspection, retrieval, and dry-run cleanup commands. Explain that `kt.put` stores bytes while `kt.artifact` records the reference.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q --level unit python_client/tests/docs/test_start_workflow_docs.py
```

Expected: `2 passed`.

- [ ] **Step 5: Run repository documentation validation**

Run:

```bash
.venv/bin/python -m pytest -q --level unit python_client/tests/docs
.venv/bin/python -m sphinx -W --keep-going -b html python_client/kubetorch/docs /tmp/kubetorch-docs-oled-preview
```

Expected: all documentation tests pass and Sphinx exits successfully without warnings.

- [ ] **Step 6: Verify the rendered page**

Reload `http://127.0.0.1:8765/start/workflow.html` and confirm the new section, Python block, shell commands, and Production handoff render in the intended order.

- [ ] **Step 7: Commit the scoped implementation**

```bash
git add python_client/tests/docs/test_start_workflow_docs.py python_client/kubetorch/docs/start/workflow.md
git commit -m "docs: add artifact workflow walkthrough"
```
