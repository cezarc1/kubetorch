# Start Workflow Artifact Walkthrough Design

## Goal

Turn the artifact mention on `start/workflow.html` into a complete, CPU-only
batch-run walkthrough. A reader should be able to create a result, store it,
register it against a run, inspect it, retrieve it, and preview cleanup without
needing a model, GPU, dataset, or external tracking service.

## Documentation flow

Extend `python_client/kubetorch/docs/start/workflow.md` after the existing
interactive-versus-batch comparison. The new section will:

1. Explain that `kt.artifact` registers a reference and does not upload the
   referenced data.
2. Provide a complete `record_result.py` workload that:
   - requires the `KT_RUN_ID` injected by `kt run`;
   - uses `KT_NAMESPACE`, defaulting to `kubetorch`;
   - writes a small `metrics.json` file;
   - uploads it to `runs/<run-id>/artifacts/metrics.json` with `kt.put`;
   - registers `kt://<namespace>/runs/<run-id>/artifacts/metrics.json` with
     `kt.artifact`;
   - adds a concise completion note with `kt.note`.
3. Show a `kt run` command that snapshots the source and runs the script. The
   workload image will be an explicit substitution rather than an invented
   public image; it must include Python, Kubetorch, and `rsync`.
4. Show how to use the printed run ID with `kt runs show`, `kt runs logs`, and
   `kt runs artifact list`.
5. Show how to retrieve the object with `kt get` and preview deletion with
   `kt runs delete --dry-run`.

The existing Production handoff section remains after the walkthrough and
continues linking to the detailed Batch Runs and Agent Workflows guides.

## Correctness boundaries

The example must preserve the distinction between a data-store key and an
artifact URI:

- `kt.put` receives the namespace-relative key.
- `kt.artifact` receives the full `kt://<namespace>/<key>` URI.
- The upload happens before registration.
- No run ID is hard-coded; the workload reads `KT_RUN_ID`.

Failure behavior remains direct and visible. A missing `KT_RUN_ID`, failed
upload, or failed registration should terminate the workload rather than
publishing a misleading success note.

## Validation

Add a focused documentation test under `python_client/tests/docs/`. It will
extract the `record_result.py` block from the MyST page, compile it, and execute
it with a temporary working directory, injected run environment, and patched
Kubetorch side-effect functions. The test will verify:

- the metrics file is valid JSON;
- the upload key is `runs/<run-id>/artifacts/metrics.json`;
- `kt.put` precedes `kt.artifact`;
- the registered URI is the matching `kt://` URI;
- the note is recorded last;
- the page includes launch, inspection, retrieval, and dry-run cleanup commands.

Then run the focused test, the complete documentation test suite, and a
warning-as-error Sphinx HTML build. Finally, inspect the rendered local page to
confirm headings, code blocks, and navigation render correctly.

Cluster execution is a separate integration check because it requires a
configured Kubernetes cluster and a suitable workload image. The documentation
will provide the real command needed for that check without claiming the static
test launched Kubernetes.
