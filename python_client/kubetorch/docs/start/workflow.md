# Developer workflow

Kubetorch moves infrastructure choices into the same Python program as the
workload. The normal loop has four stages.

## 1. Describe compute

```python
import kubetorch as kt

gpu_workers = kt.Compute(
    cpus="8",
    memory="32Gi",
    gpus="1",
    image=kt.Image("nvcr.io/nvidia/pytorch:25.04-py3").pip_install(
        ["transformers", "datasets"]
    ),
).distribute("pytorch", workers=2)
```

The compute request can also include labels, tolerations, volumes, secrets,
autoscaling, inactivity timeouts, and image setup commands. Kubetorch translates
it into the appropriate Kubernetes resources.

## 2. Dispatch a callable

```python
remote_train = kt.fn(train).to(gpu_workers)
remote_trainer = kt.cls(Trainer).to(gpu_workers)
```

Functions and classes remain normal Python. `.to()` creates or finds the remote
service, uploads source, and returns a proxy.

## 3. Call it from the driver

```python
metrics = remote_train(epochs=3)

trainer = remote_trainer(model_id="Qwen/Qwen2.5-1.5B")
trainer.load_data("s3://my-data/train")
trainer.train(epochs=3)
```

Calls are sent over HTTP. Return values are serialized according to the
callable's allowed serialization modes; logs and remote exceptions are relayed
to the driver.

## 4. Iterate or preserve a run

Re-dispatch a callable to sync local source changes onto warm compute. For a
one-shot experiment where provenance matters more than a persistent service,
use `kt run` instead and inspect it through `kt runs`.

```text
interactive development               experiment evidence
-----------------------               -------------------
kt.fn/kt.cls → .to(compute)           kt run --intent ...
          ↓                                      ↓
  call remote methods                 Job + source + logs
          ↓                                      ↓
  edit and hot-sync                   notes + artifacts
```

## Record and inspect a batch artifact

An artifact record is a pointer, not an upload. Store the result first with
`kt.put`, then attach its stable URI to the current run with `kt.artifact`.

Create `record_result.py`:

```python
# record_result.py
import json
import os
from pathlib import Path

import kubetorch as kt


def main():
    run_id = os.environ["KT_RUN_ID"]
    namespace = os.getenv("KT_NAMESPACE", "kubetorch")
    metrics = {"accuracy": 0.98, "epochs": 3}

    metrics_path = Path("metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    artifact_key = f"runs/{run_id}/artifacts/metrics.json"
    artifact_uri = f"kt://{namespace}/{artifact_key}"

    kt.put(
        key=artifact_key,
        src=metrics_path,
        namespace=namespace,
        force=True,
    )
    kt.artifact(
        name="metrics",
        uri=artifact_uri,
        kind="kt-data-store",
        metadata={"content_type": "application/json"},
    )
    kt.note("Recorded the metrics artifact.")
    print(f"Recorded {artifact_uri}")


if __name__ == "__main__":
    main()
```

Submit it with a workload image containing Python, Kubetorch, and `rsync`. The
shell guard stops immediately with that requirement if the variable is unset:

```bash
: "${KT_WORKLOAD_IMAGE:?Set KT_WORKLOAD_IMAGE to a pullable image containing Python, Kubetorch, and rsync}"

kt run \
  --name artifact-workflow \
  --intent "Record and retrieve a metrics artifact" \
  --namespace kubetorch \
  --image "$KT_WORKLOAD_IMAGE" \
  --source-dir . \
  -- \
  python record_result.py
```

`kt run` prints the submitted run ID. Replace `RUN_ID` below with that value to
inspect the run and its registered references:

```bash
kt runs show RUN_ID
kt runs logs RUN_ID
kt runs artifact list RUN_ID
```

The artifact list prints
`kt://kubetorch/runs/RUN_ID/artifacts/metrics.json`. Its data-store key is the
portion after the namespace, so the result can be restored independently of
the run record:

```bash
kt get runs/RUN_ID/artifacts/metrics.json ./retrieved-metrics.json \
  --namespace kubetorch
```

When the evidence is no longer needed, preview cleanup before deleting it:

```bash
kt runs delete RUN_ID --dry-run
```

The detailed {doc}`../guides/batch_runs` guide covers external artifact URIs,
operator-added references, and deletion behavior.

## Production handoff

Because the driver is ordinary Python, the same program can run from CI,
Airflow, Temporal, or another service. Production hardening usually means:

- pinning images and Python dependencies;
- giving callables stable names and namespaces;
- moving credentials into Kubernetes Secrets;
- selecting explicit storage and resource limits;
- recording batch intent and external artifacts;
- testing teardown, preemption, and retry paths.

See {doc}`../guides/dev-prod` and {doc}`../guides/agent-workflows` for those
handoffs.
