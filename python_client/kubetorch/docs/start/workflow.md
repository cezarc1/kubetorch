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
