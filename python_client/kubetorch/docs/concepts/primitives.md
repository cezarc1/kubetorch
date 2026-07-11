# Core primitives

## Compute

`kt.Compute` describes resources and deployment behavior. It is immutable in
intent but chainable in use:

```python
compute = (
    kt.Compute(cpus="4", memory="16Gi", gpus="1", image=image)
    .distribute("pytorch", workers=4)
)
```

Distribution and autoscaling change the Kubernetes workload type; they do not
change the Python function being sent.

## Image

`kt.Image` begins with an existing container image and records additional setup
steps such as packages, commands, environment variables, and secrets. Setup
layers are cached so dependency changes do not require rewriting application
logic.

```python
image = (
    kt.Image("nvcr.io/nvidia/pytorch:25.04-py3")
    .pip_install(["transformers==4.56.1", "datasets==4.1.0"])
    .env({"TOKENIZERS_PARALLELISM": "false"})
)
```

## Fn and Cls

`kt.fn` wraps a function; `kt.cls` wraps a class. `.to(compute)` deploys the
callable and returns a proxy. A class proxy preserves remote instance state
between method calls while its service remains available.

## App

`kt.app` groups decorated compute and callable definitions for declarative
deployment from a module or CLI entrypoint.

## Batch run

`kt run` snapshots a source directory, creates a Kubernetes Job, and stores a
run record. The Python helpers `kt.note`, `kt.artifact`, and `kt.put` let code
publish evidence while it executes.

API signatures live in {doc}`../api/python`; this guide focuses on how the
objects fit together.
