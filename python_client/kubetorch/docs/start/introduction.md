# What is Kubetorch?

Kubetorch is a Python interface for running ordinary functions and classes on
Kubernetes compute. You describe resources and dependencies in Python, send a
callable to that compute, and invoke it from your existing process as though it
were local.

```python
import kubetorch as kt

def hello_world():
    return "hello from Kubernetes"

compute = kt.Compute(cpus="1", memory="2Gi")
remote_hello = kt.fn(hello_world).to(compute)
print(remote_hello())
```

This model works from a notebook, IDE, test runner, CI job, orchestrator, or
production service. The same Python controls single pods, autoscaled services,
PyTorch workers, and Ray clusters.

## The core loop

1. **Describe compute.** Select CPUs, memory, GPUs, an image, secrets, volumes,
   distribution, and scaling behavior.
2. **Dispatch code.** Wrap a function with `kt.fn` or a class with `kt.cls`,
   then call `.to(compute)`.
3. **Call it normally.** Kubetorch handles the remote request, log streaming,
   errors, and returned value.
4. **Iterate.** Update local code and dispatch again; Kubetorch syncs the
   changed source without forcing a complete image rebuild.

## Two execution styles

**Remote callables** keep compute available while you call functions or class
methods interactively. They are a good fit for model services, distributed
training actors, notebooks, and composite systems such as RL.

**Batch runs** submit a Kubernetes Job and preserve an evidence record: intent,
command, source snapshot, sanitized environment, start time, logs, notes, and
artifact references. They are a good fit for reproducible experiments and
agent-operated workflows. See {doc}`../guides/batch_runs`.

## What this fork adds

This fork keeps the original Pythonic execution model and adds an inspectable
run layer designed for iterative ML work:

| Capability | Remote callables | Batch runs |
| --- | --- | --- |
| Persistent compute | Yes | One Job per run |
| Hot source sync | Yes | Source snapshot per run |
| Streamed logs and errors | Yes | Yes, with durable log references |
| Intent and author | Application-defined | First-class run metadata |
| Notes and artifact references | Application-defined | First-class APIs and CLI |
| Agent-friendly history | Service discovery | `kt runs list/show/logs` |

Kubetorch does not replace PyTorch, Ray, vLLM, VeRL, Airflow, or your model
code. It gives those systems a consistent way to acquire Kubernetes compute and
be called from normal Python.

## Where to go next

- {doc}`installation` installs the client and control plane.
- {doc}`quickstart` dispatches a first callable and records a batch run.
- {doc}`../tutorials/training/mnist` starts with a familiar training example.
- {doc}`../tutorials/reinforcement-learning/basic-grpo` shows heterogeneous RL
  services coordinated from one driver.
