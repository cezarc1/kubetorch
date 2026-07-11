# Quickstart

This quickstart verifies both Kubetorch execution styles: an interactive remote
function and an inspectable batch run.

## 1. Dispatch a function

Create `hello.py`:

```python
import kubetorch as kt


def cluster_identity():
    import os
    import socket

    return {
        "host": socket.gethostname(),
        "namespace": os.getenv("KT_NAMESPACE"),
    }


if __name__ == "__main__":
    compute = kt.Compute(cpus="1", memory="2Gi")
    remote_identity = kt.fn(cluster_identity).to(compute)
    print(remote_identity())
```

Run it from the environment where you installed the client:

```bash
python hello.py
```

The first dispatch creates cluster resources and syncs source. Re-running after
changing `cluster_identity` reuses compute and synchronizes the update.

## 2. Record a batch run

Create a small run image from the fork or use a workload image that contains
Python, the Kubetorch client, and `rsync`. Then submit a source snapshot:

```bash
kt run \
  --name hello-batch \
  --intent "Verify source, command, logs, and run history" \
  --namespace kubetorch \
  --image ghcr.io/cezarc1/kubetorch:0.5.2 \
  --source-dir . \
  -- \
  python hello.py
```

Use the printed run ID to inspect durable evidence:

```bash
kt runs list --namespace kubetorch
kt runs show RUN_ID
kt runs logs RUN_ID
kt runs note add RUN_ID "Quickstart inspected." --author "$USER"
kt runs artifact list RUN_ID
```

## 3. Clean up deliberately

Remote callable resources and batch records have separate lifecycles:

```bash
kt teardown cluster-identity --yes
kt runs delete RUN_ID --dry-run
kt runs delete RUN_ID --yes
```

The dry run shows the run record, Job, source snapshot, logs, and standard
`kt://` artifacts that deletion will remove.

Next, follow {doc}`../tutorials/training/mnist` for model training or
{doc}`../guides/batch_runs` for the complete evidence workflow.
