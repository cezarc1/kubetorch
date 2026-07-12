# Kubetorch

**Agent-friendly ML batch runs and Pythonic remote execution on Kubernetes**

Originally built by [Runhouse](https://www.run.house), which has [as of March
2026 shut down](https://www.linkedin.com/posts/greenbergdon_im-excited-to-share-that-the-runhouse-team-share-7453528259448860673-2n_r/).

Kubetorch runs ML workloads on Kubernetes while keeping the evidence needed to
understand them later: source snapshots, commands, sanitized environment,
intent, start time, logs, notes, and artifact references. This fork focuses on
agent-friendly batch runs while preserving the upstream Pythonic remote
execution model for notebooks, IDEs, CI, and production code.

## What This Fork Emphasizes

- **Inspectable batch runs**: submit Kubernetes Jobs with a durable run ID,
  intent, author, source key, status, logs, notes, and artifact references.
- **Artifact-first ML workflows**: register checkpoints, metrics, data
  manifests, W&B/TensorBoard links, and other result references with copyable
  `kt://...` URIs.
- **Agent-ready history**: list prior runs before launching the next one, then
  let an agent read exact code, environment, logs, notes, and output artifacts.
- **Kubernetes-native execution**: use the cluster scheduler and storage you
  already operate rather than forcing every experiment into a DAG orchestrator.

## Quick Start

Check out the [Docs](https://cezarc1.github.io/kubetorch/).

Prerequisites:

- a working Kubernetes context in `kubectl`;
- `helm`, `uv`, and Python 3.9+ on your local machine;
- a cluster storage class for the Kubetorch data store;
- NVIDIA cluster support only if you plan to request GPUs.

Kubetorch is intended to work on standard Kubernetes distributions, including
local k3s clusters and managed AKS, EKS, and GKE clusters. Cluster-specific setup
usually comes down to storage classes, GPU drivers/device plugins, ingress, and
image pull secrets.

Install the Python client from this checkout so you are using this fork:

```bash
git clone https://github.com/cezarc1/kubetorch.git
cd kubetorch
uv venv
source .venv/bin/activate
uv pip install -e "./python_client[client]"
```

Install the Kubetorch control plane:

```bash
helm upgrade --install kubetorch oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.2 \
  -n kubetorch --create-namespace

kt check --namespace kubetorch
kt runs list --namespace kubetorch
```

If your cluster already manages the NVIDIA device plugin and DCGM exporter,
disable the bundled chart dependencies:

```bash
helm upgrade --install kubetorch oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.2 \
  -n kubetorch --create-namespace \
  --set nvidia-device-plugin.enabled=false \
  --set dcgm-exporter.enabled=false \
  --set kubetorchConfig.deployment_namespaces[0]=kubetorch
```

Run a small evidence-capture smoke test:

```bash
mkdir -p /tmp/kt-smoke
cd /tmp/kt-smoke

cat > smoke.py <<'PY'
import json
import os
from pathlib import Path

import kubetorch as kt

namespace = os.environ.get("KT_NAMESPACE", "kubetorch")
run_id = os.environ["KT_RUN_ID"]

Path("results").mkdir(exist_ok=True)
metrics_path = Path("results/metrics.json")
metrics_path.write_text(json.dumps({"ok": True, "run_id": run_id}) + "\n")

artifact_key = f"runs/{run_id}/artifacts/metrics.json"
kt.put(key=artifact_key, src=metrics_path, namespace=namespace, force=True)
kt.artifact(
    name="metrics",
    kind="json",
    uri=f"kt://{namespace}/{artifact_key}",
    author="agent",
)
kt.note("Smoke run completed and metrics artifact registered.", author="agent")
print("hello from kubetorch")
PY

kt run \
  --name readme-smoke \
  --intent "Verify source sync, logs, notes, and artifact capture" \
  --namespace kubetorch \
  --image ghcr.io/cezarc1/kubetorch:0.5.2 \
  --source-dir . \
  -- \
  python smoke.py
```

After `kt run` prints the run ID, inspect the result:

```bash
RUN_ID=readme-smoke-...

kt runs list --namespace kubetorch
kt runs show "$RUN_ID"
kt runs logs "$RUN_ID"
kt runs artifact list "$RUN_ID"
kt runs note add "$RUN_ID" "Result note: smoke test inspected." --author agent
kt runs delete "$RUN_ID" --dry-run
```

Use `kt runs delete "$RUN_ID" --yes` when you want to delete the run record,
Kubernetes Job, source snapshot, logs, and standard `kt://` artifacts.

## Agent Playbook

Before launching a run:

1. Run `kt check --namespace kubetorch`.
2. Run `kt runs list --namespace kubetorch` to understand existing work.
3. Pick a short `--name` and a concrete `--intent`.
4. Use `--source-dir .` from the exact source tree to preserve.
5. Use immutable image tags for real experiments.

After a run starts:

1. Use `kt runs show RUN_ID` for status, command, source key, env, notes, and
   artifact references.
2. Use `kt runs logs RUN_ID` before guessing at failures.
3. Register findings with `kt runs note add RUN_ID ... --author agent`.
4. Register external outputs with `kt runs artifact add`, for example W&B,
   TensorBoard, S3, Hugging Face, or checkpoint URIs.
5. Use `kt runs delete RUN_ID --dry-run` before cleanup.

Inside a Kubetorch run container, the wrapper sets `KT_RUN_ID`, `KT_NAMESPACE`,
`KT_WORKDIR_KEY`, and `KT_LOGS_KEY`. Python code can call `kubetorch.note(...)`,
`kubetorch.artifact(...)`, and `kubetorch.put(...)` to publish run evidence while
it is still executing.

## What Gets Captured

| Evidence | Captured | Where to look |
| --- | --- | --- |
| Intent and command | yes | `kt runs list`, `kt runs show RUN_ID` |
| Source snapshot | yes | `source_key` in `kt runs show RUN_ID` |
| Environment | sanitized | `env` in `kt runs show RUN_ID` |
| Start time and status | yes | `kt runs list`, `kt runs show RUN_ID` |
| stdout/stderr logs | yes | `kt runs logs RUN_ID` |
| Result notes | yes | `kt runs note add`, `kt runs show RUN_ID` |
| Artifacts | references and `kt://` data | `kt runs artifact list RUN_ID` |
| External trackers | pointer only | artifact URI, for example W&B or TensorBoard |

Kubetorch does not make ML deterministic. It captures enough context to compare
runs and understand what changed when results are not reproducible.

## Run Image Requirements

`kt run` snapshots `--source-dir` into the Kubetorch data store, creates a
Kubernetes Job, and syncs the source into the Job workdir before running your
command. The run image must include:

- Python;
- the Kubetorch client;
- `rsync` for source and artifact sync;
- the dependencies needed by your workload.

The public fork base image is:

```text
ghcr.io/cezarc1/kubetorch:0.5.2
```

For private workload images, configure `kubetorchConfig.imagePullSecrets` on the
Helm release or pass `--image-pull-secret` to `kt run`.

## Real Examples

- `examples/wetlandbirds_shakedown/`: Visual WetlandBirds shakedown jobs for
  testing source sync, logs, notes, artifacts, and small eval outputs.
- `examples/qwen3_asr_orin/`: Jetson/Orin Qwen3-ASR profiling and export
  helpers.
- `examples/tutorials/`: the maintained Runhouse tutorial catalog, including
  MNIST, PyTorch DDP, Ray, fault tolerance, vLLM, GRPO, VeRL, Airflow, and
  Temporal examples. The corresponding guides are published in the
  [tutorial documentation](https://cezarc1.github.io/kubetorch/tutorials/).

Use these after the smoke test passes. They exercise more realistic dependency,
image, GPU, and data paths.

## Python Remote Execution

```python
import kubetorch as kt

def hello_world():
    return "Hello from Kubetorch!"

if __name__ == "__main__":
    compute = kt.Compute(cpus=".1")
    remote_hello = kt.fn(hello_world).to(compute)
    print(remote_hello())
```

## Install and Upgrade Details

### Python Client

From this checkout:

```bash
uv venv
source .venv/bin/activate
uv pip install -e "./python_client[client]"
```

If you are using a published Python package instead of a source checkout:

```bash
pip install "kubetorch[client]"
```

For this fork, install from an exact commit or matching release tag until a
published package is available:

```bash
pip install "git+https://github.com/cezarc1/kubetorch.git@main#subdirectory=python_client"
```

### Kubernetes Deployment With Helm

The chart and default images are published publicly under `ghcr.io/cezarc1`:

```bash
# Install this fork's chart and images together.
helm upgrade --install kubetorch oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.2 -n kubetorch --create-namespace

# Or download the chart locally first.
helm pull oci://ghcr.io/cezarc1/charts/kubetorch --version 0.5.2 --untar
helm upgrade --install kubetorch ./kubetorch -n kubetorch --create-namespace
```

For fork-specific install and upgrade notes, see
[`docs/kubetorch-fork-install-upgrade.md`](docs/kubetorch-fork-install-upgrade.md).

### `kt run` Images

`kt run` starts Jobs with `python -m kubetorch.run_wrapper --`, so the run image
must include the Kubetorch Python client and `kubetorch.run_wrapper`. Use a
Kubetorch runtime image or install the client into your custom training image.

For private images, authenticate the cluster with a GHCR pull secret:

```bash
echo "$GHCR_TOKEN" | helm registry login ghcr.io -u cezarc1 --password-stdin
kubectl create secret docker-registry ghcr-pull-secret \
  --namespace kubetorch \
  --docker-server=ghcr.io \
  --docker-username=cezarc1 \
  --docker-password="$GHCR_TOKEN"

helm upgrade --install kubetorch oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.2 \
  -n kubetorch --create-namespace \
  --set kubetorchConfig.imagePullSecrets[0].name=ghcr-pull-secret
```

### Publishing a New Fork Release

Build and publish the fork-owned images:

```bash
gh auth token | docker login ghcr.io -u cezarc1 --password-stdin

python release/sync_version.py 0.5.2
IMAGE_NAMESPACE=ghcr.io/cezarc1 DOCKER_PLATFORMS=linux/amd64,linux/arm64 \
  release/build_images.sh --version 0.5.2 --all --push
```

Publish the matching chart after the images are available:

```bash
GHCR_TOKEN="$(gh auth token)" GHCR_USERNAME=cezarc1 \
  release/publish_chart.sh --version 0.5.2
```

Upgrade a cluster to the same tag:

```bash
helm upgrade kubetorch oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.2 \
  -n kubetorch
```

## Troubleshooting

| Symptom | First checks |
| --- | --- |
| `kt check` fails | Verify current kube context, namespace, controller pod, and data-store pod. |
| `ImagePullBackOff` | Confirm the image tag exists and is public, or configure an image pull secret. |
| Run stays pending | Check GPU/resource requests, node capacity, taints, and storage class. |
| No run logs | Use `kt runs show RUN_ID`, inspect the Job pod, and check data-store health. |
| Source not found in job | Check `--source-dir`, `source_key`, and data-store sync errors in logs. |
| Agent lost context | Start with `kt runs list`, then `kt runs show`, logs, notes, and artifacts. |

## Documentation

Published docs: <https://cezarc1.github.io/kubetorch/>

Build docs locally:

```bash
pip install -r python_client/kubetorch/docs/requirements.txt
cd python_client/kubetorch/docs && make html
open _build/html/index.html
```

## Repository Map

This repo contains the pieces needed to build and operate this fork:

- `python_client/`: SDK and CLI;
- `charts/kubetorch/`: Helm chart;
- `services/`: controller and data-store services;
- `release/default_images/`: default workload images;
- `release/`: release scripts and version sync;
- `examples/`: runnable shakedowns and workload examples.

## Non-Goals

- Kubetorch is not an experiment tracker replacement; it records run evidence
  and artifact pointers, including external systems like W&B.
- Kubetorch is not a DAG orchestrator; it favors direct Kubernetes-native batch
  jobs for many ML training and evaluation workflows.
- Kubetorch does not make ML deterministic; it captures enough context to
  understand and compare runs even when results are not reproducible.

---

[Apache 2.0 License](LICENSE)

This fork publishes its chart, images, and documentation under the `cezarc1`
GitHub/GHCR namespace.
