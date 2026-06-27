# Kubetorch

**Agent-friendly ML batch runs and Pythonic remote execution on Kubernetes**

Originally built by [Runhouse](https://www.run.house), which has [as of March
2026 shut down](https://www.linkedin.com/posts/greenbergdon_im-excited-to-share-that-the-runhouse-team-share-7453528259448860673-2n_r/).

This fork of the original KubeTorch focuses on making ML runs on Kubernetes reproducible and agent-first. For example, each batch run can capture the exact source snapshot, sanitized environment, intent, start time, logs, notes, and artifact references so humans and coding agents can inspect what happened after the container exits.

Kubetorch still supports the upstream Pythonic remote execution model: bring cluster compute into notebooks, IDEs, CI, or production code without rewriting your workload around a DAG system.

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

## Batch Run Quickstart

```bash
kt runs list --namespace kubetorch

kt run \
  --name nanogpt-baseline \
  --intent "GPT-2 124M baseline smoke on cluster GPU" \
  --namespace kubetorch \
  --image ghcr.io/cezarc1/nanogpt-kubetorch:dev \
  --source-dir . \
  --resources '{"requests":{"cpu":"2","memory":"12Gi","nvidia.com/gpu":"1"},"limits":{"nvidia.com/gpu":"1"}}' \
  -- \
  uv run python train.py config/train_shakespeare_char.py

kt runs show RUN_ID
kt runs logs RUN_ID
kt runs artifact list RUN_ID
kt runs note add RUN_ID "Result note: baseline completed; inspect val loss" --author agent
```

Use batch runs for SFT, RL, evals, data preprocessing, and long GPU jobs where
the outcome needs to be explainable later.

## Python Remote Execution

```python
import kubetorch as kt

def hello_world():
    return "Hello from Kubetorch!"

if __name__ == "__main__":
    # Define your compute
    compute = kt.Compute(cpus=".1")

    # Send local function to freshly launched remote compute
    remote_hello = kt.fn(hello_world).to(compute)

    # Runs remotely on your Kubernetes cluster
    result = remote_hello()
    print(result)  # "Hello from Kubetorch!"
```

## Installation

### 1. Python Client

```bash
pip install "kubetorch[client]"
```

For local development from this checkout:

```bash
cd python_client
pip install -e ".[client]"
```

### 2. Kubernetes Deployment With Helm

The fork-owned chart is published as a public OCI Helm chart. The chart
itself can be fetched anonymously:

```bash
helm upgrade --install kubetorch oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.0 \
  -n kubetorch --create-namespace
```

Kubernetes will then pull the controller, data-store, and default workload
images from `ghcr.io/cezarc1`. Unauthenticated installs only work if those image
packages are public too. If any image package is still private, authenticate
first:

```bash
echo "$GHCR_TOKEN" | helm registry login ghcr.io -u cezarc1 --password-stdin
kubectl create secret docker-registry ghcr-pull-secret \
  --namespace kubetorch \
  --docker-server=ghcr.io \
  --docker-username=cezarc1 \
  --docker-password="$GHCR_TOKEN"
```

Then pass the pull secret to the chart:

```bash
helm upgrade --install kubetorch oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.0 \
  -n kubetorch --create-namespace \
  --set kubetorchConfig.imagePullSecrets[0].name=ghcr-pull-secret
```

For homelab installs where the NVIDIA device plugin and DCGM exporter are
already managed elsewhere:

```bash
helm upgrade --install kubetorch oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.0 \
  -n kubetorch --create-namespace \
  --set nvidia-device-plugin.enabled=false \
  --set dcgm-exporter.enabled=false \
  --set kubetorchConfig.deployment_namespaces[0]=kubetorch
```

### 3. Upgrade The Fork

Build and publish the fork-owned images:

```bash
gh auth token | docker login ghcr.io -u cezarc1 --password-stdin

IMAGE_NAMESPACE=ghcr.io/cezarc1 DOCKER_PLATFORMS=linux/amd64,linux/arm64 \
  release/build_images.sh --version 0.5.0 --all --push
```

Publish the matching chart after the images are available:

```bash
GHCR_TOKEN="$(gh auth token)" GHCR_USERNAME=cezarc1 \
  release/publish_chart.sh --version 0.5.0
```

Upgrade the cluster to the same tag:

```bash
helm upgrade kubetorch oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.0 \
  -n kubetorch
```

Use immutable tags for controller, data-store, and workload images when testing
agent-first batch-run behavior.

## Documentation

```bash
pip install -r python_client/kubetorch/docs/requirements.txt
cd python_client/kubetorch/docs && make html
open _build/html/index.html
```

Published fork docs are expected at
[`cezarc1.github.io/kubetorch`](https://cezarc1.github.io/kubetorch/) after the
GitHub Pages workflow runs on `main`.

## Source Layout

This repo now includes the customer-facing OSS deployment components that were
previously split across internal and OSS repos:

- `python_client/` for the SDK
- `charts/kubetorch/` for the Helm chart
- `services/` for the controller and data store sources
- `release/default_images/` for the workload base images
- `release/` for release scripts and version sync

## Non-Goals

- Kubetorch is not an experiment tracker replacement; it records run evidence
  and artifact pointers, including external systems like W&B.
- Kubetorch is not a DAG orchestrator; it favors direct Kubernetes-native batch
  jobs for many ML training and evaluation workflows.
- Kubetorch does not make ML deterministic; it captures enough context to
  understand and compare runs even when results are not reproducible.

---

[Apache 2.0 License](LICENSE)

This fork now publishes its own chart, images, and documentation under the
`cezarc1` GitHub/GHCR namespace.
