# Install Kubetorch

Kubetorch has two pieces: a Python client in the process where you write or
orchestrate work, and a Helm release in the Kubernetes cluster where work runs.

## Prerequisites

- Python 3.9 or newer, `uv` or `pip`, and a modern `rsync`.
- `kubectl` configured for the target cluster.
- Helm 3 and a cluster storage class.
- Cluster GPU support only when workloads request NVIDIA GPUs.

macOS ships an old `rsync`; install a current version with `brew install rsync`.

## Install this fork's client

Install from a checkout so the client and examples match this documentation:

```bash
git clone https://github.com/cezarc1/kubetorch.git
cd kubetorch
uv venv
source .venv/bin/activate
uv pip install -e "./python_client[client]"
```

For a pinned non-editable install, use an exact release or commit:

```bash
pip install "git+https://github.com/cezarc1/kubetorch.git@0.5.2#subdirectory=python_client"
```

## Install the control plane

The chart and its controller/data-store image defaults are maintained together.
Use the forked chart rather than the former Runhouse chart:

```bash
helm upgrade --install kubetorch \
  oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.2 \
  --namespace kubetorch \
  --create-namespace
```

If your cluster already operates the NVIDIA device plugin and DCGM exporter,
disable the bundled dependencies:

```bash
helm upgrade --install kubetorch \
  oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.2 \
  --namespace kubetorch \
  --create-namespace \
  --set nvidia-device-plugin.enabled=false \
  --set dcgm-exporter.enabled=false \
  --set 'kubetorchConfig.deployment_namespaces[0]=kubetorch'
```

### Private GHCR packages

If the chart or workload images are private in your account, create an image
pull secret and tell Kubetorch to attach it:

```bash
kubectl create secret docker-registry ghcr-pull-secret \
  --namespace kubetorch \
  --docker-server=ghcr.io \
  --docker-username="$GITHUB_USER" \
  --docker-password="$GHCR_TOKEN"

helm upgrade --install kubetorch \
  oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.2 \
  --namespace kubetorch \
  --set 'kubetorchConfig.imagePullSecrets[0].name=ghcr-pull-secret'
```

## Verify the installation

```bash
kubectl -n kubetorch get pods,pvc
kt config set namespace kubetorch
kt config set install_namespace kubetorch
kt check --namespace kubetorch
```

`kt check` verifies controller connectivity, deployment readiness, source sync,
and GPU/logging support when applicable. Continue with {doc}`quickstart` once it
passes.

## Upgrade safely

Use matching immutable chart and image versions. The fork's controller keeps a
persistent SQLite database, so its chart uses a PVC-safe `Recreate` strategy.
Review the repository's
[`fork install and upgrade guide`](https://github.com/cezarc1/kubetorch/blob/main/docs/kubetorch-fork-install-upgrade.md)
before changing storage, controller strategy, or image coordinates.
