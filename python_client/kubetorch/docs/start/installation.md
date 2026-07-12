# Install Kubetorch

Kubetorch has two pieces: a Python client in the process where you write or
orchestrate work, and a Helm release in the Kubernetes cluster where work runs.
It runs on an existing Kubernetes or k3s cluster, including Amazon EKS, Azure
AKS, Google GKE, and k3s homelabs.

## Prerequisites

- Python 3.9 or newer, `uv` or `pip`, and a modern `rsync`.
- `kubectl` configured for the target cluster.
- Helm 3 and a cluster storage class.
- Cluster GPU support only when workloads request NVIDIA GPUs.

macOS ships an old `rsync`; install a current version with `brew install rsync`.

### NVIDIA GPU clusters

The NVIDIA GPU Operator is not included in the Kubetorch chart. NVIDIA's
[GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html)
is the most complete option for a new GPU cluster because it can manage the
drivers, NVIDIA Container Toolkit, Kubernetes device plugin, and DCGM exporter.

The Kubetorch chart includes only the standalone NVIDIA device plugin and DCGM
exporter dependencies. Their defaults are:

- `nvidia-device-plugin.enabled=true`
- `dcgm-exporter.enabled=false`

Choose one owner for the GPU components:

- **GPU Operator or another cluster-level GPU stack:** install that first, then
  disable Kubetorch's bundled device plugin and DCGM exporter to avoid duplicate
  DaemonSets.
- **Manually prepared GPU nodes:** ensure compatible NVIDIA drivers and
  container runtime configuration already work on every GPU node. Kubetorch can
  then install its bundled device plugin, but it will not configure the host
  drivers and container runtime for you.

CPU-only clusters do not need any NVIDIA components; disable the bundled device
plugin with `--set nvidia-device-plugin.enabled=false`.

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

If your cluster already operates the NVIDIA GPU Operator, device plugin, or
DCGM exporter, disable the overlapping bundled dependencies:

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
