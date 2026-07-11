# System overview

Kubetorch has a local Python client and a small control plane installed in a
Kubernetes cluster.

```text
Python driver
  ├─ Compute / Image / Secret / Volume specifications
  ├─ Fn and Cls proxies
  └─ kt run and kt runs
            │
            ▼
Kubetorch controller ── Kubernetes API
            │              ├─ Deployments / Knative Services
            │              ├─ RayClusters / distributed workers
            │              └─ Jobs for batch runs
            ▼
Kubetorch data store
  ├─ synchronized source
  ├─ run logs and artifacts
  └─ application data
```

## Client

The client turns Python resource objects into controller requests, maintains
local configuration, performs source synchronization, opens port forwards when
needed, and exposes remote callables as local proxies.

## Controller

The controller creates and discovers Kubernetes resources. It tracks callable
services and durable batch-run records. In this fork the controller state is
SQLite-backed and mounted on a persistent volume, so chart upgrades must retain
the controller PVC and use a storage-safe deployment strategy.

## Runtime service

Each remote callable runs a Kubetorch HTTP service inside the selected image.
The service imports synchronized source, invokes the requested function or class
method, streams logs, and serializes the result or exception.

## Data store

The data store is a namespace-scoped key/value file service used by source sync,
batch-run evidence, and user code. The `kt://NAMESPACE/KEY` URI identifies data
stored there without pretending it is a local filesystem path.

## Kubernetes remains visible

Kubetorch does not hide the cluster. Namespaces, Services, Deployments, Jobs,
PVCs, scheduling constraints, autoscalers, and GPU resources remain ordinary
Kubernetes objects that platform teams can inspect and govern.
