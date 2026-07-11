# Autoscaling

Autoscaling changes a callable service from a fixed deployment into a set of
replicas that responds to demand.

```python
compute = kt.Compute(
    cpus="4",
    memory="16Gi",
    gpus="1",
    image=image,
).autoscale(
    min_scale=0,
    max_scale=4,
    concurrency=8,
)
```

## Choose a useful concurrency target

The target should match the workload rather than the HTTP server. A GPU model
that already performs continuous batching may need low request concurrency per
replica; a lightweight CPU function can tolerate much more.

## Scale to zero

`min_scale=0` releases idle compute but makes the next request pay cold-start,
image-pull, and model-load time. Keep one warm replica when latency matters or
when initialization is expensive.

## Capacity still comes from Kubernetes

An autoscaler can request replicas but cannot create GPUs that the cluster does
not have. Node autoscaling, taints, topology, quotas, PVC access modes, and image
pull permissions can all limit successful scale-out.

## Observe before tuning

Measure queue time, cold-start time, service concurrency, GPU utilization, and
error rates together. Raising `max_scale` without resolving a scheduling or
storage bottleneck only creates more pending pods.
