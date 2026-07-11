# Distributed workloads

Kubetorch can create PyTorch and Ray worker groups while keeping the launch and
call interface in Python.

## PyTorch distributed

```python
compute = kt.Compute(gpus="1", image=image).distribute(
    "pytorch",
    workers=4,
)
remote_train = kt.fn(train).to(compute)
results = remote_train(config)
```

The runtime provides rank and world-size context. Training code is still
responsible for initializing its framework correctly, partitioning data, and
ensuring only the appropriate rank writes shared outputs.

## Ray

Use Ray when a workload needs Ray actors, tasks, datasets, Tune, or an existing
Ray-native library:

```python
ray_compute = kt.Compute(gpus="1", image=ray_image).distribute("ray", workers=4)
```

Kubetorch provisions the cluster and dispatches the entrypoint; Ray controls its
internal scheduling.

## Topology and storage

Multi-node GPU jobs depend on network topology, NCCL configuration, compatible
drivers, and storage that can be mounted by the required workers. Express node
labels, affinity, and tolerations in compute configuration rather than assuming
all GPUs are interchangeable.

## Dynamic membership

Elastic or resizable training requires algorithm-level checkpoint and sampler
logic. The {doc}`../tutorials/fault-tolerance/dynamic-world-size` examples show
how membership changes are surfaced; they are not a guarantee that an arbitrary
training loop can change world size safely.
