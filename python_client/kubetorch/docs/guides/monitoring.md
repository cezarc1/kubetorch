# Monitoring and observability

Observability spans three layers: the driver, the Kubetorch service or Job, and
the Kubernetes cluster.

## Driver and callable logs

Remote stdout, stderr, and exceptions stream back to the driver. `kt logs`
retrieves service logs after the fact. Configure structured application logging
inside the workload image when logs feed a cluster collector.

## Batch evidence

`kt runs show` combines intent, command, source key, sanitized environment,
status, notes, and artifact references. `kt runs logs` retrieves the captured
run log without requiring the original driver process.

## Cluster signals

Use Kubernetes events and your existing metrics stack for scheduling, node,
container, network, and GPU signals. If the cluster already manages the NVIDIA
device plugin or DCGM exporter, disable the duplicate chart dependencies.

## External trackers

Weights & Biases, TensorBoard, MLflow, and object-storage outputs remain useful.
Register their stable URL or object URI as a run artifact so Kubetorch history
connects infrastructure evidence to model evidence without copying the tracker.
