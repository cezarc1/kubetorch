# Supporting primitives

## Secrets

Use `kt.Secret` or `kt.secret` to attach Kubernetes Secrets to compute without
putting values in synchronized source. The CLI can create provider-oriented or
custom secrets:

```bash
kt secrets create hf-token --env-var HF_TOKEN --namespace kubetorch
```

## Volumes

`kt.Volume` describes a PVC mount. The storage class determines whether a
volume can be mounted by one node or many; Kubetorch cannot turn a
`ReadWriteOnce` backend into shared storage.

```python
cache = kt.Volume(
    name="model-cache",
    size="100Gi",
    mount_path="/models",
    storage_class="fast-rwo",
)
compute = kt.Compute(gpus="1", volumes=[cache])
```

## Config

Local configuration selects defaults such as username, deployment namespace,
install namespace, controller URL, and global volumes. Inspect it before
debugging surprising resource names or namespaces:

```bash
kt config list
```

## Metrics, logs, and debug configuration

Logging, metrics, and debugger settings travel with compute or callable
configuration. They describe how the runtime emits evidence; they do not replace
cluster-level observability.
