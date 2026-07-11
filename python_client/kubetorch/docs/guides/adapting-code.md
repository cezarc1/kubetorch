# Adapt existing code

Kubetorch works best when infrastructure wraps a clean Python entrypoint rather
than being threaded through model logic.

## Start with the boundary

Choose the smallest function or class that owns the expensive operation:

```python
def train(config: dict) -> dict:
    dataset = load_dataset(config["dataset"])
    model = build_model(config["model"])
    return fit(model, dataset, config)
```

Keep local argument parsing, experiment selection, and result comparison in the
driver. Send `train`, not the entire orchestration process.

## Make files discoverable

Kubetorch synchronizes source from the project root. Use normal imports and
paths relative to the synchronized project rather than laptop-specific absolute
paths. Put large datasets and checkpoints in a volume, the Kubetorch data store,
or external object storage.

## Move dependencies into the image

Start from an image that already contains the expensive runtime stack, then add
the smaller project dependencies:

```python
image = kt.Image("nvcr.io/nvidia/pytorch:25.04-py3").pip_install(
    ["transformers==4.56.1", "datasets==4.1.0"]
)
compute = kt.Compute(gpus="1", image=image)
remote_train = kt.fn(train).to(compute)
```

## Keep credentials out of arguments

Pass secret names in compute configuration and read values from the environment
inside the remote process. Do not serialize tokens as function arguments or
commit them to source.

## Return small values

Return metrics and lightweight metadata. Store checkpoints, tables, and media
as artifacts and return their keys or URIs. This makes retries cheaper and run
history more useful.

## Add evidence for one-shot work

If the entrypoint is a complete training/evaluation command, consider `kt run`
instead of a persistent callable. Record a concrete `--intent`, preserve its
source directory, and register output artifacts from inside the process.
