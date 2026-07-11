# Call modes

Remote callables support synchronous, asynchronous, and distributed calls.

## Synchronous

The default call blocks the driver until the remote method returns or raises.

```python
result = remote_train(epochs=3)
```

## Asynchronous

Set `async_` on the proxy when the driver should overlap independent calls:

```python
remote_model.async_ = True
pending = [remote_model.generate(prompt) for prompt in prompts]
results = await asyncio.gather(*pending)
```

Use async mode for concurrency in the driver; autoscaling controls concurrency
and replica count in the service.

## Distributed

Calls to a distributed service may be vectorized across workers. The return
shape therefore depends on the distribution backend and worker selection.
PyTorch examples often use `workers=[0]` when an operation should execute only
on the rank-zero process.

## Serialization

JSON is the safest portable return format. Pickle is useful for Python-specific
objects but must only be enabled between trusted processes. Some large tensors
and files are better exchanged through the data store or an external artifact
system than returned through the call response.

See {doc}`../guides/async` and {doc}`../guides/serialization`.
