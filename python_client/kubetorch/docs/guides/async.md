# Asynchronous calls

Async mode lets one driver overlap remote work without moving orchestration into
a separate workflow language.

```python
import asyncio


async def main():
    model.async_ = True
    calls = [model.generate(prompt) for prompt in prompts]
    return await asyncio.gather(*calls)
```

## Async is a driver concern

Async calls make the local process non-blocking. They do not automatically add
service replicas or GPU capacity. Combine them with autoscaling when a service
should process calls concurrently across replicas.

## Preserve ordering deliberately

`asyncio.gather` returns results in input order, but completion order can differ.
Attach request IDs when results are written to external stores or streamed to
another system.

## Handle partial failure

For independent operations, use `return_exceptions=True` and record failures
beside successful results. For coupled training steps, cancel outstanding calls
when one component fails so stale rollouts or gradients do not continue.

## RL pipelines

The {doc}`../tutorials/reinforcement-learning/async-grpo` tutorial overlaps
rollout generation with training. Its services have different images and GPU
profiles; the Python driver is the synchronization point.
