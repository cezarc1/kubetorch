# Decorators and declarative deployment

Decorators keep infrastructure definitions close to entrypoints that should be
deployed together.

```python
import kubetorch as kt


@kt.compute(cpus="2", memory="4Gi")
def preprocess(path: str):
    ...
```

The imperative and decorator interfaces describe the same underlying concepts.
Use imperative `kt.fn(...).to(compute)` calls when the driver dynamically
constructs services. Use decorators and `kt deploy` when a module is the stable
deployment unit.

Distribution, autoscaling, and async decorators compose with compute
definitions. Keep resource policy explicit: decorators should make the expected
namespace, image, scaling, and distribution visible to reviewers rather than
hide them in helper metaprogramming.
