# Tutorials

These examples preserve the strongest material from the original Runhouse
catalog and keep the runnable source beside Kubetorch itself. Start with a
familiar single-service example, then move into distributed and heterogeneous
systems.

```{admonition} Read the validation badge
:class: important

**Validated** records a real execution. **Adapted** means the example uses the
current API and passes repository checks. **Reference** identifies examples
that require external infrastructure or are supporting code.
```

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Training
:link: training/index
:link-type: doc
MNIST, Lightning, LoRA, ResNet, XGBoost, and object detection.
:::
:::{grid-item-card} Reinforcement learning
:link: reinforcement-learning/index
:link-type: doc
Basic and asynchronous GRPO, VeRL, TRL, and code sandboxes.
:::
:::{grid-item-card} Inference and batch
:link: inference/index
:link-type: doc
vLLM, Triton, RAG, embeddings, and large-model serving.
:::
:::{grid-item-card} Distributed computing
:link: distributed/index
:link-type: doc
PyTorch DDP, Ray, TensorFlow, and distributed data preparation.
:::
:::{grid-item-card} Fault tolerance
:link: fault-tolerance/index
:link-type: doc
Preemption, batch sizing, world-size changes, and compute retries.
:::
:::{grid-item-card} Orchestration
:link: orchestration/index
:link-type: doc
Kubeflow Trainer and Temporal integrations.
:::
:::{grid-item-card} Imported source inventory
:link: source-inventory
:link-type: doc
Requirements, configuration, helper modules, and supporting integrations.
:::
::::

```{toctree}
:hidden:

training/index
reinforcement-learning/index
inference/index
distributed/index
fault-tolerance/index
orchestration/index
source-inventory
```
