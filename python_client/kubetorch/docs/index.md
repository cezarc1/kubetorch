---
html_theme.sidebar_secondary.remove: true
---

# Kubetorch Documentation

<div class="kt-hero">
  <p class="kt-eyebrow">Pythonic remote code execution · Kubernetes native</p>
  <div class="kt-hero-heading" aria-hidden="true">A easier, agent-first, way to train and run inference on Kubernetes</div>
  <p class="kt-lede">Kubetorch gives notebooks, scripts, CI, and agents the same direct path to Kubernetes GPU aware compute, without you having to know Kubernetes.</p>
  <div class="kt-execution-rail" aria-label="Kubetorch execution flow">
    <code>train.py</code><b>→</b><code>.to(compute)</code><b>→</b><code>k8s</code><b>→</b><code>kt://artifacts</code>
  </div>
  <p class="kt-actions"><a class="kt-primary" href="start/quickstart.html">Run hello world</a><a href="tutorials/index.html">Explore 33 tutorials</a></p>
</div>

Kubetorch was created by the [Runhouse team](https://www.linkedin.com/posts/greenbergdon_im-excited-to-share-that-the-runhouse-team-share-7453528259448860673-2n_r/). This fork, maintained by [@cezarc1](https://github.com/cezarc1) preserves the original idea while adding durable batch runs, source and log capture, notes, artifact references, and an agent first approach.

Kubetorch is meant to be run on a large k8s cluster to small k3s homelabs.

Kubetorchs' main goal is to be a tool for researchers, and engineers to use in their daily work without getting burned by k8s, dealing with slow deployments and rsync.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Start with the system
:link: start/introduction
:link-type: doc

Understand the compute, callable, image, and data-store primitives before
dispatching your first function.
:::

:::{grid-item-card} Train something familiar
:link: tutorials/training/mnist
:link-type: doc

Use the classic MNIST tutorial to see ordinary PyTorch code become a remote
class on Kubernetes.
:::

:::{grid-item-card} Build an RL system
:link: tutorials/reinforcement-learning/basic-grpo
:link-type: doc

Coordinate distinct inference, reward, and distributed-training services from
one Python process.
:::

:::{grid-item-card} Operate with evidence
:link: guides/batch_runs
:link-type: doc

Submit inspectable Jobs and preserve intent, source, logs, notes, and output
artifacts for the next engineer or agent.
:::
::::

```{toctree}
:hidden:
:caption: Start here

start/introduction
start/installation
start/quickstart
start/workflow
```

```{toctree}
:hidden:
:caption: Concepts

concepts/overview
concepts/primitives
concepts/supporting-primitives
concepts/call-modes
concepts/decorators
concepts/data-store
```

```{toctree}
:hidden:
:caption: Guides

guides/adapting-code
guides/async
guides/autoscaling
guides/debugging
guides/dev-prod
guides/distributed
guides/serialization
guides/monitoring
```

```{toctree}
:hidden:
:caption: Agent and batch runs

guides/batch_runs
guides/agent-workflows
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:hidden:
:caption: API reference

api/python
api/cli
```
