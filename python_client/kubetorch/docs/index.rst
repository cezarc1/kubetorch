Kubetorch Documentation
=======================

Kubetorch is a Python client, controller, and data store for running ML
workloads on Kubernetes. This fork emphasizes agent-friendly batch runs: each
run should be inspectable through its source snapshot, intent, environment,
logs, notes, and artifact references.

Start with the batch-run guide if you are using Kubetorch for training,
evaluation, data processing, or experiment shakedowns. Use the API references
when you need exact Python or CLI surfaces.

.. toctree::
   :maxdepth: 1
   :caption: Guides and API Reference

   guides/batch_runs
   api/python
   api/cli

External references
-------------------

* `Upstream Kubetorch documentation <https://www.run.house/kubetorch/introduction>`_
* `Upstream examples <https://www.run.house/examples>`_
