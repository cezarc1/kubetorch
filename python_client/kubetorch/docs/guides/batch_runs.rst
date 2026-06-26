Batch Runs for Agent Operators
==============================

Kubetorch batch runs are short-lived Kubernetes Jobs with durable run records.
They are intended for training, evaluation, data processing, and experiment
shakedowns where the source, intent, environment, logs, notes, and artifacts
must remain inspectable after the container exits.

The practical model is closer to a lightweight Slurm-style run history on
Kubernetes than to a DAG orchestrator. Kubernetes schedules the Pods; Kubetorch
captures the run evidence that a human or coding agent needs later.

Before submitting
-----------------

Start by looking at existing runs so the next run is informed by prior source,
configuration, logs, artifacts, and findings::

   kt runs list --namespace kubetorch
   kt runs show RUN_ID
   kt runs logs RUN_ID
   kt runs artifact list RUN_ID

Use a short ``--intent`` when submitting. Treat it as the experiment question an
agent should see later, for example ``"grpo baseline with smaller KL"`` or
``"BioCLIP eval smoke on sampled wetland bird crops"``.

Submit a run
------------

``kt run`` snapshots the source directory into the Kubetorch data store, creates
a run record, submits a Kubernetes Job, and syncs the source snapshot into the
Job workdir before executing the command::

   kt run \
     --name wetlandbirds-bioclip-eval \
     --intent "Sampled BioCLIP zero-shot eval smoke on Visual WetlandBirds" \
     --namespace kubetorch \
     --image ghcr.io/cezarc1/kubetorch-wetlandbirds-shakedown:dev \
     --source-dir . \
     --env PYTHONPATH=examples/wetlandbirds_shakedown \
     --resources '{"requests":{"cpu":"2","memory":"10Gi","nvidia.com/gpu":"1"},"limits":{"nvidia.com/gpu":"1"}}' \
     -- \
     python -m wetlandbirds_shakedown bioclip-eval-smoke --output-dir results --namespace kubetorch

The run image must include Python, the Kubetorch client, ``rsync``, and the
workload dependencies needed by the command. Private images should use the
cluster's configured image pull secrets or explicit ``--image-pull-secret``
flags.

During and after a run
----------------------

Use ``kt runs show`` for source keys, sanitized environment, status, notes, and
artifact references. Use ``kt runs artifact list RUN_ID`` for a copyable artifact
URI view.

``kt runs logs RUN_ID`` returns the persisted controller log summary. Long
stdout/stderr streams are stored under the run-scoped ``logs_key`` in the data
store and are referenced as ``kt://`` URIs in the log summary.

Inside the run, code can attach findings and result references with the Python
helpers::

   import kubetorch as kt

   kt.note("validation accuracy improved but behavior F1 regressed", author="agent")
   kt.artifact("metrics", uri="kt://kubetorch/experiments/run-123/metrics.json", kind="kt-data-store")
   kt.artifact("wandb", uri="wandb://entity/project/run-123", kind="wandb")

After the run, an operator can add notes or external references from the CLI::

   kt runs note add RUN_ID "Result note: smoke passed; not benchmark-valid" --author agent
   kt runs artifact add RUN_ID --name wandb --uri wandb://entity/project/run-123 --kind wandb
   kt runs artifact list RUN_ID

Recommended artifacts
---------------------

For ML training and eval jobs, prefer publishing a small set of stable,
agent-readable references:

* ``metrics.json`` or ``comparison.json`` for result numbers
* ``environment.json`` and dataset manifests for reproducibility context
* checkpoint directory references plus individual checkpoint artifacts
* external tracker links such as ``wandb://...`` or TensorBoard locations
* ``issues.md`` or notes for known shortfalls discovered during the run

Cleanup
-------

For temporary shakedowns, delete the run record and run-scoped data once the
needed evidence has been exported or recorded::

   kt runs delete RUN_ID --dry-run
   kt runs delete RUN_ID --yes

By default, deletion removes the run record, Kubernetes Job, ``runs/RUN_ID``
source/log prefix, and standard run-scoped ``kt://`` artifacts. Use
``--keep-data`` or ``--keep-job`` when you need to preserve either side.

Current limitations
-------------------

Kubetorch submits Kubernetes Jobs, but it is not a queue or fair-share scheduler.
Kubernetes still places Pods, and features such as queueing, reservations,
arrays, retries, and fair-share policy should be handled by Kubernetes-native
systems or future Kubetorch work.

Run records are only as durable as the configured controller backing store. If
the controller uses pod-local SQLite, records can disappear when the controller
pod is replaced, even if Kubernetes Jobs or data-store files still exist.
