# From development to production

Kubetorch's main production advantage is that the execution entrypoint need not
be rewritten for a separate platform. The hardening work is still explicit.

## Pin the environment

- Use immutable workload image tags.
- Pin important Python packages and model revisions.
- Record the Kubetorch chart/client version used for validation.
- Keep example smoke settings separate from full production configurations.

## Stabilize identity

Give long-lived services clear names and namespaces. Configure deployment
namespaces and RBAC in the chart rather than depending on a developer's current
kubectl namespace.

## Make data lifecycle visible

Use volumes for mounted working sets, `kt://` keys for Kubetorch-managed files,
and explicit external URIs for object stores and experiment trackers. Decide
which outputs are run-owned and which survive run deletion.

## Separate interactive and batch paths

Interactive services optimize for fast iteration and warm state. Batch Jobs
optimize for inspectability and bounded lifecycle. A mature project often uses
both: interactive services while developing a model and immutable batch runs
for evaluations, exports, or scheduled retraining.

## Test failure behavior

Exercise preemption, dependency errors, bad input, storage pressure, and cleanup
before calling a workload production-ready. The fault-tolerance tutorials show
several patterns without claiming that retries are automatically safe for every
training algorithm.
