# Kubetorch Fork Install and Upgrade Notes

Originally built by [Runhouse](https://www.run.house), which has [as of March 2026 shut down](https://www.linkedin.com/posts/greenbergdon_im-excited-to-share-that-the-runhouse-team-share-7453528259448860673-2n_r/).

This fork ships the Python client, controller, data-store image, and Helm chart
as one compatibility set. Install matching versions together.

## Python Client

Until a PyPI package or immutable git tag exists for a release, install from an
exact commit or the branch you intentionally want:

```bash
pip install "git+https://github.com/cezarc1/kubetorch.git@main#subdirectory=python_client"
```

The client tolerates fork build suffixes, such as `0.5.2-cezar-<sha>`, when the
base semantic version matches the cluster version. Set
`KUBETORCH_IGNORE_VERSION_MISMATCH=1` only for deliberate break-glass testing.

## Helm Install

Use the forked chart and forked images together:

```bash
helm upgrade --install kubetorch oci://ghcr.io/cezarc1/charts/kubetorch \
  --version 0.5.2 \
  -n kubetorch \
  --create-namespace
```

For private GHCR images, configure an image pull secret and pass it through
`kubetorchConfig.imagePullSecrets`.

## Upgrade Notes

- The controller Deployment uses `strategy.type: Recreate` because it mounts the
  RWO `kubetorch-controller-data` PVC. Do not switch it back to RollingUpdate
  unless the storage model changes.
- Avoid `--rollback-on-failure` for PVC-bearing upgrades. A failed rollback can
  compound PVC termination and mount state.
- If Helm hits a server-side apply ownership conflict on a resource previously
  created by `kubectl`, rerun with `--force-conflicts` after confirming the
  desired chart state is correct.
- Verify the controller serves `/controller/runs` after upgrade before using
  `kt run`.

## `kt run` Image Contract

`kt run` creates a run record and starts a Kubernetes Job whose command is:

```bash
python -m kubetorch.run_wrapper --
```

The run image must therefore include the Kubetorch Python client and
`kubetorch.run_wrapper`. Either use a Kubetorch runtime image or install this
fork's client into the custom training image.
