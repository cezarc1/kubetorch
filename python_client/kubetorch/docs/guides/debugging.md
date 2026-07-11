# Debugging remote workloads

Start with the layer closest to the symptom. A Python exception, a failed source
sync, an unschedulable pod, and a missing GPU require different evidence.

## Callable health

```bash
kt check SERVICE_NAME --namespace kubetorch
kt describe SERVICE_NAME --namespace kubetorch
kt logs SERVICE_NAME --namespace kubetorch
```

`kt check` reports deployment, pod, source-sync, logging, and GPU readiness. Use
`kubectl describe pod` when scheduling or volume events are the likely cause.

## Interactive debugger

Call a method with debugging enabled or add a `breakpoint()` in trusted
development code, then connect through the CLI:

```bash
kt debug POD_NAME --namespace kubetorch
```

```{youtube} 9vQww8bhCzY
:title: PDB debugging for distributed training
```

## Batch runs

```bash
kt runs show RUN_ID
kt runs logs RUN_ID
```

Inspect the recorded command, source key, environment, status, and log tail
before changing code. Add a note with the root cause so the next run or agent
does not repeat the same diagnosis.

## Common failure boundaries

- **Pending pod:** capacity, taints, affinity, quota, PVC, or image-pull secret.
- **CrashLoopBackOff:** entrypoint, missing dependency, permissions, or runtime
  image mismatch.
- **Source sync failure:** `rsync`, data-store access, or project-root selection.
- **Remote import error:** code not inside the synchronized tree or dependency
  absent from the image.
- **GPU unavailable:** device plugin, resource request, driver/runtime mismatch,
  or another workload holding the device.
