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

Call a remote function or method with `debug=True`, or add a `breakpoint()` in
trusted development code. In a distributed workload, guard the breakpoint by
rank when you only want one debugger session:

```python
if torch.distributed.get_rank() == 0:
    breakpoint()
```

The remote runtime prints the complete connection command. Copy that command
rather than reconstructing it; depending on where the client runs, it can
include the pod name, port, namespace, mode, and pod IP:

```bash
kt debug POD_NAME --port 5678 --namespace kubetorch --mode pdb --pod-ip POD_IP
```

The terminal session provides normal PDB commands while the other ranks wait or
continue according to the coordination in your training code. Finish or
continue the debugger before tearing down the service.

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
