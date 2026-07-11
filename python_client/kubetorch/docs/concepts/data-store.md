# Data store

The Kubetorch data store provides namespace-scoped file and directory transfer.
It backs source synchronization and the standard `kt://` run-artifact scheme,
and is also available to workload code.

## CLI

```bash
kt put datasets/sample ./sample --namespace kubetorch
kt ls datasets --namespace kubetorch
kt get datasets/sample ./restored --namespace kubetorch
kt rm datasets/sample --recursive --namespace kubetorch
```

## Python

```python
from pathlib import Path
import kubetorch as kt

Path("metrics.json").write_text('{"accuracy": 0.98}\n')
kt.put("runs/demo/artifacts/metrics.json", "metrics.json", namespace="kubetorch")
kt.get("runs/demo/artifacts/metrics.json", "restored.json", namespace="kubetorch")
```

## Keys and URIs

A key is relative to a namespace. Use stable prefixes for datasets, shared
checkpoints, and run-owned data. A URI such as
`kt://kubetorch/runs/demo/artifacts/metrics.json` can be recorded in a run
without copying the contents into controller metadata.

## Lifecycle

Batch-run deletion removes standard data under that run's prefix. Shared inputs
and checkpoints should live outside run-owned prefixes. Always inspect
`kt runs delete RUN_ID --dry-run` before cleanup.
