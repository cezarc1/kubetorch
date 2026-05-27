from __future__ import annotations

try:
    from kubetorch.data_store import DataStoreClient
except Exception:
    DataStoreClient = None

from .constants import DATASET_PREFIX, EXPERIMENT_PREFIX


def cleanup_datastore(namespace: str) -> list[str]:
    if DataStoreClient is None:
        raise RuntimeError("kubetorch is not installed; cannot clean data store")

    client = DataStoreClient(namespace=namespace)
    deleted = []
    for key in (DATASET_PREFIX, EXPERIMENT_PREFIX):
        client.rm(key=key, prefix_mode=True, verbose=True)
        deleted.append(key)
    return deleted
