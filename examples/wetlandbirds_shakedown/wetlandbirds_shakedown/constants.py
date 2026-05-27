from __future__ import annotations

import os

DATASET_ID = os.getenv(
    "WETLANDBIRDS_DATASET_ID",
    "academic-datasets/Visual-WetlandBirds-Dataset",
)
RUN_GROUP = "wetlandbirds-shakedown"
DATASET_PREFIX = "datasets/visual-wetlandbirds-shakedown"
EXPERIMENT_PREFIX = "experiments/wetlandbirds-shakedown"
HF_CACHE_KEY = f"{DATASET_PREFIX}/hf-cache"


def run_scoped_key(filename: str) -> str:
    run_id = os.getenv("KT_RUN_ID", "local")
    return f"{EXPERIMENT_PREFIX}/{run_id}/{filename.lstrip('/')}"
