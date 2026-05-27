# Visual WetlandBirds Kubetorch Shakedown

This example validates Kubetorch batch runs with a real GPU/data workflow:

1. record the run environment;
2. ingest `academic-datasets/Visual-WetlandBirds-Dataset` through Hugging Face `datasets`;
3. run a BioCLIP/OpenCLIP GPU smoke test;
4. clean all experiment data from the Kubetorch data store.

The experiment is intentionally small. It is a framework shakedown before a real
Visual WetlandBirds training run.

## Local tests

```bash
uv run pytest -q
```

## Cluster entrypoints

```bash
python -m wetlandbirds_shakedown env-probe --output-dir results
python -m wetlandbirds_shakedown ingest-hf --output-dir results --namespace kubetorch
python -m wetlandbirds_shakedown bioclip-smoke --output-dir results --namespace kubetorch
python -m wetlandbirds_shakedown bioclip-eval-smoke --output-dir results --namespace kubetorch --split test --sample-rows 50
python -m wetlandbirds_shakedown cleanup --output-dir results --namespace kubetorch
```

`ingest-hf`, `bioclip-smoke`, and `bioclip-eval-smoke` default to Hugging Face
streaming mode. Use `--materialize` only after choosing a compatible `torch` /
`torchcodec` / `datasets` stack for local video decoding and Arrow cache
generation.

## Kubetorch eval run

Build and push the run image, then submit a sampled eval through Kubetorch:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f examples/wetlandbirds_shakedown/Dockerfile \
  -t ghcr.io/cezarc1/kubetorch-wetlandbirds-shakedown:dev \
  --push .

kt run \
  --name wetlandbirds-bioclip-eval \
  --intent "Sampled BioCLIP zero-shot eval smoke on Visual WetlandBirds" \
  --namespace kubetorch \
  --image ghcr.io/cezarc1/kubetorch-wetlandbirds-shakedown:dev \
  --source-dir . \
  --env PYTHONPATH=examples/wetlandbirds_shakedown \
  --image-pull-secret ghcr-pull-secret \
  -- \
  python -m wetlandbirds_shakedown bioclip-eval-smoke \
    --output-dir results \
    --namespace kubetorch \
    --split test \
    --sample-rows 50
```

The eval writes `eval_config.json`, `predictions.jsonl`, `metrics.json`,
`performance.json`, and `issues.md`, then publishes them as run artifacts.

## Kubetorch run image requirements

`kt run` snapshots the local source directory into the Kubetorch data store and
the run wrapper syncs it back into the Job container. The run image therefore
must include:

- Python with the matching Kubetorch client installed;
- `rsync`, used by the data-store sync path;
- the dependencies needed by the selected entrypoint.

Private images should be pulled through the cluster's configured
`kubetorchConfig.imagePullSecrets`. The controller health config exposes those
secret names, and `kt run` uses them by default unless `--image-pull-secret` is
provided.

After a shakedown, use `kt runs delete RUN_ID --yes` to remove the run record,
the Kubernetes Job, the `runs/RUN_ID` source/log prefix, and standard
`kt://.../RUN_ID/...` artifacts.
