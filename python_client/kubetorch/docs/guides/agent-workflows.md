# Agent workflow

Kubetorch's run history gives an agent a concrete place to inspect prior work
before consuming more compute.

## Before launching

```bash
kt check --namespace kubetorch
kt runs list --namespace kubetorch
```

Read recent intents, statuses, and timestamps. Choose a short name and a specific
intent that says what is changing and what result would be useful.

## Launch one controlled change

```bash
kt run \
  --name grpo-kl-smoke \
  --intent "Compare reduced KL coefficient against the current GRPO smoke baseline" \
  --namespace kubetorch \
  --image IMAGE_AT_IMMUTABLE_TAG \
  --source-dir . \
  -- python train.py --config configs/smoke.yaml
```

Use immutable images for meaningful experiments and snapshot the exact source
tree being evaluated.

## Inspect before guessing

```bash
kt runs show RUN_ID
kt runs logs RUN_ID
kt runs artifact list RUN_ID
```

If the run fails, record the root cause and next action:

```bash
kt runs note add RUN_ID \
  "OOM during rollout generation; reduce max_model_len before changing batch size." \
  --author agent
```

## Register outputs

Inside a run, use `kt.artifact` for checkpoints, metrics, tracker URLs, reports,
and dataset manifests. Prefer stable URIs and explain the result in a note.

## Clean up with a preview

```bash
kt runs delete RUN_ID --dry-run
```

Only use `--yes` after confirming which Job, source, logs, record, and run-owned
artifacts will be removed. Shared datasets and baselines should live outside the
run prefix.
