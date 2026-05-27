# BioCLIP Eval Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sampled Visual WetlandBirds BioCLIP eval smoke command and run it through Kubetorch so the framework records source, logs, notes, artifacts, metrics, and performance characteristics.

**Architecture:** The eval remains an example-level command under `examples/wetlandbirds_shakedown`. It reuses the existing Hugging Face dataset helpers, crop extraction helper, reporting/artifact publishing utilities, and Typer CLI. Metrics and performance are produced as standalone artifacts so later full-test-set evals can reuse the same schema.

**Tech Stack:** Python 3.11, `uv`, Typer, Hugging Face `datasets`, OpenCLIP/BioCLIP, PyTorch, Kubetorch run/artifact APIs.

---

### Task 1: Metrics And Performance Helpers

**Files:**
- Create: `examples/wetlandbirds_shakedown/wetlandbirds_shakedown/eval_metrics.py`
- Test: `examples/wetlandbirds_shakedown/tests/test_eval_metrics.py`

- [ ] Write failing tests for top-1/top-3 accuracy, macro-F1, per-class recall, and throughput math.
- [ ] Implement pure-Python metric helpers that accept prediction dictionaries and timing values.
- [ ] Run `cd examples/wetlandbirds_shakedown && uv run pytest tests/test_eval_metrics.py -q`.

### Task 2: BioCLIP Eval Smoke Command

**Files:**
- Create: `examples/wetlandbirds_shakedown/wetlandbirds_shakedown/bioclip_eval.py`
- Modify: `examples/wetlandbirds_shakedown/wetlandbirds_shakedown/cli.py`
- Test: `examples/wetlandbirds_shakedown/tests/test_bioclip_eval.py`

- [ ] Write failing tests with fake dataset/model objects showing that the command records config, predictions, metrics, performance, and issues.
- [ ] Implement `run_bioclip_eval_smoke(...)` with streaming dataset loading, best-effort crop extraction, zero-shot species scoring, and JSON/JSONL artifact writes.
- [ ] Publish `eval_config.json`, `predictions.jsonl`, `metrics.json`, `performance.json`, and `issues.md` through `publish_file`.
- [ ] Add a Typer command named `bioclip-eval-smoke`.
- [ ] Run `cd examples/wetlandbirds_shakedown && uv run pytest tests/test_bioclip_eval.py -q`.

### Task 3: Docs And Kubetorch Run Verification

**Files:**
- Modify: `examples/wetlandbirds_shakedown/README.md`

- [ ] Document the local command and the intended `kt run` command.
- [ ] Run the example test suite with `cd examples/wetlandbirds_shakedown && uv run pytest -q`.
- [ ] Submit a small Kubetorch eval run with `kt run` using the shakedown image/source directory if the cluster and image are available.
- [ ] Record the run id, `kt runs show`, logs, notes, and artifact URIs.
