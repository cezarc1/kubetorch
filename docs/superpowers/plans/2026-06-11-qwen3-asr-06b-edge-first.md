# Qwen3-ASR 0.6B Edge-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative Qwen3-ASR 0.6B experiment path that treats TensorRT-Edge-LLM as the primary Orin Nano runtime and SGLang as a comparison runtime.

**Architecture:** Keep the existing 1.7B evidence intact. Make the Kubetorch helper generate model-specific Edge-LLM bundle names from the model path, document ONNX as an intermediate artifact rather than the trusted runtime, and add suspended GitOps/runbook scaffolding for the 0.6B Edge-LLM-first track.

**Tech Stack:** Python, pytest, TensorRT-Edge-LLM bundle generation, Kustomize/Flux GitOps, Kubernetes Jobs/Deployments on Jetson Orin Nano.

---

### Task 1: Make Edge-LLM Bundle Names Model-Aware

**Files:**
- Modify: `examples/qwen3_asr_orin/qwen3_asr_orin/quantize.py`
- Modify: `examples/qwen3_asr_orin/tests/test_quantize.py`
- Modify: `examples/qwen3_asr_orin/tests/test_cli.py`

- [ ] Add a helper that derives a safe model slug from `model_path`, for example `Qwen3-ASR-0.6B`.
- [ ] Use the slug for quantized, ONNX, engine, and results directory names.
- [ ] Add assertions that a `Qwen/Qwen3-ASR-0.6B` bundle writes `Qwen3-ASR-0.6B-int8`, `Qwen3-ASR-0.6B-int8-ONNX`, and `Qwen3-ASR-0.6B-int8-Engines`.
- [ ] Run `uv run pytest tests/test_quantize.py tests/test_cli.py -q` from `examples/qwen3_asr_orin`.

### Task 2: Document The 0.6B Edge-First Experiment

**Files:**
- Modify: `examples/qwen3_asr_orin/README.md`

- [ ] Reframe the example as a Qwen3-ASR Orin bakeoff covering both `0.6B` and `1.7B`.
- [ ] Add a conservative 0.6B command block using `--model-path /models/Qwen3-ASR-0.6B` and `--output-dir results/qwen3-asr-orin/trt-edgellm-06b-int8`.
- [ ] State that ONNX is an export artifact; reported performance must come from built TensorRT engines on the Jetson.
- [ ] Keep the existing 1.7B observations unchanged as historical evidence.

### Task 3: Add GitOps 0.6B Edge-LLM Track

**Files:**
- Create: `/Users/cezar/Code/cezar-4090-cluster/apps/qwen3-asr-06b-edgellm-orin/README.md`
- Create: `/Users/cezar/Code/cezar-4090-cluster/apps/qwen3-asr-06b-edgellm-orin/kustomization.yaml`
- Create: `/Users/cezar/Code/cezar-4090-cluster/apps/qwen3-asr-06b-edgellm-orin/pvc.yaml`
- Create: `/Users/cezar/Code/cezar-4090-cluster/apps/qwen3-asr-06b-edgellm-orin/job.yaml`
- Modify: `/Users/cezar/Code/cezar-4090-cluster/clusters/cezar-4090/sync/apps.yaml`
- Modify: `/Users/cezar/Code/cezar-4090-cluster/justfile`

- [ ] Create a suspended Job and PVC mirroring the 1.7B Edge-LLM app, but name resources with `qwen3-asr-06b`.
- [ ] Add a suspended Flux Kustomization that depends on `jetson-nvidia-device-plugin`.
- [ ] Add the app to the local `just` inventory lists without resuming it.
- [ ] Run `just validate`.

### Task 4: Validate

**Files:**
- No new files.

- [ ] Run focused Python tests from `examples/qwen3_asr_orin`.
- [ ] Run `git diff --check` in both repositories.
- [ ] Run GitOps validation in `/Users/cezar/Code/cezar-4090-cluster`.
- [ ] Do not start the Jetson workload in this pass.
