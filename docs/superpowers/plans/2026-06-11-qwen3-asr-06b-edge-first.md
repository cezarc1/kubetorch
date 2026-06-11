# Qwen3-ASR 0.6B Edge-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative Qwen3-ASR 0.6B experiment path that treats TensorRT-Edge-LLM as the primary Orin Nano runtime and SGLang as a comparison runtime.

**Architecture:** Keep the existing 1.7B evidence intact. Make the Kubetorch helper generate model-specific Edge-LLM bundle names from the model path, document ONNX as an intermediate artifact rather than the trusted runtime, and add suspended GitOps/runbook scaffolding for the 0.6B Edge-LLM-first track.

**Tech Stack:** Python, pytest, TensorRT-Edge-LLM bundle generation, Kustomize/Flux GitOps, Kubernetes Jobs/Deployments on Jetson Orin Nano.

---

### Task 1: Make Edge-LLM Bundle Names Model-Aware

**Files:**
- Modify: `examples/qwen3_asr_orin/scripts/edgellm_quant_export_smoke.py`
- Create: `examples/qwen3_asr_orin/tests/test_edgellm_export_smoke.py`

- [x] Add a helper that derives a safe model slug from `model_path`, for example `Qwen3-ASR-0.6B`.
- [x] Use the slug for quantized and ONNX export directory names in the Edge-LLM export smoke path.
- [x] Add assertions that a `Qwen/Qwen3-ASR-0.6B` export writes `Qwen3-ASR-0.6B-int8_sq` and keeps ONNX graph artifacts.
- [x] Run `uv run pytest tests/test_edgellm_export_smoke.py -q` from `examples/qwen3_asr_orin`.

### Task 2: Document The 0.6B Edge-First Experiment

**Files:**
- Modify: `examples/qwen3_asr_orin/README.md`

- [x] Reframe the example as a Qwen3-ASR Orin bakeoff covering both `0.6B` and `1.7B`.
- [x] Add a conservative 0.6B command block using `--model-path /models/Qwen3-ASR-0.6B` and `--output-dir results/qwen3-asr-orin/trt-edgellm-06b-int8`.
- [x] State that ONNX is an export artifact; reported performance must come from built TensorRT engines on the Jetson.
- [x] Keep the existing 1.7B observations unchanged as historical evidence.
- [x] Record the 0.6B TensorRT full-slice result and the 0.6B SGLang host-wedge result.

### Task 3: Add GitOps 0.6B Edge-LLM Track

**Files:**
- Create: `/Users/cezar/Code/cezar-4090-cluster/apps/qwen3-asr-06b-edgellm-orin/README.md`
- Create: `/Users/cezar/Code/cezar-4090-cluster/apps/qwen3-asr-06b-edgellm-orin/kustomization.yaml`
- Create: `/Users/cezar/Code/cezar-4090-cluster/apps/qwen3-asr-06b-edgellm-orin/pvc.yaml`
- Create: `/Users/cezar/Code/cezar-4090-cluster/apps/qwen3-asr-06b-edgellm-orin/job.yaml`
- Modify: `/Users/cezar/Code/cezar-4090-cluster/clusters/cezar-4090/sync/apps.yaml`
- Modify: `/Users/cezar/Code/cezar-4090-cluster/justfile`

- [x] Create a suspended Job and PVC mirroring the 1.7B Edge-LLM app, but name resources with `qwen3-asr-06b`.
- [x] Add a suspended Flux Kustomization that depends on `jetson-nvidia-device-plugin`.
- [x] Add the app to the local `just` inventory lists without resuming it.
- [x] Run `just validate`.

### Task 4: Validate

**Files:**
- No new files.

- [x] Run focused Python tests from `examples/qwen3_asr_orin`.
- [x] Run `git diff --check` in both repositories.
- [x] Run GitOps validation in `/Users/cezar/Code/cezar-4090-cluster`.
- [x] Start the Jetson workload after the scaffold was accepted and Flux app was reconciled.

### Execution Evidence

- `Qwen/Qwen3-ASR-0.6B` INT8 ModelOpt export succeeded on the 4090 through Kubetorch run `qwen-asr-06b-edgellm-int8-export-fixed-0c43ce67`.
- The Kubetorch data-store transfer needed explicit rsync filters because the repo `.gitignore` excludes `*.onnx` and `results/`.
- The Jetson GitOps Job `stt-bench/qwen3-asr-06b-edgellm-hosttrt` completed and produced full-slice TensorRT results: aggregate RTF `1.18`, mean latency `9.82s`, errors `0`, rescored mean WER `6.55%`.
- A smallest-feasible SGLang 0.6B retry reached readiness, but the first real in-cluster audio request made `jetson-orin-nano-01` stop posting kubelet status and SSH timed out during banner exchange.
- Remaining cleanup requires the Jetson to recover or be manually rebooted so the temporary swap file and terminating SGLang pods can be removed.
