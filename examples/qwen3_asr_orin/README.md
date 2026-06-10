# Qwen3-ASR Orin Bakeoff

This example benchmarks `Qwen/Qwen3-ASR-1.7B` as an Orin-only STT workload.
The 4090 is used only to prepare or quantize artifacts; reported inference
numbers should come from `jetson-orin-nano-01`.

## Goal

```text
Run an Orin-only Qwen3-ASR-1.7B inference bakeoff, using the 4090 only to build
INT8 artifacts, and produce reproducible speed/quality evidence through
Kubetorch and GitOps.
```

## Prepare The Dataset Manifest

Use a small public LibriSpeech + FLEURS slice:

```sh
qwen3-asr-orin prepare-manifest \
  --output-dir data/qwen3-asr-orin \
  --librispeech-count 8 \
  --fleurs-count-per-language 2
```

The command writes `manifest.jsonl` with local audio paths, transcripts,
language, split, and duration. This is the source of truth for both FP16 and
INT8 calibration/benchmark runs.

## Run An Orin SGLang Benchmark

The GitOps app in `cezar-4090-cluster/apps/qwen3-asr-sglang-orin` exposes the
service inside the cluster as:

```text
http://qwen3-asr-sglang.stt-bench.svc.cluster.local:30000/v1
```

Run the benchmark client from a Kubetorch batch run or any pod with network
access to `stt-bench`:

```sh
qwen3-asr-orin bench \
  --manifest data/qwen3-asr-orin/manifest.jsonl \
  --output-dir results/qwen3-asr-orin/sglang-fp16-c1 \
  --base-url http://qwen3-asr-sglang.stt-bench.svc.cluster.local:30000/v1 \
  --model qwen3-asr \
  --runtime sglang \
  --precision fp16 \
  --concurrency 1
```

Outputs:

- `summary.json`: aggregate speed, latency, RTF, throughput, errors, and sanity WER
- `samples.jsonl`: one row per audio sample
- `environment.json`: runtime/precision/Python platform metadata

## Build The Quantization Image

Build from the Kubetorch repo root so the Dockerfile can copy the local
Kubetorch client and this example:

```sh
docker build \
  -f examples/qwen3_asr_orin/Dockerfile.quant \
  -t ghcr.io/cezarc1/kubetorch-qwen3-asr-orin-quant:dev \
  .
```

For cluster use, push an immutable tag and pass it to `kt run`. The image
contains Kubetorch, this example, NVIDIA PyTorch 25.08, Transformers, and
NVIDIA ModelOpt.

## INT8 Quantization Boundary

The first quantization command records a reproducible ModelOpt SmoothQuant INT8
spec and intentionally stops before claiming a deployable TensorRT artifact:

```sh
qwen3-asr-orin quantize-modelopt \
  --manifest data/qwen3-asr-orin/manifest.jsonl \
  --output-dir results/qwen3-asr-orin/int8-modelopt \
  --write-spec-only
```

The next implementation step is the model-specific audio `forward_loop` needed
for safe Qwen3-ASR calibration/export. The current scaffold keeps audio tower
and `lm_head` in FP16 and targets the decoder quantizers with W8A8 SmoothQuant.

## TensorRT-Edge-LLM Bundle

Stock SGLang and vLLM both reached Qwen3-ASR model initialization on
`jetson-orin-nano-01`, but OOMed on the 8 GB Nano before serving traffic. The
next realistic INT8 path is a TensorRT-Edge-LLM-style artifact:

1. quantize the local Qwen3-ASR checkpoint on the 4090/x86 host with ModelOpt,
2. export the quantized checkpoint to ONNX,
3. build TensorRT engines on an Orin-class Jetson builder,
4. benchmark the final engines on the Orin Nano.

Generate the calibration/benchmark input bundle from the shared manifest:

```sh
qwen3-asr-orin export-trt-edgellm-bundle \
  --manifest data/qwen3-asr-orin/manifest.jsonl \
  --output-dir results/qwen3-asr-orin/trt-edgellm-int8 \
  --model-path /models/Qwen3-ASR-1.7B \
  --qwen-asr-root /opt/Qwen3-ASR \
  --format int8 \
  --copy-audio
```

The command writes:

- `audio/`: materialized calibration and benchmark audio files
- `manifest.jsonl`: rewritten manifest pointing at the bundle audio paths
- `prompts.txt`: tab-separated sample ID and reference transcript
- `pipeline.json`: ordered quantize, ONNX export, engine build, and Nano benchmark commands

This keeps the self-quantized INT8 artifact path reproducible without pretending
that a ModelOpt checkpoint alone is directly serveable by Transformers.
