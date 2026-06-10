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

## Build The Quantization Images

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

The SGLang and host-TensorRT Edge-LLM images use the same repo-root build
pattern:

```sh
docker build \
  -f examples/qwen3_asr_orin/Dockerfile.sglang-orin \
  -t ghcr.io/cezarc1/kubetorch-qwen3-asr-orin-sglang:<immutable-tag> \
  .

docker build \
  -f examples/qwen3_asr_orin/Dockerfile.edgellm-hosttrt \
  -t ghcr.io/cezarc1/kubetorch-qwen3-asr-orin-edgellm-hosttrt:<immutable-tag> \
  .
```

Use immutable tags for GitOps. The SGLang image pins the probe-compatible
`kernels==0.14.1` / `kernels-data==0.14.1` pair because `kernels==0.15.2`
failed on import during the Jetson run.

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

The first Orin probes showed that stock FP16 serving is possible only in a very
tight SGLang configuration, and that INT8 TensorRT-Edge-LLM is the realistic
edge-serving path.

Observed on `jetson-orin-nano-01`:

- vLLM `0.22.1` installs from an arm64 wheel and contains `Qwen3ASRConfig`, but
  the server did not reach readiness on the 8 GB Orin Nano in the tested FP16
  configurations.
- SGLang `0.5.12.post1` installs from arm64 CUDA 13 wheels. Pin
  `kernels==0.14.1` and `kernels-data==0.14.1`; `kernels==0.15.2` failed during
  `import sglang` with `ValueError: Either a revision or a version must be specified`.
- SGLang FP16 reached `/v1/models` and completed one OpenAI-compatible
  `/v1/audio/transcriptions` request only with temporary host swap and a small
  cache budget:

```sh
sglang serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --trust-remote-code \
  --dtype half \
  --context-length 512 \
  --mem-fraction-static 0.85 \
  --max-total-tokens 128 \
  --max-running-requests 1 \
  --enable-multimodal \
  --skip-server-warmup \
  --disable-cuda-graph \
  --disable-radix-cache \
  --host 127.0.0.1 \
  --port 18001
```

The SGLang smoke request used a 6 second LibriSpeech dummy clip and returned:

```json
{"text":"Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.","usage":{"type":"duration","seconds":6}}
```

Request wall time was `256.3s`, including first-request FlashInfer/SGLang JIT
compilation for `sm_87`. The generated FlashInfer cache lived under
`/root/.cache/flashinfer/0.6.11.post1/87/` inside the probe container. Treat this
as a proof of execution, not a steady-state benchmark.

The current INT8 path is a TensorRT-Edge-LLM-style artifact:

1. quantize the local Qwen3-ASR checkpoint on the 4090/x86 host with ModelOpt,
2. export the quantized checkpoint to ONNX,
3. build TensorRT engines on an Orin-class Jetson builder,
4. benchmark the final engines on the Orin Nano.

A host-TensorRT Edge-LLM INT8 smoke on `jetson-orin-nano-01` already produced
usable transcripts. The working build used JetPack 7.2 host CUDA/TensorRT
rather than the NGC container TensorRT, because the container TensorRT rejected
Orin SM87. It mounted host `/usr/local/cuda-13.2` and `/usr`, built engines with
TensorRT `10.16.2.10`, and needed temporary 16 GB swap during engine build.

Observed INT8 smoke result:

- audio engine build: `113s`, `611MB`
- LLM engine build: `270s`, `2.0GB`
- benchmark slice: 4 samples, `28.675s` total audio
- cold wall time: `19.864s`, cold RTF `0.693`
- profiled steady GPU time: `4.280s`, steady RTF `0.149`
- LLM generation: `28.7 tok/s`; prefill: `1673 tok/s`
- peak unified memory: `5049MB`
- sanity micro-WER: `7.6%` after stripping the leading `language English` prefix

The host-TensorRT Edge-LLM path is now packaged by the bundle command below.
The remaining work is to run it as a durable Kubetorch/GitOps job over the full
LibriSpeech + FLEURS slice and archive those results.

Generate the calibration/benchmark input bundle from the shared manifest:

```sh
qwen3-asr-orin export-trt-edgellm-bundle \
  --manifest data/qwen3-asr-orin/manifest.jsonl \
  --output-dir results/qwen3-asr-orin/trt-edgellm-int8 \
  --model-path /models/Qwen3-ASR-1.7B \
  --qwen-asr-root /opt/Qwen3-ASR \
  --format int8 \
  --jetson-node-name jetson-orin-nano-01 \
  --host-cuda-path /usr/local/cuda-13.2 \
  --host-usr-path /usr \
  --temp-swap-gib 16 \
  --runtime-output-dir /work/bundle \
  --copy-audio
```

The command writes:

- `audio/`: materialized calibration and benchmark audio files
- `manifest.jsonl`: rewritten manifest pointing at the bundle audio paths
- `prompts.txt`: tab-separated sample ID and reference transcript
- `pipeline.json`: ordered quantize, ONNX export, engine build, and Nano benchmark commands
- `scripts/run_jetson_hosttrt_pipeline.sh`: Jetson-side host-CUDA/TensorRT
  runner that validates `aarch64`, records Jetson environment metadata, builds
  the audio/LLM engines, preprocesses audio, runs the benchmark, and creates
  temporary swap only when `ALLOW_TEMP_SWAP=1` is set

This keeps the self-quantized INT8 artifact path reproducible without pretending
that a ModelOpt checkpoint alone is directly serveable by Transformers.

On the Jetson host or in a pod with host JetPack CUDA/TensorRT mounted at the
paths recorded in `pipeline.json`:

```sh
ALLOW_TEMP_SWAP=1 \
  results/qwen3-asr-orin/trt-edgellm-int8/scripts/run_jetson_hosttrt_pipeline.sh \
  results/qwen3-asr-orin/trt-edgellm-int8/pipeline.json
```

For a Kubernetes pod where host `/usr` is mounted under `/host/usr`, generate
the bundle with:

```sh
qwen3-asr-orin export-trt-edgellm-bundle \
  --manifest data/qwen3-asr-orin/manifest.jsonl \
  --output-dir results/qwen3-asr-orin/trt-edgellm-int8 \
  --model-path /models/Qwen3-ASR-1.7B \
  --qwen-asr-root /opt/Qwen3-ASR \
  --host-cuda-path /host/usr/local/cuda-13.2 \
  --host-usr-path /host/usr \
  --runtime-output-dir /work/bundle \
  --copy-audio
```
