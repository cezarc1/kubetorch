# Qwen3-ASR Orin Bakeoff

This example benchmarks Qwen3-ASR as an Orin-only STT workload. The conservative
path now starts with `Qwen/Qwen3-ASR-0.6B` on TensorRT-Edge-LLM, then compares
SGLang only after the TensorRT engine path is measured. The 4090 is used only to
prepare or quantize artifacts; reported inference numbers should come from
`jetson-orin-nano-01`.

## Goal

```text
Run an Orin-only Qwen3-ASR inference bakeoff, using the 4090 only to build INT8
artifacts, and produce reproducible speed/quality evidence through Kubetorch and
GitOps. Start with `Qwen/Qwen3-ASR-0.6B`; keep the `1.7B` results as historical
evidence and a stretch target.
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
  --model-id Qwen/Qwen3-ASR-0.6B \
  --manifest data/qwen3-asr-orin/manifest.jsonl \
  --output-dir results/qwen3-asr-orin/06b-int8-modelopt \
  --write-spec-only
```

The next implementation step is the model-specific audio `forward_loop` needed
for safe Qwen3-ASR calibration/export. The current scaffold keeps audio tower
and `lm_head` in FP16 and targets the decoder quantizers with W8A8 SmoothQuant.
The CLI defaults to `Qwen/Qwen3-ASR-0.6B`; pass `--model-id Qwen/Qwen3-ASR-1.7B`
only when intentionally reproducing the earlier larger-model run.

## TensorRT-Edge-LLM Bundle

The Orin probes showed that stock FP16 serving is possible only in a very tight
SGLang configuration, and that TensorRT-Edge-LLM is the realistic edge-serving
path for this 8 GB Jetson Orin Nano. The current conservative target is
`Qwen/Qwen3-ASR-0.6B` with INT8 first, falling back to INT4 only if TensorRT or
memory pressure makes INT8 impractical.

ONNX is an intermediate export artifact in this path. Do not report ONNX export
success as a performance result. Report only TensorRT engine build success and
engine-backed inference numbers from the Jetson.

Observed on `jetson-orin-nano-01`:

- vLLM `0.22.1` installs from an arm64 wheel and contains `Qwen3ASRConfig`, but
  the server did not reach readiness on the 8 GB Orin Nano in the tested FP16
  configurations. A recovered-node retry with `gpu_memory_utilization=0.30`,
  `max_model_len=1024`, `max_num_seqs=1`, and eager mode reached
  `Qwen3ASRForConditionalGeneration` engine startup with `dtype=torch.bfloat16`,
  then OOMKilled at the `7Gi` memory limit during model load.
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

A later SGLang FP16 deployment retry reached package install/model startup and
consumed about `5.8GiB` before serving. Flux scaled the Deployment back to zero
mid-startup, and the Jetson kubelet then reported
`KubeletNotReady: PLEG is not healthy`. Do not use the stock SGLang Deployment
as the main benchmark path on this Nano without a prebuilt image, a recovery
plan, and manual isolation from Flux reconciliation.

After reboot and cleanup, the same SGLang Deployment failed more cleanly:
the node stayed Ready, but the container OOMKilled at the `7Gi` limit before
readiness. Logs reached weight loading with `avail mem=3.77 GB`, selected
`triton_attn` for multimodal attention, and then the rank-0 scheduler died with
exit code `-9`.

The current INT8 path is a TensorRT-Edge-LLM-style artifact:

1. quantize the local Qwen3-ASR checkpoint on the 4090/x86 host with ModelOpt,
2. export the quantized checkpoint to ONNX,
3. build TensorRT engines on an Orin-class Jetson builder,
4. benchmark the final engines on the Orin Nano.

A host-TensorRT Edge-LLM INT8 run on `jetson-orin-nano-01` produced usable
transcripts over the full 16-sample LibriSpeech + FLEURS slice. The working
build used JetPack 7.2 host CUDA/TensorRT rather than the NGC container
TensorRT, because the container TensorRT rejected Orin SM87. It mounted host
`/usr/local/cuda-13.2` and `/usr`, built engines with TensorRT `10.16.2.10`,
and needed temporary 16 GB swap during engine build.

Observed INT8 smoke result:

- audio engine build: `113s`, `611MB`
- LLM engine build: `270s`, `2.0GB`
- benchmark slice: 4 samples, `28.675s` total audio
- cold wall time: `19.864s`, cold RTF `0.693`
- profiled steady GPU time: `4.280s`, steady RTF `0.149`
- LLM generation: `28.7 tok/s`; prefill: `1673 tok/s`
- peak unified memory: `5049MB`
- sanity micro-WER: `7.6%` after stripping the leading `language English` prefix

Observed full-slice INT8 result from the GitOps Job
`stt-bench/qwen3-asr-edgellm-hosttrt`:

- benchmark slice: 16 samples, `132.73s` total audio
- runtime: TensorRT-Edge-LLM commit `f9cc74623d95d7acf1addab6026b9d410ba81f52`
- total request latency: `224.10s`
- mean latency: `14.01s`
- aggregate RTF: `1.69`
- mean per-sample RTF: `2.43`
- errors: `0`
- rescored mean WER: `12.81%`

The full-slice benchmark launches `llm_inference` once per sample, so the RTF
includes repeated process/model startup. A persistent Edge-LLM server or batched
driver is the next optimization target before judging steady-state throughput.

Recompute the saved-run sanity WER with the package helper:

```sh
qwen3-asr-orin rescore-edgellm-results \
  --samples results/qwen3-asr-orin/jetson-hosttrt-run/results-int8/samples.jsonl \
  --output-dir results/qwen3-asr-orin/jetson-hosttrt-run/results-int8-rescored
```

The host-TensorRT Edge-LLM path is packaged by the bundle command below.

Generate the 0.6B calibration/benchmark input bundle from the shared manifest:

```sh
qwen3-asr-orin export-trt-edgellm-bundle \
  --manifest data/qwen3-asr-orin/manifest.jsonl \
  --output-dir results/qwen3-asr-orin/trt-edgellm-06b-int8 \
  --model-path /models/Qwen3-ASR-0.6B \
  --qwen-asr-root /opt/Qwen3-ASR \
  --format int8 \
  --jetson-node-name jetson-orin-nano-01 \
  --host-cuda-path /usr/local/cuda-13.2 \
  --host-usr-path /usr \
  --temp-swap-gib 16 \
  --runtime-output-dir /work/bundle \
  --copy-audio
```

For the historical 1.7B path, keep the same command shape but use
`--output-dir results/qwen3-asr-orin/trt-edgellm-17b-int8` and
`--model-path /models/Qwen3-ASR-1.7B`.

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
  results/qwen3-asr-orin/trt-edgellm-06b-int8/scripts/run_jetson_hosttrt_pipeline.sh \
  results/qwen3-asr-orin/trt-edgellm-06b-int8/pipeline.json
```

For a Kubernetes pod where host `/usr` is mounted under `/host/usr`, generate
the bundle with:

```sh
qwen3-asr-orin export-trt-edgellm-bundle \
  --manifest data/qwen3-asr-orin/manifest.jsonl \
  --output-dir results/qwen3-asr-orin/trt-edgellm-06b-int8 \
  --model-path /models/Qwen3-ASR-0.6B \
  --qwen-asr-root /opt/Qwen3-ASR \
  --host-cuda-path /host/usr/local/cuda-13.2 \
  --host-usr-path /host/usr \
  --runtime-output-dir /work/bundle \
  --copy-audio
```
