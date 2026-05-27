FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_CACHE=/workspace/.cache/huggingface/hub \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git curl ca-certificates rsync && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

WORKDIR /opt/wetlandbirds_shakedown
COPY examples/wetlandbirds_shakedown/pyproject.toml examples/wetlandbirds_shakedown/uv.lock ./
COPY examples/wetlandbirds_shakedown/wetlandbirds_shakedown ./wetlandbirds_shakedown
RUN uv pip install --system --compile-bytecode .

COPY python_client /opt/kubetorch/python_client
RUN uv pip install --system --compile-bytecode '/opt/kubetorch/python_client[client]'

WORKDIR /workspace
