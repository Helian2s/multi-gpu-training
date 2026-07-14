# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:26.06-py3@sha256:abd110b23600e877173dafc3078385b7c13ddacd7e0c6a6acb0a864586d59622

ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="multi-gpu-training PyTorch runtime" \
      org.opencontainers.image.description="Native PyTorch runtime for NCP-GENL GPU acceleration experiments" \
      org.opencontainers.image.source="https://github.com/Helian2s/multi-gpu-training" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base.name="nvcr.io/nvidia/pytorch:26.06-py3@sha256:abd110b23600e877173dafc3078385b7c13ddacd7e0c6a6acb0a864586d59622"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/multi-gpu-training

COPY containers/pytorch/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt && \
    rm /tmp/requirements-runtime.txt

COPY common/ common/
COPY configs/ configs/
COPY experiments/_template/ experiments/_template/
COPY scripts/ scripts/
COPY requirements-preparation.txt requirements-preparation.txt
