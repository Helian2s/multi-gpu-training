# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/nemo:26.06@sha256:bb1dbe94646d5a6490570823cafa0d6f753e1cb60df8f5e89e3b32f3f87893fc

ARG VCS_REF=unknown
ARG BUILD_DATE=unknown

LABEL org.opencontainers.image.title="multi-gpu-training NeMo/Megatron runtime" \
      org.opencontainers.image.description="NeMo Framework and Megatron Core runtime for NCP-GENL model-parallel experiments" \
      org.opencontainers.image.source="https://github.com/Helian2s/multi-gpu-training" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base.name="nvcr.io/nvidia/nemo:26.06@sha256:bb1dbe94646d5a6490570823cafa0d6f753e1cb60df8f5e89e3b32f3f87893fc"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/NeMo:/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM

WORKDIR /workspace/multi-gpu-training

COPY common/ common/
COPY configs/ configs/
COPY experiments/_template/ experiments/_template/
COPY scripts/ scripts/
COPY requirements-preparation.txt requirements-preparation.txt
