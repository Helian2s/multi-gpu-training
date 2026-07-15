# syntax=docker/dockerfile:1.7
ARG NEMO_BASE_IMAGE=nvcr.io/nvidia/nemo:26.06@sha256:bb1dbe94646d5a6490570823cafa0d6f753e1cb60df8f5e89e3b32f3f87893fc
FROM ${NEMO_BASE_IMAGE}

ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
ARG CUDA_SAMPLES_REF=b7c5481c556c3fe98db060207ecaa41a4b9a9abc
ARG CUDA_SAMPLES_ARCHITECTURES=80
ARG NEMO_BASE_IMAGE

LABEL org.opencontainers.image.title="multi-gpu-training NeMo/Megatron runtime" \
      org.opencontainers.image.description="NeMo Framework and Megatron Core runtime for NCP-GENL model-parallel experiments" \
      org.opencontainers.image.source="https://github.com/Helian2s/multi-gpu-training" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base.name="${NEMO_BASE_IMAGE}" \
      org.opencontainers.image.cuda-samples.revision="${CUDA_SAMPLES_REF}" \
      org.opencontainers.image.cuda-samples.architectures="${CUDA_SAMPLES_ARCHITECTURES}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/NeMo:/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM:/opt/megatron-lm:/opt/NeMo-Framework-Launcher/launcher_scripts \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install --yes --no-install-recommends openssh-server rsync && \
    rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub && \
    mkdir -p /etc/ssh/sshd_config.d /run/sshd && \
    printf '%s\n' \
      'PermitRootLogin prohibit-password' \
      'PasswordAuthentication no' \
      'PubkeyAuthentication yes' \
      'AuthorizedKeysFile .ssh/authorized_keys' \
      > /etc/ssh/sshd_config.d/runpod.conf && \
    rm -rf /var/lib/apt/lists/*

RUN git init /tmp/cuda-samples && \
    git -C /tmp/cuda-samples remote add origin https://github.com/NVIDIA/cuda-samples.git && \
    git -C /tmp/cuda-samples fetch --depth 1 origin "${CUDA_SAMPLES_REF}" && \
    git -C /tmp/cuda-samples checkout --detach FETCH_HEAD && \
    sed -i -E \
      "s/^set\\(CMAKE_CUDA_ARCHITECTURES .*/set(CMAKE_CUDA_ARCHITECTURES ${CUDA_SAMPLES_ARCHITECTURES})/" \
      /tmp/cuda-samples/cpp/5_Domain_Specific/p2pBandwidthLatencyTest/CMakeLists.txt && \
    cmake \
      -S /tmp/cuda-samples/cpp/5_Domain_Specific/p2pBandwidthLatencyTest \
      -B /tmp/cuda-samples-build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /tmp/cuda-samples-build --target p2pBandwidthLatencyTest && \
    install -m 0755 /tmp/cuda-samples-build/p2pBandwidthLatencyTest /usr/local/bin/p2pBandwidthLatencyTest && \
    rm -rf /tmp/cuda-samples /tmp/cuda-samples-build

COPY containers/nemo/runpod_start.sh /usr/local/bin/runpod_start
RUN chmod 0755 /usr/local/bin/runpod_start

WORKDIR /workspace/multi-gpu-training

COPY common/ common/
COPY configs/ configs/
COPY experiments/ experiments/
COPY infra/runpod/ infra/runpod/
COPY scripts/ scripts/
COPY requirements-preparation.txt requirements-preparation.txt

CMD ["/usr/local/bin/runpod_start"]
