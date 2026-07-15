# NeMo/Megatron Image

`Dockerfile` builds the local candidate NeMo/Megatron runtime from the inspected
immutable NGC NeMo `linux/amd64` digest. It preserves the NVIDIA-pinned
framework stack and sets the explicit Python path required for `nemo`,
`megatron.core`, and `megatron.bridge` imports.

The image also builds and installs NVIDIA CUDA Samples
`p2pBandwidthLatencyTest` at a pinned revision so the same Runpod NeMo image can
run EXP-10 communication qualification before EXP-11/EXP-13 model-parallel
measurements on the same two-GPU Pod. The sample build defaults to
`CUDA_SAMPLES_ARCHITECTURES=80`, matching the A100 SXM target.

For Runpod Pods, the image keeps the NVIDIA base entrypoint and sets
`/usr/local/bin/runpod_start` as the default command. That command configures
the injected Runpod SSH public key, starts `sshd`, exports environment variables
for interactive shells, and then sleeps in the foreground so the Pod remains
available for manual queue launch and artifact copy-out.

Build locally with:

```bash
make build-nemo-image
```

The local tag is `multi-gpu-training-nemo:local`. It is not a recorded
experiment image until provider-side GPU qualification passes and the same
content is pushed to ECR and GHCR with immutable digest records.

The default base is the accepted NeMo 26.06 image. Runpod hosts with NVIDIA
driver 570 / CUDA 12.8 need the CUDA 12.8-compatible NeMo 25.04 base instead:

```bash
make build-nemo-image \
  NEMO_BASE_IMAGE=nvcr.io/nvidia/nemo:25.04@sha256:0b981b39cb822feec53d22f789c48be8967bc48567b7b051f4d0a3ffaeb44703 \
  CUDA_SAMPLES_ARCHITECTURES=80 \
  NEMO_IMAGE=multi-gpu-training-nemo:runpod-cuda128-local
```
