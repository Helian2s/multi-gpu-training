# NeMo/Megatron runtime

The [Dockerfile](Dockerfile) packages the synthetic parallelism executors inside
a pinned NVIDIA NeMo environment. It preserves the NVIDIA framework stack,
sets import paths, builds `p2pBandwidthLatencyTest`, and adds SSH/rsync support
for Runpod operation.

EXP-12 used this image family on AWS. EXP-10, EXP-11, EXP-13, and EXP-14 used a
CUDA 12.8-compatible build on Runpod. Their framework versions and immutable
image digests are recorded in [the reports](../../experiments/README.md).

The current executors import Megatron modules for environment checks and then
run custom PyTorch distributed workloads. They are not full Qwen Megatron
Bridge recipes; see [validation status](../../docs/validation-status.md).

## Local builds

From the repository root, the default pinned NeMo 26.06 base is selected by:

```bash
make build-nemo-image
```

The July Runpod phase used a CUDA 12.8-compatible NeMo 25.04 base. To build that
base variant with the current source:

```bash
make build-nemo-image \
  NEMO_BASE_IMAGE=nvcr.io/nvidia/nemo:25.04@sha256:0b981b39cb822feec53d22f789c48be8967bc48567b7b051f4d0a3ffaeb44703 \
  CUDA_SAMPLES_ARCHITECTURES=80 \
  NEMO_IMAGE=multi-gpu-training-nemo:runpod-cuda128-local
```

Both commands load a local `linux/amd64` image without publishing it. The CUDA
Samples target defaults to architecture 80 for the A100 profile; select and
qualify build settings for the actual target hardware before execution.

## Runpod startup and cleanup

[runpod_start.sh](runpod_start.sh) configures the injected SSH public key,
starts `sshd`, exposes the environment to interactive shells, and keeps the
Pod available for manual queue execution and artifact copy-out. It does not
start an experiment queue automatically.

The successful EXP-14 run hotpatched the executor to skip explicit NCCL
process-group destruction after metrics were written. The source now contains
that fix, but the historical GHCR image does not. A replacement image needs
publication and provider qualification before future reproducibility runs.
