# PyTorch Image

`Dockerfile` builds the local candidate native PyTorch runtime from the
inspected immutable NGC PyTorch `linux/amd64` digest. It contains the NVIDIA
PyTorch, CUDA, NCCL, and profiler stack from the base image plus the minimal
Hugging Face runtime packages needed for the accepted Qwen/WikiText workload.
For EXP-01 it also builds and installs CUDA Samples
`p2pBandwidthLatencyTest` from a pinned NVIDIA CUDA Samples commit.

Build locally with:

```bash
make build-pytorch-image
```

The local tag is `multi-gpu-training-pytorch:local`. It is not a recorded
experiment image until provider-side GPU qualification passes and the same
content is pushed to ECR and GHCR with immutable digest records.
