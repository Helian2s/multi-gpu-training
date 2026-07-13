# PyTorch image

Base this image on a pinned NVIDIA NGC PyTorch release. It will contain the
native PyTorch experiment runtime, NCCL/CUDA tools, NVIDIA profilers, shared
qualification utilities, and only the model/data libraries accepted for the
shared workload.

The image is not ready for implementation until the base image digest and the
selected workload dependencies are accepted and compatibility-tested together.
