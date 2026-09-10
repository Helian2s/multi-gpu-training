# PyTorch runtime

The [Dockerfile](Dockerfile) packages the shared runners and experiment code on
a pinned NVIDIA NGC PyTorch base. It retains the base's CUDA/NCCL stack, adds
pinned Hugging Face runtime dependencies, and builds NVIDIA's
`p2pBandwidthLatencyTest` from a pinned CUDA Samples revision.

The image family was used for AWS EXP-01–09: communication qualification,
Qwen-based training, precision and memory variants, attention benchmarks,
PyTorch profiling, and controlled diagnostics.

## Local build

From the repository root:

```bash
make build-pytorch-image
```

The default local tag is `multi-gpu-training-pytorch:local`. Override it with
`PYTORCH_IMAGE=...`. The target uses Buildx with `--platform linux/amd64 --load`
and does not push to a registry.

Exact package pins are in [requirements-runtime.txt](requirements-runtime.txt).
The historical AWS reports record the images actually used; a build from the
current Dockerfile is not automatically the same historical image.

AWS ECR images were deleted in the recorded July cleanup. A future run needs
an authorized publication, an immutable digest, host qualification, and the
relevant training-correctness fixes from
[the validation review](../../docs/validation-status.md).
