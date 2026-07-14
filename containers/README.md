# Container images

The project maintains two independent `linux/amd64` image families:

- `pytorch/` for native PyTorch, DDP, FSDP, collectives, kernels, and profiling.
- `nemo/` for NeMo Framework and Megatron Core model-parallel experiments.

Do not add a Dockerfile until its NGC base release and dependency versions are
selected and compatibility-tested together. Keep exact pins in the Dockerfile
and dependency lock files, with a short compatibility record; do not duplicate
those implementation details in `PROJECT_DECISIONS.md`.

Current local base-image inspection is recorded in
[base-image-compatibility.md](base-image-compatibility.md). That record admits
candidate bases for Dockerfile design but does not replace provider-side GPU
qualification or final project image digest recording.

Each image is built once and published to two private registries:

```text
<aws-account-id>.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch:<version>
<aws-account-id>.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-nemo:<version>
ghcr.io/<github-owner>/multi-gpu-training-pytorch:<version>
ghcr.io/<github-owner>/multi-gpu-training-nemo:<version>
```

AWS pulls the ECR references through an instance role. Runpod pulls the GHCR
mirrors through read-only registry credentials. Recorded experiments use an
immutable per-registry digest, never only a mutable tag. The build record must
show that both references came from the same build rather than rebuilding for
each provider.
