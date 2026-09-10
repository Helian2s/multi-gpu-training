# GPU runtime images

The project includes two `linux/amd64` image families built from pinned NVIDIA
NGC bases. They package the experiment code and GPU tooling before renting a
host, keeping environment setup separate from measured execution.

| Family | Contents and use | Build guide |
| --- | --- | --- |
| PyTorch | Native training, DDP/FSDP, GEMM/attention benchmarks, profiling, and AWS communication tools | [pytorch/](pytorch/README.md) |
| NeMo/Megatron | NVIDIA framework stack, synthetic parallelism executors, P2P tooling, and Runpod SSH startup | [nemo/](nemo/README.md) |

The Dockerfiles record source/build metadata and pin the NGC base plus the CUDA
Samples revision. Additional PyTorch-image Python packages are pinned in
[pytorch/requirements-runtime.txt](pytorch/requirements-runtime.txt).
[Base-image compatibility notes](base-image-compatibility.md) preserve the
historical inspection record.

## Build locally

From the repository root with Docker and Buildx available:

```bash
make build-pytorch-image
make build-nemo-image
```

These targets build and load local images; they do not publish them. Large
native builds were performed on Ubuntu x86_64. macOS ARM64 can inspect source
and build through emulation, but a local image build does not validate CUDA.

## Registry and run identity

AWS runs pulled from private ECR; Runpod runs pulled from GHCR. Each recorded
run identifies its image by immutable digest. The project policy is to verify
identical content when mirroring one build between registries. That does not
make images from different builds equivalent: the July AWS and Runpod NeMo
runs used different base/runtime versions, recorded in their reports.

A locally rebuilt image has its own identity. Record the exact source state,
base, dependencies, and resulting digest, then perform provider-side GPU
qualification before measured use. Publishing an image requires authorization.

## Historical availability

The [July handoff](../HANDOFF.md) records deletion of the AWS ECR images on
2026-07-16. The published Runpod image used for EXP-14 also predates the
committed NCCL cleanup fix; that run used a container-side hotpatch. Neither
an old digest in YAML nor a successful local build establishes rerun readiness.
See [validation status](../docs/validation-status.md) and
[provider readiness](../infra/TOOLING.md) before preparing another run.
