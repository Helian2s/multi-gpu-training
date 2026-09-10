# Runpod execution tooling

The Runpod phase used A100-SXM4-80GB GPUs for NVLink communication and synthetic
parallelism workloads. Both queues ran in the CUDA 12.8-compatible
NeMo/Megatron image, so communication qualification and the synthetic workloads
could share a Pod without changing images.

## Queues used

| Queue | Physical GPUs | Experiments |
| --- | ---: | --- |
| RUNPOD-A2-Megatron | 2 | EXP-10 communication; EXP-11 TP/SP and EXP-13 CP with 1/2-visible-GPU phases |
| RUNPOD-A4-Megatron | 4 | EXP-14 DP=4, TP=4, and TP=2 × DP=2 |

Both profiles select Runpod Secure Cloud and the exact GPU type
`NVIDIA A100-SXM4-80GB`. Every run remains on one host. The recorded topology
was `NV12`; GPU names alone are not qualification evidence, and `NV12` alone
is not a claim that a particular NVSwitch fabric was verified.

## What the tooling implements

[runpod_queue.py](runpod_queue.py) validates queue configuration and generates
plans, container scripts, and a `runpodctl pod create` command. It does not
execute the generated creation command. The image startup script enables SSH
and keeps the Pod available for an operator to launch the queue.

The configurations are [a2_megatron_queue.yaml](a2_megatron_queue.yaml) and
[a4_megatron_queue.yaml](a4_megatron_queue.yaml). From the repository root:

```bash
make runpod-a2-megatron-queue-plan PYTHON=.venv/bin/python
make runpod-a4-megatron-queue-plan PYTHON=.venv/bin/python
make runpod-a2-megatron-queue-script PYTHON=.venv/bin/python RUN_ID=review-a2
make runpod-a4-megatron-pod-create-command PYTHON=.venv/bin/python RUN_ID=review-a4
```

These commands print local plans or shell text. Running the printed Pod
creation command is a separate, billable action requiring explicit approval.
Its registry credential reference belongs to the original environment and
must be replaced with an authorized local configuration for another account.

## Storage and lifetime

The checked-in queues use Pod volume storage mounted at `/runpod-volume`,
with no network-volume ID. Artifacts are copied by the operator to
`artifacts/runs/runpod-volume-mirror/` before Pod deletion. The
[artifact guide](../../artifacts/README.md) describes the mirror layout.

The generated Pod creation command contains a termination deadline calculated
when the command is generated. Generate a fresh command immediately before an
authorized launch. Each run unit also has a timeout.

The container script does not call the provider to stop or delete the Pod.
`stop_on_success=true` prints an instruction to the operator; the inspection
interval is not an automatic stop timer. Artifact copy-out and early shutdown
remain manual, including after failures. Preserve and verify artifacts before
the hard deadline rather than relying on Pod storage surviving deletion.

## Recorded results and remaining work

The July records include EXP-10/11/13 artifacts and a completed EXP-14 report.
The first EXP-14 attempt hung during NCCL cleanup. A rerun used a container-side
hotpatch to skip explicit process-group destruction; that fix is now in source.
Publish a replacement immutable image before future reproducibility runs.

The synthetic TP/SP/CP/hybrid paths have correctness issues described in
[validation status](../../docs/validation-status.md). Their recorded performance
is not yet evidence of equivalent full-model training.

Provider access, image availability, account limits, capacity, and prices must
be checked again before another launch. [TOOLING.md](../TOOLING.md) and
[HANDOFF.md](../../HANDOFF.md) contain dated observations, not live Pod state.
