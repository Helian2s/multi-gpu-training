# Operational Experiment Timeline

This file records the experiment execution history that is easy to lose in chat: capacity attempts, image/auth issues, launch bugs, hotfixes, and which run IDs are the measurements to trust. It is intentionally concise and points to the per-experiment history files for measurements.

## AWS-A1 capacity and user-data fix

Initial AWS-A1 capacity probes across us-west-2 AZs failed with InsufficientInstanceCapacity. A later AWS-A1 launch in us-west-2b reached SSM and showed one RTX PRO 6000 GPU, but bootstrap failed because generated heredoc delimiters were indented. Tests and launch generation were fixed before the successful AWS-A1 queue run.

## AWS-A2 launch and queue generator fixes

AWS-A2 capacity was scarce. The first created AWS-A2 instance failed before measurement because queue user-data wrote shell commands through a TSV file and lost JSON command fields. The queue generator was fixed to emit shell-quoted run-unit calls.

## EXP-02 FP16 executor fix

A first AWS-A2 retry reached EXP-02 and failed because FP16 training loaded model parameters as FP16 while also using GradScaler. The executor was changed so FP16+GradScaler keeps model parameters in FP32 and uses FP16 autocast.

## AWS-A2 PyTorch completion

The run aws-a2-full-fp16fix-20260715T023252Z completed EXP-01, EXP-02, EXP-07, EXP-08, and EXP-09 with exit status 0. The same run provided the AWS two-GPU communication, precision, DDP, FSDP, and failure diagnosis measurements.

## AWS-A2 Megatron completion

EXP-12 first hit capacity failures in fixed AZ attempts. The successful AWS-selected placement launched in us-west-2b, ran QUAL-A2 and EXP-12-A2V1/A2V2, and stopped after the success hold.

## Runpod registry and image readiness

Early Runpod attempts exposed GHCR authorization/image-start issues. The usable path became the GHCR NeMo image with Runpod SSH startup and CUDA 12.8 support, pulled by digest through a read-only registry auth.

## Runpod A2 completion

The run runpod-a2-megatron-20260715T203057Z completed QUAL-RUNPOD-A2, EXP-10, EXP-11, and EXP-13. Artifacts were copied from the Pod volume into the local runpod-volume mirror before the Pod was stopped.

## EXP-14 cleanup hotfix

The first Runpod A4 run wrote metrics but hung during explicit NCCL process-group destruction. The successful rerun used a container-side hotpatch, now committed in source, to skip explicit destroy by default and rely on short-lived subprocess exit for NCCL cleanup.

## Qualification And Queue Artifact Directories

| Kind | Directory |
| --- | --- |
| qualification | artifacts/runs/aws-s3-mirror/QUAL-A2/runs/aws-a2-megatron-exp12-20260715T0340Z |
| qualification | artifacts/runs/runpod-volume-mirror/QUAL-RUNPOD-A2-Megatron/runpod-a2-megatron-20260715T193128Z |
| qualification | artifacts/runs/runpod-volume-mirror/QUAL-RUNPOD-A2-Megatron/runpod-a2-megatron-20260715T193226Z |
| qualification | artifacts/runs/runpod-volume-mirror/QUAL-RUNPOD-A2-Megatron/runpod-a2-megatron-20260715T203057Z |
| qualification | artifacts/runs/runpod-volume-mirror/QUAL-RUNPOD-A4-Megatron/runpod-a4-megatron-20260715T210918Z |
| qualification | artifacts/runs/runpod-volume-mirror/QUAL-RUNPOD-A4-Megatron/runpod-a4-megatron-20260715T212017Z |
| queue | artifacts/runs/aws-s3-mirror/AWS-A1-PyTorch/runs/aws-a1-pytorch-20260715T003637Z |
| queue | artifacts/runs/aws-s3-mirror/AWS-A2-Megatron/runs/aws-a2-megatron-exp12-20260715T0340Z |
| queue | artifacts/runs/aws-s3-mirror/AWS-A2-PyTorch/runs/aws-a2-full-fp16fix-20260715T023252Z |
| queue | artifacts/runs/runpod-volume-mirror/RUNPOD-A2-Megatron/runpod-a2-megatron-20260715T193128Z |
| queue | artifacts/runs/runpod-volume-mirror/RUNPOD-A2-Megatron/runpod-a2-megatron-20260715T193226Z |
| queue | artifacts/runs/runpod-volume-mirror/RUNPOD-A2-Megatron/runpod-a2-megatron-20260715T203057Z |
| queue | artifacts/runs/runpod-volume-mirror/RUNPOD-A4-Megatron/runpod-a4-megatron-20260715T210918Z |
| queue | artifacts/runs/runpod-volume-mirror/RUNPOD-A4-Megatron/runpod-a4-megatron-20260715T212017Z |
