# EXP-09: Controlled troubleshooting and failure diagnosis - report

Report status: Not run

## Executive conclusion

Not run. This report remains open until `EXP-09-A2V1` and `EXP-09-A2V2`
complete on AWS-A2 with an immutable EXP-09-capable PyTorch image.

## Run inventory

No measured runs yet.

## Environment

Planned environment: AWS `g7e.12xlarge` through profile `AWS-A2`, one- and
two-visible-GPU phases on the same physical host profile, pinned PyTorch image
`037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:ffde9efc9d69ea98fb4da0bb22736a7c7efdee9f72a21e825b6aa51377892bb8`.

## Method

The planned method is the bounded failure suite in `experiment.yaml`: healthy
smoke, OOM or fragmentation, FP16 non-finite gradients, input starvation,
mismatched collective, rank straggler, and NCCL-timeout evidence.

## Results

No results yet.

## Interpretation

Pending measured logs, telemetry, fault classification, cleanup evidence, and
recovery proof.

## Limitations and anomalies

Known pre-run limitation: EXP-09 has a measured executor and recorded ECR image
digest, but it has not been validated on GPU.

## Cost

No provider cost incurred by this report state.

## Exam takeaway

Pending measured result.

## Reproduction

Pending GPU smoke validation.
