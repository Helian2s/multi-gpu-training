# EXP-02: Mixed precision and Tensor Cores in distributed training - report

Report status: Not run

## Executive conclusion

Not run. This report remains open until `EXP-02-A2V1` and `EXP-02-A2V2`
complete on AWS-A2 with an immutable EXP-02-capable PyTorch image.

## Run inventory

No measured runs yet.

## Environment

Planned environment: AWS `g7e.12xlarge` through profile `AWS-A2`, RTX PRO 6000
Blackwell Server Edition GPUs, pinned PyTorch image
`037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:ffde9efc9d69ea98fb4da0bb22736a7c7efdee9f72a21e825b6aa51377892bb8`.

## Method

The planned method is the predeclared precision sweep in `experiment.yaml`:
GEMM shape checks, one-visible-GPU training precision variants, and a bounded
two-rank DDP precision check using the accepted model and input lock.

## Results

No results yet.

## Interpretation

Pending measured kernel, numerical, memory, and DDP communication evidence.

## Limitations and anomalies

Known pre-run limitation: EXP-02 has a measured executor and recorded ECR image
digest, but it has not been validated on GPU.

## Cost

No provider cost incurred by this report state.

## Exam takeaway

Pending measured result.

## Reproduction

Pending GPU smoke validation.
