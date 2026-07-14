# EXP-08: FSDP sharding and ZeRO-style memory trade-offs - report

Report status: Not run

## Executive conclusion

Not run. This report remains open until `EXP-08-A2V2` completes on AWS-A2 with
an immutable EXP-08-capable PyTorch image.

## Run inventory

No measured runs yet.

## Environment

Planned environment: AWS `g7e.12xlarge` through profile `AWS-A2`, two visible
GPUs on one host, pinned PyTorch image
`037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:ffde9efc9d69ea98fb4da0bb22736a7c7efdee9f72a21e825b6aa51377892bb8`.

## Method

The planned method is the predeclared DDP versus FSDP sweep in
`experiment.yaml`: replicated DDP baseline, partial sharding, full sharding, and
state-dict round-trip checks.

## Results

No results yet.

## Interpretation

Pending measured memory attribution, throughput, collective traffic,
initialization, and checkpoint evidence.

## Limitations and anomalies

Known pre-run limitation: EXP-08 has a measured executor and recorded ECR image
digest, but it has not been validated on GPU.

## Cost

No provider cost incurred by this report state.

## Exam takeaway

Pending measured result.

## Reproduction

Pending GPU smoke validation.
