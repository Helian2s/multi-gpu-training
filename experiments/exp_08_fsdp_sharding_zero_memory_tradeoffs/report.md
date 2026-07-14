# EXP-08: FSDP sharding and ZeRO-style memory trade-offs - report

Report status: Not run

## Executive conclusion

Not run. This report remains open until `EXP-08-A2V2` completes on AWS-A2 with
an immutable EXP-08-capable PyTorch image.

## Run inventory

No measured runs yet.

## Environment

Planned environment: AWS `g7e.12xlarge` through profile `AWS-A2`, two visible
GPUs on one host, pinned PyTorch image digest to be recorded after the next
accepted-source rebuild and ECR push.

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

Known pre-run limitation: EXP-08 has a measured executor, but it has not been
validated on GPU and does not yet have a recorded image digest.

## Cost

No provider cost incurred by this report state.

## Exam takeaway

Pending measured result.

## Reproduction

Pending GPU smoke validation and immutable image publication.
