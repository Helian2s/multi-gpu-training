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
digest to be recorded after the next accepted-source rebuild and ECR push.

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

Known pre-run limitation: EXP-09 has a measured executor, but it has not been
validated on GPU and does not yet have a recorded image digest.

## Cost

No provider cost incurred by this report state.

## Exam takeaway

Pending measured result.

## Reproduction

Pending GPU smoke validation and immutable image publication.
