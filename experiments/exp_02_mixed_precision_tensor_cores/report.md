# EXP-02: Mixed precision and Tensor Cores in distributed training - report

Report status: Not run

## Executive conclusion

Not run. This report remains open until `EXP-02-A2V1` and `EXP-02-A2V2`
complete on AWS-A2 with an immutable EXP-02-capable PyTorch image.

## Run inventory

No measured runs yet.

## Environment

Planned environment: AWS `g7e.12xlarge` through profile `AWS-A2`, RTX PRO 6000
Blackwell Server Edition GPUs, pinned PyTorch image digest to be recorded after
the next accepted-source rebuild and ECR push.

## Method

The planned method is the predeclared precision sweep in `experiment.yaml`:
GEMM shape checks, one-visible-GPU training precision variants, and a bounded
two-rank DDP precision check using the accepted model and input lock.

## Results

No results yet.

## Interpretation

Pending measured kernel, numerical, memory, and DDP communication evidence.

## Limitations and anomalies

Known pre-run limitation: EXP-02 has a measured executor, but it has not been
validated on GPU and does not yet have a recorded image digest.

## Cost

No provider cost incurred by this report state.

## Exam takeaway

Pending measured result.

## Reproduction

Pending GPU smoke validation and immutable image publication.
