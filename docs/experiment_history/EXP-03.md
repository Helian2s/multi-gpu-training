# EXP-03: Microbatch, global batch, and gradient accumulation

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-03`, `pytorch`, `aws`, `AWS-A1`

## Goal And Design

Educational goal: Learn how microbatch size, accumulation steps, and effective global batch are related, and how to improve throughput without accidentally changing the optimization problem.

### Scenario

Describe the realistic engineering situation and the decision the team must
make.

### Question

State one primary question this experiment answers.

### Hypothesis

State a falsifiable expectation, including the direction of the expected
change. Avoid claiming an exact improvement unless a source or model justifies
that threshold.

### Decision rule

Define what evidence confirms, rejects, or leaves the hypothesis inconclusive.
Keep the machine-readable equivalent in `expected_results.yaml`.

### Configuration Snapshot

| Field | Value |
| --- | --- |
| Framework/image family | pytorch |
| Image | `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:e12af417e7e905f30182122a95d73610e3acc9cb41829093d0265dfd6cca4225` |
| Provider | aws |
| Compute profile | AWS-A1 |
| Resource type | g7e.2xlarge |
| GPU type | RTX PRO 6000 Blackwell Server Edition |
| Physical GPUs | 1 |
| Normal visible GPUs | 1 |
| Workload profile | benchmark |
| Baseline | EXP-03-A1-mb1-acc8-global8 |
| Declared variants | 4 |

## Current Conclusion

With fixed effective global batch 8 on one AWS GPU, larger microbatches reduced optimizer-step time until microbatch 4, then memory pressure rose sharply at microbatch 8. All variants produced finite loss.

## Interpretation

More accumulation steps lower peak memory but increase repeated forward/backward overhead per optimizer update. The best point in this short run was microbatch 4 with accumulation 2, not the largest microbatch.

## Run Inventory

### aws-a1-pytorch-20260715T003637Z

- Run ID: `aws-a1-pytorch-20260715T003637Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-03/runs/aws-a1-pytorch-20260715T003637Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:e12af417e7e905f30182122a95d73610e3acc9cb41829093d0265dfd6cca4225`
- Manifest status: `completed`
- Run units: `EXP-03-A1`
- Planned variant count in manifest/plan: `4`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-03/runs/aws-a1-pytorch-20260715T003637Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status.txt`=0
- Finished UTC markers: `finished_utc.txt`=2026-07-15T00:42:13Z
- Raw artifact file count below `raw/`: `9`

## Measured Results From `aws-a1-pytorch-20260715T003637Z`

| Variant | Strategy | Precision | World | Microbatch | Accum | Global batch | Seq | Mean step ms | Peak GiB | Loss | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-03-A1-mb1-acc8-global8 | single | bf16 | 1 | 1 | 8 | 8 | 1024 | 573.475 | 17.65 | r0=6.915 | true |
| EXP-03-A1-mb2-acc4-global8 | single | bf16 | 1 | 2 | 4 | 8 | 1024 | 477.162 | 22.42 | r0=6.913 | true |
| EXP-03-A1-mb4-acc2-global8 | single | bf16 | 1 | 4 | 2 | 8 | 1024 | 465.284 | 31.95 | r0=6.915 | true |
| EXP-03-A1-mb8-acc1-global8 | single | bf16 | 1 | 8 | 1 | 8 | 1024 | 479.772 | 47.81 | r0=6.916 | true |

## Formal Report Status

- Report file: `experiments/exp_03_microbatch_gradient_accumulation/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

Tune microbatch and accumulation together. Gradient accumulation controls effective batch without requiring the largest possible microbatch.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.
