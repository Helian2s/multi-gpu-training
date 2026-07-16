# EXP-05: PyTorch SDPA/FlashAttention and operator fusion

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-05`, `pytorch`, `aws`, `AWS-A1`

## Goal And Design

Educational goal: Learn how to distinguish attention-backend improvements from general graph/operator fusion effects using kernel evidence, memory measurements, and correctness checks.

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
| Baseline | EXP-05-A1-sdpa-math-bf16 |
| Declared variants | 5 |

## Current Conclusion

Automatic/FlashAttention SDPA was dramatically faster and smaller than the math backend for the Qwen-like GQA shape. Automatic SDPA measured about 0.171 ms/iteration versus 4.777 ms/iteration for math.

## Interpretation

Backend selection dominated this attention microbenchmark. `torch.compile` reduced math-backend steady-state time but added compile warmup, while automatic compiled SDPA was slightly slower than automatic eager in this short measurement.

## Run Inventory

### aws-a1-pytorch-20260715T003637Z

- Run ID: `aws-a1-pytorch-20260715T003637Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-05/runs/aws-a1-pytorch-20260715T003637Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:e12af417e7e905f30182122a95d73610e3acc9cb41829093d0265dfd6cca4225`
- Manifest status: `completed`
- Run units: `EXP-05-A1`
- Planned variant count in manifest/plan: `5`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-05/runs/aws-a1-pytorch-20260715T003637Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status.txt`=0
- Finished UTC markers: `finished_utc.txt`=2026-07-15T00:43:10Z
- Raw artifact file count below `raw/`: `11`

## Measured Results From `aws-a1-pytorch-20260715T003637Z`

| Variant | Backend | Compile | Mean iteration ms | Peak MiB | Warmup/compile s | Checksum |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-05-A1-sdpa-math-bf16 | math | false | 4.777 | 1408.1 | 0.209 | -659.63 |
| EXP-05-A1-sdpa-auto-bf16 | automatic | false | 0.171 | 64.3 | 0.01 | -659.116 |
| EXP-05-A1-sdpa-flash-forced-bf16 | flash_attention | false | 0.171 | 64.3 | 0.011 | -659.116 |
| EXP-05-A1-sdpa-math-compile-bf16 | math | true | 1.972 | 32.0 | 3.058 | -659.629 |
| EXP-05-A1-sdpa-auto-compile-bf16 | automatic | true | 0.187 | 32.0 | 1.09 | -659.116 |

## Formal Report Status

- Report file: `experiments/exp_05_sdpa_flashattention_operator_fusion/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

Verify which attention backend actually ran. A fused attention path can matter more than small code-level changes.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.
