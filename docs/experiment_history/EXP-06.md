# EXP-06: Profiler triangulation

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-06`, `pytorch`, `aws`, `AWS-A1`

## Goal And Design

Educational goal: Learn which profiler answers which performance question, and how to connect framework-level, system-timeline, and kernel-level evidence into one bottleneck diagnosis.

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
| Workload profile | profiler |
| Baseline | EXP-06-A1-torch-profiler-math |
| Declared variants | 4 |

## Current Conclusion

Profiler triangulation produced PyTorch Profiler traces/key averages for math and automatic attention backends, and confirmed Nsight Systems and Nsight Compute were available with prepared commands.

## Interpretation

The run separates profiler roles: PyTorch Profiler captured operator and timeline artifacts directly, while Nsight tools were validated for deeper host/CUDA/kernel inspection from an interactive host shell.

## Run Inventory

### aws-a1-pytorch-20260715T003637Z

- Run ID: `aws-a1-pytorch-20260715T003637Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-06/runs/aws-a1-pytorch-20260715T003637Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:e12af417e7e905f30182122a95d73610e3acc9cb41829093d0265dfd6cca4225`
- Manifest status: `completed`
- Run units: `EXP-06-A1`
- Planned variant count in manifest/plan: `4`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-06/runs/aws-a1-pytorch-20260715T003637Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status.txt`=0
- Finished UTC markers: `finished_utc.txt`=2026-07-15T00:43:17Z
- Raw artifact file count below `raw/`: `13`

## Measured Results From `aws-a1-pytorch-20260715T003637Z`

| Variant | Profiler | Target | Status | Artifact or command |
| --- | --- | --- | --- | --- |
| EXP-06-A1-torch-profiler-math | torch_profiler | math | completed | /workspace/multi-gpu-training/artifacts/runs/EXP-06/aws-a1-pytorch-20260715T003637Z/raw/EXP-06-A1-torch-profiler-math/torch_profiler_key_averages.txt |
| EXP-06-A1-torch-profiler-auto | torch_profiler | automatic | completed | /workspace/multi-gpu-training/artifacts/runs/EXP-06/aws-a1-pytorch-20260715T003637Z/raw/EXP-06-A1-torch-profiler-auto/torch_profiler_key_averages.txt |
| EXP-06-A1-nsight-systems-math | nsight_systems | math | ready | nsys profile python -m common.pytorch_executor --worker --config <experiment.yaml> --variant EXP-06-A1-nsight-systems-math --run-dir <run-dir> |
| EXP-06-A1-nsight-compute-auto | nsight_compute | automatic | ready | ncu profile python -m common.pytorch_executor --worker --config <experiment.yaml> --variant EXP-06-A1-nsight-compute-auto --run-dir <run-dir> |

## Formal Report Status

- Report file: `experiments/exp_06_profiler_triangulation/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

Use PyTorch Profiler for framework-level triage first, then Nsight Systems/Compute when the question requires CUDA timeline or kernel counter evidence.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.
