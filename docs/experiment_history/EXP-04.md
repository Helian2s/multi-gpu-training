# EXP-04: Activation checkpointing/recomputation

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-04`, `pytorch`, `aws`, `AWS-A1`

## Goal And Design

Educational goal: Learn when recomputing activations is a good memory trade, how much memory it saves, and how to measure the added compute cost.

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
| Baseline | EXP-04-A1-no-checkpointing-seq2048 |
| Declared variants | 4 |

## Current Conclusion

Activation checkpointing reduced peak memory at both sequence lengths and increased step time. At sequence length 4096, peak memory fell from about 28.7 GiB to 17.1 GiB, while step time rose from about 269 ms to 318 ms.

## Interpretation

The run shows the expected recomputation trade: store fewer activations during forward, recompute during backward, and pay extra compute time to fit longer or larger batches.

## Run Inventory

### aws-a1-pytorch-20260715T003637Z

- Run ID: `aws-a1-pytorch-20260715T003637Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-04/runs/aws-a1-pytorch-20260715T003637Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:e12af417e7e905f30182122a95d73610e3acc9cb41829093d0265dfd6cca4225`
- Manifest status: `completed`
- Run units: `EXP-04-A1`
- Planned variant count in manifest/plan: `4`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-04/runs/aws-a1-pytorch-20260715T003637Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status.txt`=0
- Finished UTC markers: `finished_utc.txt`=2026-07-15T00:42:50Z
- Raw artifact file count below `raw/`: `9`

## Measured Results From `aws-a1-pytorch-20260715T003637Z`

| Variant | Strategy | Precision | World | Microbatch | Accum | Global batch | Seq | Mean step ms | Peak GiB | Loss | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-04-A1-no-checkpointing-seq2048 | single | bf16 | 1 | 1 | 1 | 1 | 2048 | 154.49 | 19.22 | r0=7.398 | true |
| EXP-04-A1-hf-gradient-checkpointing-seq2048 | single | bf16 | 1 | 1 | 1 | 1 | 2048 | 177.517 | 16.09 | r0=7.397 | true |
| EXP-04-A1-no-checkpointing-seq4096 | single | bf16 | 1 | 1 | 1 | 1 | 4096 | 268.877 | 28.75 | r0=7.499 | true |
| EXP-04-A1-hf-gradient-checkpointing-seq4096 | single | bf16 | 1 | 1 | 1 | 1 | 4096 | 318.307 | 17.14 | r0=7.5 | true |

## Formal Report Status

- Report file: `experiments/exp_04_activation_checkpointing_recomputation/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

Use checkpointing when memory is the binding constraint; avoid it when the same workload already fits and throughput is the priority.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.
