# EXP-12: Pipeline schedules and bubble size

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-12`, `nemo-megatron`, `aws`, `AWS-A2`

## Goal And Design

Educational goal: Learn how pipeline stage balance, microbatch count, and schedule choice determine bubble overhead, activation memory, and throughput.

### Configuration Snapshot

| Field | Value |
| --- | --- |
| Framework/image family | nemo-megatron |
| Image | `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-nemo@sha256:ea7616a570d7e271eff25b4f3c0655a9910024e119171e2ead7569f6714b35fb` |
| Provider | aws |
| Compute profile | AWS-A2 |
| Resource type | g7e.12xlarge |
| GPU type | RTX PRO 6000 Blackwell Server Edition |
| Physical GPUs | 2 |
| Normal visible GPUs | variant-dependent |
| Workload profile | benchmark |
| Baseline | EXP-12-A2V1-pp1-mb4-balanced |
| Declared variants | 6 |

## Current Conclusion

Pipeline schedule mechanics were validated on AWS-A2. More microbatches reduced estimated bubble fraction, PP=2 introduced point-to-point activation/gradient traffic, and imbalanced stages were slower than balanced stages.

## Interpretation

The AWS PCIe run is valid for schedule/bubble mechanics but not as a Runpod NVLink Megatron throughput claim.

## Run Inventory

### aws-a2-megatron-exp12-20260715T0340Z

- Run ID: `aws-a2-megatron-exp12-20260715T0340Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-12/runs/aws-a2-megatron-exp12-20260715T0340Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-nemo@sha256:ea7616a570d7e271eff25b4f3c0655a9910024e119171e2ead7569f6714b35fb`
- Manifest status: `completed`
- Run units: `EXP-12-A2V2`
- Planned variant count in manifest/plan: `5`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-12/runs/aws-a2-megatron-exp12-20260715T0340Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status-EXP-12-A2V1.txt`=0, `exit_status-EXP-12-A2V2.txt`=0
- Finished UTC markers: `finished-EXP-12-A2V1-utc.txt`=2026-07-15T03:56:14Z, `finished-EXP-12-A2V2-utc.txt`=2026-07-15T03:57:19Z
- Raw artifact file count below `raw/`: `19`

## Measured Results From `aws-a2-megatron-exp12-20260715T0340Z`

| Variant | Schedule | PP | Microbatches | Bubble | Max step ms | Tokens/s | Peak MiB | P2P MiB | Stages |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-12-A2V1-pp1-mb4-balanced | no_pipeline | 1 | 4 | 0 | 3.074 | 666,128 | 155.0 | 0.0 | 4 |
| EXP-12-A2V2-pp2-flush-mb2-balanced | flush | 2 | 2 | 0.333 | 1.738 | 589,223 | 117.0 | 28.0 | 2 |
| EXP-12-A2V2-pp2-flush-mb8-balanced | flush | 2 | 8 | 0.111 | 5.312 | 771,084 | 177.0 | 112.0 | 2 |
| EXP-12-A2V2-pp2-1f1b-mb2-balanced | one_f_one_b | 2 | 2 | 0.333 | 1.777 | 576,400 | 116.0 | 28.0 | 2 |
| EXP-12-A2V2-pp2-1f1b-mb8-balanced | one_f_one_b | 2 | 8 | 0.111 | 5.586 | 733,266 | 122.0 | 112.0 | 2 |
| EXP-12-A2V2-pp2-1f1b-mb8-imbalanced | one_f_one_b | 2 | 8 | 0.111 | 6.905 | 593,198 | 136.0 | 112.0 | 3 |

## Formal Report Status

- Report file: `experiments/exp_12_pipeline_schedules_bubble_size/report.md`
- Report status line: Completed from AWS-A2 run `aws-a2-megatron-exp12-20260715T0340Z`

## Exam / Study Takeaway

Pipeline parallelism needs enough microbatches and balanced stage work. Splitting layers alone does not guarantee utilization.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.
