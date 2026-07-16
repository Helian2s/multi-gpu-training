# EXP-07: DDP scaling and communication overlap

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-07`, `pytorch`, `aws`, `AWS-A2`

## Goal And Design

Educational goal: Learn how to evaluate DDP speedup, communication overlap, bucket behavior, and local-batch effects when moving from one rank to two ranks on one server.

### Scenario

A two-GPU PyTorch job does not approach a 2x speedup. The team needs to
distinguish between insufficient per-rank computation, expensive all-reduce,
poor overlap, bucket behavior, and synchronization frequency before changing
the model or requesting larger hardware.

### Question

When does replicated DDP scale well from one visible GPU to two visible GPUs on
one `AWS-A2` host, and when do gradient synchronization and small local batches
dominate the step?

### Hypothesis

The two-rank DDP run will scale better when each rank has enough local work to
hide all-reduce. Fixed-global-batch scaling should lose efficiency because work
per rank is smaller. Local accumulation with `no_sync` should reduce
synchronization frequency when gradients are synchronized before each optimizer
step.

### Decision rule

Confirm the hypothesis when two-rank efficiency tracks the amount of local work
per rank and profiler evidence shows whether all-reduce overlaps backward work
or extends the critical path.

Reject it when scaling behavior cannot be explained by local work,
communication timing, bucket readiness, or correctness-preserving accumulation.

Leave it inconclusive when profiling timelines, rank logs, loss checks, input
identity, or image identity are incomplete.

### Configuration Snapshot

| Field | Value |
| --- | --- |
| Framework/image family | pytorch |
| Image | `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486` |
| Provider | aws |
| Compute profile | AWS-A2 |
| Resource type | g7e.12xlarge |
| GPU type | RTX PRO 6000 Blackwell Server Edition |
| Physical GPUs | 2 |
| Normal visible GPUs | variant-dependent |
| Workload profile | benchmark |
| Baseline | EXP-07-A2V1-one-rank-baseline |
| Declared variants | 7 |

## Current Conclusion

The two-rank DDP variants completed on AWS-A2. One rank measured about 113 ms/step, while two-rank DDP with batch 2 measured about 240 ms/step; this is not a throughput speedup for the small one-sample-per-rank workload. A larger bucket variant was slightly faster than the default/small bucket variants.

## Interpretation

For this bounded workload, communication/synchronization overhead and small local work dominated the benefit of adding a second GPU. No-sync gradient accumulation traded fewer synchronizations for larger effective batch and longer optimizer-step windows.

## Run Inventory

### aws-a2-full-fp16fix-20260715T023252Z

- Run ID: `aws-a2-full-fp16fix-20260715T023252Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-07/runs/aws-a2-full-fp16fix-20260715T023252Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486`
- Manifest status: `completed`
- Run units: `EXP-07-A2V2`
- Planned variant count in manifest/plan: `6`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-07/runs/aws-a2-full-fp16fix-20260715T023252Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status-EXP-07-A2V1.txt`=0, `exit_status-EXP-07-A2V2.txt`=0
- Finished UTC markers: `finished-EXP-07-A2V1-utc.txt`=2026-07-15T02:36:28Z, `finished-EXP-07-A2V2-utc.txt`=2026-07-15T02:38:28Z
- Raw artifact file count below `raw/`: `22`

## Measured Results From `aws-a2-full-fp16fix-20260715T023252Z`

| Variant | Strategy | Precision | World | Microbatch | Accum | Global batch | Seq | Mean step ms | Peak GiB | Loss | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-07-A2V1-one-rank-fixed-local | single | bf16 | 1 | 1 | 1 | 1 | 1024 | 112.708 | 16.09 | r0=5.822 | true |
| EXP-07-A2V2-two-rank-fixed-local | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 239.647 | 19.29 | r0=5.493; r1=6.229 | true |
| EXP-07-A2V2-two-rank-fixed-global | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 239.726 | 19.29 | r0=5.491; r1=6.226 | true |
| EXP-07-A2V2-two-rank-bucket-small | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 240.91 | 19.29 | r1=6.225; r0=5.495 | true |
| EXP-07-A2V2-two-rank-bucket-default | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 239.83 | 19.29 | r0=5.494; r1=6.228 | true |
| EXP-07-A2V2-two-rank-bucket-large | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 236.851 | 19.29 | r1=6.229; r0=5.496 | true |
| EXP-07-A2V2-two-rank-no-sync-accumulation | ddp | bf16 | 2 | 1 | 4 | 8 | 1024 | 453.723 | 20.86 | r0=6.176; r1=5.739 | true |

## Formal Report Status

- Report file: `experiments/exp_07_ddp_scaling_communication_overlap/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

DDP only scales when each rank has enough local compute to amortize gradient synchronization. Bucket settings and accumulation change that balance.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.
