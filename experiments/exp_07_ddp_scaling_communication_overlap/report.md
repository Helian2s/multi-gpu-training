# EXP-07: DDP scaling and communication overlap - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-07.md](../../docs/experiment_history/EXP-07.md).

## Executive conclusion

The two-rank DDP variants completed on AWS-A2. One rank measured about 113 ms/step, while two-rank DDP with batch 2 measured about 240 ms/step; this is not a throughput speedup for the small one-sample-per-rank workload. A larger bucket variant was slightly faster than the default/small bucket variants.

## Run inventory

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

## Measured results

### `aws-a2-full-fp16fix-20260715T023252Z` metrics

| Variant | Strategy | Precision | World | Microbatch | Accum | Global batch | Seq | Mean step ms | Peak GiB | Loss | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-07-A2V1-one-rank-fixed-local | single | bf16 | 1 | 1 | 1 | 1 | 1024 | 112.708 | 16.09 | r0=5.822 | true |
| EXP-07-A2V2-two-rank-fixed-local | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 239.647 | 19.29 | r0=5.493; r1=6.229 | true |
| EXP-07-A2V2-two-rank-fixed-global | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 239.726 | 19.29 | r0=5.491; r1=6.226 | true |
| EXP-07-A2V2-two-rank-bucket-small | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 240.91 | 19.29 | r1=6.225; r0=5.495 | true |
| EXP-07-A2V2-two-rank-bucket-default | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 239.83 | 19.29 | r0=5.494; r1=6.228 | true |
| EXP-07-A2V2-two-rank-bucket-large | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 236.851 | 19.29 | r1=6.229; r0=5.496 | true |
| EXP-07-A2V2-two-rank-no-sync-accumulation | ddp | bf16 | 2 | 1 | 4 | 8 | 1024 | 453.723 | 20.86 | r0=6.176; r1=5.739 | true |

## Interpretation

For this bounded workload, communication/synchronization overhead and small local work dominated the benefit of adding a second GPU. No-sync gradient accumulation traded fewer synchronizations for larger effective batch and longer optimizer-step windows.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

DDP only scales when each rank has enough local compute to amortize gradient synchronization. Bucket settings and accumulation change that balance.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.
