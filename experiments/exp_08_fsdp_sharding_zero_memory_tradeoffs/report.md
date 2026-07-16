# EXP-08: FSDP sharding and ZeRO-style memory trade-offs - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-08.md](../../docs/experiment_history/EXP-08.md).

## Executive conclusion

FSDP completed on two AWS GPUs and reduced peak memory versus DDP from about 19.3 GiB to 14.5 GiB per rank while also reducing step time in this short run from about 239 ms to about 205 ms. The prefetch variant was skipped because its admission gate was not met.

## Run inventory

### aws-a2-full-fp16fix-20260715T023252Z

- Run ID: `aws-a2-full-fp16fix-20260715T023252Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-08/runs/aws-a2-full-fp16fix-20260715T023252Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486`
- Manifest status: `completed`
- Run units: `EXP-08-A2V2`
- Planned variant count in manifest/plan: `4`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-08/runs/aws-a2-full-fp16fix-20260715T023252Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status-EXP-08-A2V2.txt`=0
- Finished UTC markers: `finished-EXP-08-A2V2-utc.txt`=2026-07-15T02:39:16Z
- Raw artifact file count below `raw/`: `11`

## Measured results

### `aws-a2-full-fp16fix-20260715T023252Z` metrics

| Variant | Strategy | Precision | World | Microbatch | Accum | Global batch | Seq | Mean step ms | Peak GiB | Loss | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-08-A2V2-ddp-replicated-baseline | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 238.609 | 19.29 | r0=6.114; r1=6.063 | true |
| EXP-08-A2V2-fsdp-shard-grad-optim | fsdp | bf16 | 2 | 1 | 1 | 2 | 1024 | 205.459 | 14.49 | r1=6.067; r0=6.116 | true |
| EXP-08-A2V2-fsdp-full-shard | fsdp | bf16 | 2 | 1 | 1 | 2 | 1024 | 205.73 | 14.49 | r0=6.117; r1=6.068 | true |
| EXP-08-A2V2-fsdp-full-shard-prefetch | None | None | 2 | None | None | None | None |  |  |  |  |

## Interpretation

The measured FSDP variants sharded model/gradient/optimizer state enough to lower memory. The result is encouraging but remains a short controlled run, not a general claim that FSDP is always faster.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

Use sharding to trade extra distributed machinery for lower per-rank state memory. Validate correctness and state-dict behavior before treating FSDP as a production recipe.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.
