# EXP-03: Microbatch, global batch, and gradient accumulation - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-03.md](../../docs/experiment_history/EXP-03.md).

## Executive conclusion

With fixed effective global batch 8 on one AWS GPU, larger microbatches reduced optimizer-step time until microbatch 4, then memory pressure rose sharply at microbatch 8. All variants produced finite loss.

## Run inventory

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

## Measured results

### `aws-a1-pytorch-20260715T003637Z` metrics

| Variant | Strategy | Precision | World | Microbatch | Accum | Global batch | Seq | Mean step ms | Peak GiB | Loss | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-03-A1-mb1-acc8-global8 | single | bf16 | 1 | 1 | 8 | 8 | 1024 | 573.475 | 17.65 | r0=6.915 | true |
| EXP-03-A1-mb2-acc4-global8 | single | bf16 | 1 | 2 | 4 | 8 | 1024 | 477.162 | 22.42 | r0=6.913 | true |
| EXP-03-A1-mb4-acc2-global8 | single | bf16 | 1 | 4 | 2 | 8 | 1024 | 465.284 | 31.95 | r0=6.915 | true |
| EXP-03-A1-mb8-acc1-global8 | single | bf16 | 1 | 8 | 1 | 8 | 1024 | 479.772 | 47.81 | r0=6.916 | true |

## Interpretation

More accumulation steps lower peak memory but increase repeated forward/backward overhead per optimizer update. The best point in this short run was microbatch 4 with accumulation 2, not the largest microbatch.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

Tune microbatch and accumulation together. Gradient accumulation controls effective batch without requiring the largest possible microbatch.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.
