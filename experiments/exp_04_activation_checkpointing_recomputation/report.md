# EXP-04: Activation checkpointing/recomputation - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-04.md](../../docs/experiment_history/EXP-04.md).

## Executive conclusion

Activation checkpointing reduced peak memory at both sequence lengths and increased step time. At sequence length 4096, peak memory fell from about 28.7 GiB to 17.1 GiB, while step time rose from about 269 ms to 318 ms.

## Run inventory

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

## Measured results

### `aws-a1-pytorch-20260715T003637Z` metrics

| Variant | Strategy | Precision | World | Microbatch | Accum | Global batch | Seq | Mean step ms | Peak GiB | Loss | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-04-A1-no-checkpointing-seq2048 | single | bf16 | 1 | 1 | 1 | 1 | 2048 | 154.49 | 19.22 | r0=7.398 | true |
| EXP-04-A1-hf-gradient-checkpointing-seq2048 | single | bf16 | 1 | 1 | 1 | 1 | 2048 | 177.517 | 16.09 | r0=7.397 | true |
| EXP-04-A1-no-checkpointing-seq4096 | single | bf16 | 1 | 1 | 1 | 1 | 4096 | 268.877 | 28.75 | r0=7.499 | true |
| EXP-04-A1-hf-gradient-checkpointing-seq4096 | single | bf16 | 1 | 1 | 1 | 1 | 4096 | 318.307 | 17.14 | r0=7.5 | true |

## Interpretation

The run shows the expected recomputation trade: store fewer activations during forward, recompute during backward, and pay extra compute time to fit longer or larger batches.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

Use checkpointing when memory is the binding constraint; avoid it when the same workload already fits and throughput is the priority.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.
