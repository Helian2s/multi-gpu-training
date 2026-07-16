# EXP-05: PyTorch SDPA/FlashAttention and operator fusion - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-05.md](../../docs/experiment_history/EXP-05.md).

## Executive conclusion

Automatic/FlashAttention SDPA was dramatically faster and smaller than the math backend for the Qwen-like GQA shape. Automatic SDPA measured about 0.171 ms/iteration versus 4.777 ms/iteration for math.

## Run inventory

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

## Measured results

### `aws-a1-pytorch-20260715T003637Z` metrics

| Variant | Backend | Compile | Mean iteration ms | Peak MiB | Warmup/compile s | Checksum |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-05-A1-sdpa-math-bf16 | math | false | 4.777 | 1408.1 | 0.209 | -659.63 |
| EXP-05-A1-sdpa-auto-bf16 | automatic | false | 0.171 | 64.3 | 0.01 | -659.116 |
| EXP-05-A1-sdpa-flash-forced-bf16 | flash_attention | false | 0.171 | 64.3 | 0.011 | -659.116 |
| EXP-05-A1-sdpa-math-compile-bf16 | math | true | 1.972 | 32.0 | 3.058 | -659.629 |
| EXP-05-A1-sdpa-auto-compile-bf16 | automatic | true | 0.187 | 32.0 | 1.09 | -659.116 |

## Interpretation

Backend selection dominated this attention microbenchmark. `torch.compile` reduced math-backend steady-state time but added compile warmup, while automatic compiled SDPA was slightly slower than automatic eager in this short measurement.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

Verify which attention backend actually ran. A fused attention path can matter more than small code-level changes.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.
