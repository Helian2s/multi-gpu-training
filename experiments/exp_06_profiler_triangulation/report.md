# EXP-06: Profiler triangulation - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-06.md](../../docs/experiment_history/EXP-06.md).

## Executive conclusion

Profiler triangulation produced PyTorch Profiler traces/key averages for math and automatic attention backends, and confirmed Nsight Systems and Nsight Compute were available with prepared commands.

## Run inventory

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

## Measured results

### `aws-a1-pytorch-20260715T003637Z` metrics

| Variant | Profiler | Target | Status | Artifact or command |
| --- | --- | --- | --- | --- |
| EXP-06-A1-torch-profiler-math | torch_profiler | math | completed | /workspace/multi-gpu-training/artifacts/runs/EXP-06/aws-a1-pytorch-20260715T003637Z/raw/EXP-06-A1-torch-profiler-math/torch_profiler_key_averages.txt |
| EXP-06-A1-torch-profiler-auto | torch_profiler | automatic | completed | /workspace/multi-gpu-training/artifacts/runs/EXP-06/aws-a1-pytorch-20260715T003637Z/raw/EXP-06-A1-torch-profiler-auto/torch_profiler_key_averages.txt |
| EXP-06-A1-nsight-systems-math | nsight_systems | math | ready | nsys profile python -m common.pytorch_executor --worker --config <experiment.yaml> --variant EXP-06-A1-nsight-systems-math --run-dir <run-dir> |
| EXP-06-A1-nsight-compute-auto | nsight_compute | automatic | ready | ncu profile python -m common.pytorch_executor --worker --config <experiment.yaml> --variant EXP-06-A1-nsight-compute-auto --run-dir <run-dir> |

## Interpretation

The run separates profiler roles: PyTorch Profiler captured operator and timeline artifacts directly, while Nsight tools were validated for deeper host/CUDA/kernel inspection from an interactive host shell.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

Use PyTorch Profiler for framework-level triage first, then Nsight Systems/Compute when the question requires CUDA timeline or kernel counter evidence.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.
