# EXP-02: Mixed precision and Tensor Cores in distributed training - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-02.md](../../docs/experiment_history/EXP-02.md).

## Executive conclusion

Reduced precision accelerated eligible matrix and training paths on AWS-A2. BF16 was the strongest measured training mode: one-rank BF16 cut step time from about 322 ms to 113 ms and peak memory from about 32.1 GiB to 16.1 GiB versus FP32. FP8 was not admitted because the native Transformer Engine SM120 path was not proven.

## Run inventory

### aws-a2-full-fp16fix-20260715T023252Z

- Run ID: `aws-a2-full-fp16fix-20260715T023252Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-02/runs/aws-a2-full-fp16fix-20260715T023252Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486`
- Manifest status: `completed`
- Run units: `EXP-02-A2V2`
- Planned variant count in manifest/plan: `3`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-02/runs/aws-a2-full-fp16fix-20260715T023252Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status-EXP-02-A2V1.txt`=0, `exit_status-EXP-02-A2V2.txt`=0
- Finished UTC markers: `finished-EXP-02-A2V1-utc.txt`=2026-07-15T02:35:08Z, `finished-EXP-02-A2V2-utc.txt`=2026-07-15T02:36:15Z
- Raw artifact file count below `raw/`: `30`

## Measured results

### `aws-a2-full-fp16fix-20260715T023252Z` metrics

GEMM microbenchmark variants:

| Variant | Precision | Shape | Matrix | Elapsed ms | TFLOP/s |
| --- | --- | --- | --- | --- | --- |
| EXP-02-A2V1-gemm-fp32-aligned | fp32 | tensor_core_aligned | 4096 | 89.527 | 76.759 |
| EXP-02-A2V1-gemm-fp32-misaligned | fp32 | deliberately_misaligned | 4097 | 99.251 | 69.288 |
| EXP-02-A2V1-gemm-tf32-aligned | tf32 | tensor_core_aligned | 4096 | 38.618 | 177.947 |
| EXP-02-A2V1-gemm-bf16-aligned | bf16 | tensor_core_aligned | 4096 | 18.626 | 368.953 |
| EXP-02-A2V1-gemm-fp16-aligned | fp16 | tensor_core_aligned | 4096 | 18.743 | 366.642 |

Training variants:

| Variant | Strategy | Precision | World | Microbatch | Accum | Global batch | Seq | Mean step ms | Peak GiB | Loss | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-02-A2V1-train-fp32-reference | single | fp32 | 1 | 1 | 1 | 1 | 1024 | 321.898 | 32.12 | r0=4.673 | true |
| EXP-02-A2V1-train-tf32 | single | tf32 | 1 | 1 | 1 | 1 | 1024 | 207.346 | 32.12 | r0=4.679 | true |
| EXP-02-A2V1-train-bf16 | single | bf16 | 1 | 1 | 1 | 1 | 1024 | 112.866 | 16.09 | r0=6.442 | true |
| EXP-02-A2V1-train-fp16 | single | fp16 | 1 | 1 | 1 | 1 | 1024 | 185.28 | 32.12 | r0=4.867 | true |
| EXP-02-A2V2-ddp-fp32-reference | ddp | fp32 | 2 | 1 | 1 | 2 | 1024 | 572.369 | 38.53 | r0=4.539; r1=4.462 | true |
| EXP-02-A2V2-ddp-bf16 | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 238.864 | 19.29 | r1=6.068; r0=6.118 | true |
| EXP-02-A2V2-ddp-fp16 | ddp | fp16 | 2 | 1 | 1 | 2 | 1024 | 437.124 | 38.52 | r1=4.634; r0=4.7 | true |

Skipped variants:

| Variant | Status | Detail |
| --- | --- | --- |
| EXP-02-A2V1-train-fp8-candidate | skipped | admission gate not satisfied: native_transformer_engine_sm120_path_detected |

## Interpretation

TF32 improved FP32-compatible matrix work, and BF16/FP16 GEMMs were the fastest microbenchmarks. In the bounded training workload, BF16 also reduced activation/parameter memory, while FP16 used FP32 parameters with autocast/GradScaler after the local executor fix.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

Do not choose a precision mode from marketing claims alone: record kernel eligibility, finite loss, memory, and distributed behavior. BF16 is the practical default here; unsupported FP8 remains excluded.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.
