# EXP-02: Mixed precision and Tensor Cores in distributed training

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-02`, `pytorch`, `aws`, `AWS-A2`

## Goal And Design

Educational goal: Learn how precision modes, Tensor Core eligibility, numerical stability, and DDP communication interact so a faster precision choice is backed by both performance and correctness evidence.

### Scenario

A financial-services training job is stable in FP32 but too slow and memory
hungry. The team wants a reduced-precision configuration on the accepted AWS
G7e hardware, but must prove that the chosen mode actually uses accelerated
matrix paths and does not hide numerical instability or DDP communication cost.

### Question

How do FP32, TF32, BF16, FP16, and admissible FP8 candidates affect Tensor Core
eligibility, throughput, memory, numerical behavior, and one-to-two-rank DDP
scaling on `AWS-A2`?

### Hypothesis

Tensor-Core-compatible shapes and reduced precision will improve throughput and
lower memory use relative to the FP32 reference. BF16 should be more robust than
FP16 because it keeps a wider exponent range. TF32 should accelerate eligible
FP32 matrix operations while retaining FP32 storage. Any FP8 result should count
only if the pinned image proves a native Transformer Engine path on the selected
GPU and records whether DDP communication bytes changed.

### Decision rule

Confirm the hypothesis when reduced-precision variants show faster eligible
kernels and higher training throughput without violating their correctness
gates, and when the DDP result explains whether communication became a larger
fraction of step time.

Reject it when reduced precision fails correctness gates or does not produce
evidence of accelerated native execution on the selected shapes.

Leave it inconclusive when kernel evidence, numerical comparison, DDP logs,
input identity, or image identity is incomplete.

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
| Baseline | EXP-02-A2V1-fp32-reference |
| Declared variants | 13 |

## Current Conclusion

Reduced precision accelerated eligible matrix and training paths on AWS-A2. BF16 was the strongest measured training mode: one-rank BF16 cut step time from about 322 ms to 113 ms and peak memory from about 32.1 GiB to 16.1 GiB versus FP32. FP8 was not admitted because the native Transformer Engine SM120 path was not proven.

## Interpretation

TF32 improved FP32-compatible matrix work, and BF16/FP16 GEMMs were the fastest microbenchmarks. In the bounded training workload, BF16 also reduced activation/parameter memory, while FP16 used FP32 parameters with autocast/GradScaler after the local executor fix.

## Run Inventory

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

## Measured Results From `aws-a2-full-fp16fix-20260715T023252Z`

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

## Formal Report Status

- Report file: `experiments/exp_02_mixed_precision_tensor_cores/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

Do not choose a precision mode from marketing claims alone: record kernel eligibility, finite loss, memory, and distributed behavior. BF16 is the practical default here; unsupported FP8 remains excluded.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.
