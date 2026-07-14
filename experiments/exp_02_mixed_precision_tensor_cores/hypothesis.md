# EXP-02: Mixed precision and Tensor Cores in distributed training

Design status: Draft

Catalog entry: [EXPERIMENT_CATALOG.md](../../EXPERIMENT_CATALOG.md)

## Scenario

A financial-services training job is stable in FP32 but too slow and memory
hungry. The team wants a reduced-precision configuration on the accepted AWS
G7e hardware, but must prove that the chosen mode actually uses accelerated
matrix paths and does not hide numerical instability or DDP communication cost.

## Question

How do FP32, TF32, BF16, FP16, and admissible FP8 candidates affect Tensor Core
eligibility, throughput, memory, numerical behavior, and one-to-two-rank DDP
scaling on `AWS-A2`?

## Hypothesis

Tensor-Core-compatible shapes and reduced precision will improve throughput and
lower memory use relative to the FP32 reference. BF16 should be more robust than
FP16 because it keeps a wider exponent range. TF32 should accelerate eligible
FP32 matrix operations while retaining FP32 storage. Any FP8 result should count
only if the pinned image proves a native Transformer Engine path on the selected
GPU and records whether DDP communication bytes changed.

## Rationale

The selected GPU can expose different fast paths depending on datatype, matrix
shape, framework flags, and kernel implementation. A pure tokens-per-second
result is not enough; the experiment must also record kernel evidence, memory,
loss/gradient agreement, overflow behavior, and NCCL communication timing.

## Variables

- Independent variables: precision mode, GEMM shape alignment, and visible GPU
  count.
- Fixed variables: one AWS `g7e.12xlarge` host, pinned model and dataset
  inputs, restored checkpoint per variant, fixed sequence length, fixed seeds,
  and one immutable PyTorch image.
- Dependent metrics: GEMM throughput, training throughput, Tensor Core kernel
  evidence, peak memory, loss/gradient deviation, non-finite events, DDP
  efficiency, and NCCL communication time/bytes.
- Baseline: `EXP-02-A2V1` FP32 reference with one visible GPU, plus
  `EXP-02-A2V2` FP32 reference for the bounded two-rank check.

## Correctness gates

- AWS-A2 qualification and input manifest verification pass.
- Compared variants use the same checkpoint, input batches, seeds, and optimizer
  settings.
- The FP32 reference produces finite loss and gradients.
- TF32 and BF16 stay within the predeclared tolerance of the FP32 reference.
- FP16 is either finite with gradient scaling or explicitly recorded as
  numerically unstable.
- Unsupported or fallback FP8 is recorded but excluded from native-precision
  comparisons.
- DDP variants use exactly two ranks on one host and record the NCCL datatype
  and bytes.

## Decision rule

Confirm the hypothesis when reduced-precision variants show faster eligible
kernels and higher training throughput without violating their correctness
gates, and when the DDP result explains whether communication became a larger
fraction of step time.

Reject it when reduced precision fails correctness gates or does not produce
evidence of accelerated native execution on the selected shapes.

Leave it inconclusive when kernel evidence, numerical comparison, DDP logs,
input identity, or image identity is incomplete.

## Invalidating conditions

- The run is not on `AWS-A2` or does not record the physical host profile.
- The immutable image digest, driver, CUDA, PyTorch, NCCL, or Transformer Engine
  version is missing.
- Compared variants use different input batches, checkpoints, model revisions,
  or seeds.
- A precision mode silently falls back but is reported as native accelerated
  execution.
- Artifacts are not durably staged out.
