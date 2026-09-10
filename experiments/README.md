# Experiment guide

The project contains 14 experiment implementations and reports from the July
2026 execution phase. Each experiment has one assigned provider and runs on one
physical host. This page describes the work performed; the
[catalog](../EXPERIMENT_CATALOG.md) owns the definitions and lifecycle status.

## What was run

| ID and report | Environment | Recorded work |
| --- | --- | --- |
| [EXP-01 — PCIe communication](exp_01_aws_pcie_p2p_nccl_communication/report.md) | AWS, 2 GPUs | CUDA peer access, P2P bandwidth/latency, and NCCL collective measurements |
| [EXP-02 — Mixed precision](exp_02_mixed_precision_tensor_cores/report.md) | AWS, 1/2 visible GPUs | GEMM and Qwen training variants across FP32/TF32/BF16/FP16; FP8 remained admission-gated |
| [EXP-03 — Batch geometry](exp_03_microbatch_gradient_accumulation/report.md) | AWS, 1 GPU | Microbatch and accumulation variants at a fixed nominal global batch |
| [EXP-04 — Activation checkpointing](exp_04_activation_checkpointing_recomputation/report.md) | AWS, 1 GPU | Memory and step-time comparisons with and without checkpointing at two sequence lengths |
| [EXP-05 — Attention backends](exp_05_sdpa_flashattention_operator_fusion/report.md) | AWS, 1 GPU | Forward SDPA math/automatic/forced-FlashAttention benchmarks, including compilation variants |
| [EXP-06 — Profiling](exp_06_profiler_triangulation/report.md) | AWS, 1 GPU | PyTorch traces and operator summaries; Nsight executable availability checks |
| [EXP-07 — DDP](exp_07_ddp_scaling_communication_overlap/report.md) | AWS, 1/2 visible GPUs | Replicated training, bucket-size variants, and accumulation with delayed synchronization |
| [EXP-08 — FSDP](exp_08_fsdp_sharding_zero_memory_tradeoffs/report.md) | AWS, 2 GPUs | DDP/FSDP memory and timing comparisons across sharding strategies |
| [EXP-09 — Failure diagnosis](exp_09_controlled_troubleshooting_failure_diagnosis/report.md) | AWS, 1/2 visible GPUs | CUDA smoke and controlled OOM, non-finite, input-wait, straggler, and collective-failure cases |
| [EXP-10 — NVLink communication](exp_10_runpod_nvlink_p2p_nccl_communication/report.md) | Runpod, 2 A100 SXM GPUs | Topology, CUDA peer access, P2P bandwidth/latency, and NCCL collectives |
| [EXP-11 — Tensor/sequence parallelism](exp_11_tensor_sequence_parallelism/report.md) | Runpod, 1/2 visible GPUs | Synthetic MLP shard and sequence-placement variants with collectives |
| [EXP-12 — Pipeline schedules](exp_12_pipeline_schedules_bubble_size/report.md) | AWS, 1/2 visible GPUs | Synthetic stages with flush/1F1B-style scheduling, microbatch sweeps, and stage imbalance |
| [EXP-13 — Context parallelism](exp_13_context_parallelism_long_sequences/report.md) | Runpod, 1/2 visible GPUs | Synthetic attention with sequence partitioning and K/V gathering |
| [EXP-14 — TP×DP hybrid](exp_14_tp2_dp2_hybrid/report.md) | Runpod, 4 A100 SXM GPUs | Synthetic DP=4, TP=4, and TP=2 × DP=2 layouts and rank-group measurements |

## Read results with their status

Twelve reports are interim: execution artifacts exist in the historical record,
but formal validation remains pending. EXP-12 and EXP-14 are marked completed
in the catalog and have table-generating analyzers. The other 12 `analyze.py`
files still contain the template error.

The [2026-09-10 source review](../docs/validation-status.md) found training-label,
sample-partitioning, and distributed-gradient issues. It also identified a gap
between the accepted Qwen training contract and the synthetic EXP-11–14
implementations. Read the review alongside every affected report, including
the two marked completed. A zero exit code is execution evidence, not a passed
numerical-equivalence test.

## Files in each experiment

| File | Purpose |
| --- | --- |
| `hypothesis.md` | Question, variables, rationale, and decision rule |
| `experiment.yaml` | Environment, workload, variants, launch settings, and intended measurements |
| `expected_results.yaml` | Declared correctness gates and interpretation criteria; not yet automatically enforced by the executors |
| `sources.bib` | Supporting research and documentation |
| `run_expNN.py` or `collect_expNN.sh` | Shared-executor entry point or communication collector |
| `analyze.py` | Analysis entry point; implementation status described above |
| `report.md` | Recorded environment, observations, interpretation, and limitations |

## Inspect plans locally

After [setting up the local check environment](../tests/README.md), run from the
repository root:

```bash
make exp-a1-dry-run PYTHON=.venv/bin/python
make exp-a2-dry-run PYTHON=.venv/bin/python
make exp-a2-megatron-dry-run PYTHON=.venv/bin/python
make exp-runpod-megatron-dry-run PYTHON=.venv/bin/python
```

These targets print selected variants and run-unit plans. They do not launch
GPU workers or cloud resources. Use [the infrastructure guide](../infra/README.md)
for provider execution, and [the artifact guide](../artifacts/README.md) for
restoring raw evidence.

## Adding or completing an experiment

1. Accept the experiment in the catalog before creating its directory.
2. Use `make new-experiment ID=EXP-NN SLUG=slug TITLE="Title"` from the repository
   root, following the continuous canonical ID sequence.
3. Define the hypothesis, configuration, correctness gates, and sources before
   measured execution. Shared behavior belongs in `common/`; provider actions
   belong in `infra/`.
4. Implement and test the workload and analyzer, then collect the required
   qualification and run artifacts on the assigned provider.
5. Validate results against the declared rules, write the report, and only then
   change the catalog status to `completed`.

Preserve original runs and their limitations when a correction requires a new
run. Use the [template](_template/hypothesis.md) for the file structure and the
[project decisions](../PROJECT_DECISIONS.md) for accepted scope.
