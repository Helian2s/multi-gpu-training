# Multi-GPU Training Lab

A personal engineering project exploring how GPU topology, numerical precision,
memory management, and parallelism affect single-node LLM training.

I built a containerized experiment harness and ran workloads on AWS RTX PRO
6000 Blackwell GPUs and Runpod A100 SXM GPUs. The work covers the full lab
workflow: preparing inputs, qualifying hardware, launching experiment queues,
collecting measurements, debugging failures, and preserving results for review.
It began as hands-on preparation for the GPU Acceleration and Optimization
domain of NCP-GENL and became a portfolio of distributed-systems experiments.

**14 experiments with recorded executions · 1–4 GPUs on one host · PyTorch and
NeMo/Megatron container environments · AWS and Runpod**

[Experiment guide](experiments/README.md) ·
[Results and run history](docs/experiment_history/README.md) ·
[Validation status](docs/validation-status.md)

## What I built

- **An experiment harness:** YAML specifications, variant selection, dry-run
  plans, isolated worker processes, `torchrun` launches, per-rank metrics, and
  execution manifests.
- **PyTorch workloads:** Qwen-based training steps with mixed precision,
  gradient accumulation, activation checkpointing, Distributed Data Parallel
  (DDP), and Fully Sharded Data Parallel (FSDP), plus GEMM and attention
  microbenchmarks.
- **Parallelism prototypes:** synthetic MLP, attention, and pipeline workloads
  for tensor, sequence, context, pipeline, and hybrid tensor/data parallelism.
  These use custom PyTorch distributed operations inside the NeMo/Megatron
  image; a full Qwen Megatron training recipe remains unfinished.
- **Cloud execution tooling:** three AWS queues and two Runpod queues, with GPU
  visibility controls, topology qualification, run timeouts, and artifact
  collection. AWS uses EC2, SSM, ECR, S3, and EBS; Runpod uses GHCR and Pod
  storage with operator-managed artifact copy-out.
- **Pinned inputs and container builds:** exact Qwen3 and WikiText revisions,
  deterministic token streams, file checksums, and two NVIDIA NGC-based image
  families for `linux/amd64`.
- **A reviewable experiment record:** hypotheses, configurations, expected
  results, reports, operational history, and local tests kept in Git. Large
  inputs and raw traces are stored separately.

## Hardware and workload

| Environment used | GPUs per host | Role in the project |
| --- | --- | --- |
| AWS G7e, RTX PRO 6000 Blackwell Server Edition, 96 GB per GPU | 1 or 2 | PyTorch training, precision, memory, profiling, PCIe communication, and the pipeline-schedule prototype |
| Runpod, A100-SXM4-80GB | 2 or 4 | NVLink communication and synthetic tensor, sequence, context, and hybrid parallelism |

Every distributed run stayed on one physical host. One-GPU baselines sometimes
used a visibility mask on a two-GPU host; the complete host still incurred cost.
The four-GPU run exercised the smallest non-trivial hybrid layout, TP=2 × DP=2.

The PyTorch training path loads **Qwen3-1.7B-Base** and tokenized
**WikiText-103**. Exact revisions are in [the input lock](configs/inputs.lock.yaml).
The intended objective is short, full-parameter continued pretraining. The
project measures execution behavior rather than training to convergence or
producing a deployable model. Known objective and distributed-correctness
issues are documented in [the validation review](docs/validation-status.md).

## Experiments performed

| Area | Experiments | Work performed |
| --- | --- | --- |
| Hardware communication | EXP-01, EXP-10 | Recorded GPU topology, CUDA peer access, P2P latency/bandwidth, and five NCCL collectives |
| Precision and memory | EXP-02–04, EXP-08 | Ran precision, batch/accumulation, checkpointing, and DDP/FSDP variants |
| Kernels and diagnostics | EXP-05–06, EXP-09 | Benchmarked attention backends and compilation, captured PyTorch traces, checked Nsight availability, and exercised controlled failure cases |
| Distributed execution | EXP-07, EXP-11–14 | Ran DDP scaling/overlap variants and synthetic TP/SP, pipeline, CP, and TP×DP prototypes |

The [experiment guide](experiments/README.md) links all 14 reports and explains
what each implementation actually measures.

## Selected observations

These values come from the July 2026 reports. They are historical measurements
from short runs; formal validation is still pending for the reports below.
Hardware and software differ between providers, so the communication rows do
not isolate interconnect as the only variable.

| Experiment | Recorded observation | Interpretation |
| --- | --- | --- |
| [AWS communication](experiments/exp_01_aws_pcie_p2p_nccl_communication/report.md) | P2P writes of **54.73–55.80 GB/s** per direction; `PIX` topology | The allocated PCIe host supported direct GPU peer access |
| [Runpod communication](experiments/exp_10_runpod_nvlink_p2p_nccl_communication/report.md) | P2P writes of **268.96–274.12 GB/s** per direction; `NV12` topology | The allocated A100 SXM host exposed NVLink connectivity |
| [Attention backends](experiments/exp_05_sdpa_flashattention_operator_fusion/report.md) | Math SDPA: **4.777 ms**; automatic and forced FlashAttention: **0.171 ms** per iteration | Backend selection had a large effect for this forward attention microbenchmark; this is not whole-model training speedup |
| [Activation checkpointing](experiments/exp_04_activation_checkpointing_recomputation/report.md) | At sequence length 4096, peak allocation changed from **28.75 to 17.14 GiB**, and step time from **268.877 to 318.307 ms** | The recorded run illustrates the memory/compute tradeoff, but used the training path affected by the label-alignment issue |

Other reports preserve observations about DDP scaling, FSDP memory use, pipeline
stage balance, and hybrid rank groups. Their conclusions should be read with
the implementation limits in [validation status](docs/validation-status.md).

## Engineering lessons

Qualification and failure handling became as important as the timed workloads.
The project encountered AWS capacity constraints, image-content mistakes, an
FP16 gradient-scaling issue, and an NCCL process-group cleanup hang. Those
events led to explicit execution queues, image/runtime checks, an FP16 parameter
precision fix, and a worker cleanup change. The
[operational timeline](docs/experiment_history/operational_timeline.md) records
the attempts and fixes.

A later source review also found that a successful process exit was too weak a
correctness gate. It identified label alignment, sample partitioning, and
distributed-gradient problems. The next development step is to strengthen
numerical validation and rerun affected comparisons.

## Project status

The catalog records **12 experiments awaiting formal report validation** and
**2 marked completed (EXP-12 and EXP-14)**. Those two completion records also
need review against the accepted workload and the subsequent correctness
findings. Report status describes the recorded lifecycle; it is not a guarantee
that every original hypothesis has been established.

The repository includes reports and text summaries of the runs. Raw artifact
mirrors, model weights, datasets, and profiler traces are excluded from Git.
Twelve experiment analysis entry points remain templates; EXP-12 and EXP-14
have table-generating analyzers. See [validation status](docs/validation-status.md)
for the remaining work and [artifact storage](artifacts/README.md) for access
and restoration details.

## Explore locally

Source inspection, repository checks, and dry-run plans require no GPU or cloud
account. From the repository root, use Python 3.12 or later, Git, and Make:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install PyYAML==6.0.3
make check PYTHON=.venv/bin/python
make exp-a1-dry-run PYTHON=.venv/bin/python
make exp-runpod-megatron-dry-run PYTHON=.venv/bin/python
```

`make check` runs syntax checks, unit tests, YAML/catalog/link validation, and
Git whitespace checks. It does not validate CUDA execution or benchmark
conclusions. [The test guide](tests/README.md) describes its coverage.

For input preparation, use [the data guide](data/README.md). For GPU execution,
start with [the infrastructure guide](infra/README.md): checked-in cloud
references describe previous runs and require fresh readiness checks. AWS
images were deleted during the recorded cleanup, and the Runpod image needs
the committed cleanup fix before reproducibility runs. Paid launches and image
publication require explicit authorization.

## Repository map

| Path | Contents |
| --- | --- |
| [experiments/](experiments/README.md) | Hypotheses, sweeps, launch entry points, expected results, and reports |
| [common/](common/README.md) | Shared runner, PyTorch and synthetic parallelism executors, GPU qualification |
| [infra/](infra/README.md) | AWS and Runpod queues, launch planning, operator commands, and readiness records |
| [containers/](containers/README.md) | NGC-based runtime images and compatibility notes |
| [configs/](configs/README.md), [data/](data/README.md) | Input identity, configuration examples, and preparation workflow |
| [scripts/](scripts/README.md), [tests/](tests/README.md) | Local tooling and regression checks |
| [docs/](docs/README.md), [artifacts/](artifacts/README.md) | Results navigation, validation review, and raw artifact layout |

[Project decisions](PROJECT_DECISIONS.md) and the
[experiment catalog](EXPERIMENT_CATALOG.md) retain the detailed design record.
Maintainer procedures and document authority are listed in the
[documentation guide](docs/README.md).
