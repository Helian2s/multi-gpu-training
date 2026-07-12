# Project decisions

Status: Accepted
Last updated: 2026-07-11

## Document role and authority

This file is the single source of truth for **accepted project-wide decisions**.
It deliberately contains both:

- A **current snapshot** in the topical sections below, optimized for quickly
  determining how the project operates now.
- An **append-only decision log**, optimized for understanding when and why the
  current state was chosen and what a later decision supersedes.

Experiment candidates, hypotheses, measurements, priorities, and selection
status belong in [EXPERIMENT_CATALOG.md](EXPERIMENT_CATALOG.md). A proposed
project-wide choice in the catalog does not become accepted merely because an
experiment uses it; once approved, it must be recorded here. If the two files
conflict on a project-wide constraint, this file takes precedence.

When a project-wide decision changes, update the current snapshot and append a
new log entry in the same commit. Do not rewrite an earlier log entry except to
correct a typo or broken reference. A new entry must identify any decision it
supersedes. The current snapshot, not the reader's reconstruction of the log,
governs experiments.

## Purpose

This repository is an experimental lab for the **GPU Acceleration and
Optimization** domain of the NVIDIA Certified Professional: Generative AI LLMs
(NCP-GENL) certification. The experiments should turn the expected behavior
described by research papers, NVIDIA documentation, and established best
practices into measurements that can be reproduced and explained.

The project is deliberately narrower than complete NCP-GENL preparation. It
focuses on GPU-accelerated LLM training: single-node distributed execution,
parallelism, memory and batch optimization, performance profiling, and
troubleshooting. Data preparation, RAG, production deployment, production
monitoring, safety, and general model evaluation are out of scope except where
a minimal piece is required to run or interpret an optimization experiment.

## Accepted infrastructure decisions

1. **Cloud provider:** Runpod.
2. **GPU type:** NVIDIA A100 SXM. Every result must record the exact GPU name,
   memory capacity, driver version, and local GPU topology reported by the
   allocated machine.
3. **One physical server only:** every experiment runs in one Runpod Pod on one
   physical host. We will not configure multi-node training or attempt to join
   two independent multi-GPU Pods.
4. **GPU count:** every experiment uses one to four GPUs. Allocations larger
   than four GPUs are out of scope.
5. **No full 5D experiment:** the project will study parallelism dimensions
   separately and in useful smaller combinations, but will not run a complete
   five-dimensional parallelism layout.
6. **Container registry:** GitHub Container Registry (GHCR). Project images use
   names under `ghcr.io/<github-owner>/<image>:<tag>`.
7. **Local-first workflow:** code, configuration, documentation, container build
   definitions, and most data preparation happen in this repository. A Runpod
   Pod should be started only for GPU validation, profiling, or an experiment
   run.

## Accepted framework decisions

The only training frameworks used to implement experiments are:

- PyTorch, for transparent low-level experiments and direct use of
  `torch.distributed`.
- NVIDIA NeMo Framework with Megatron Core, for NVIDIA's optimized LLM training
  stack and higher-level model-parallel experiments.

The Ultra-Scale Playbook may be used as a curriculum, conceptual reference, and
source of hypotheses. Its Nanotron implementation will not be used. We will not
add DeepSpeed, Hugging Face Accelerate, Colossal-AI, JAX, TensorFlow, or another
distributed-training framework merely to reproduce an example. An experiment
that the selected frameworks cannot reasonably express must be redesigned or
documented as out of scope.

Using a library for data reading or tokenization does not make it an approved
training framework. Such a library may be added only when an experiment needs
it and must be pinned in that experiment's environment.

## Container strategy

We maintain two reproducible image families rather than installing the training
stack after a billable Pod starts:

1. **PyTorch image:** derived from an NVIDIA NGC PyTorch container. It is used
   for native PyTorch, DDP, FSDP, collective-communication, precision, memory,
   kernel, and profiler experiments. Its upstream reference has the form
   `nvcr.io/nvidia/pytorch:<pinned-release>-py3`.
2. **NeMo/Megatron image:** derived from an NVIDIA NGC NeMo Framework Training
   container. It is used for Megatron Core/NeMo implementations of tensor,
   pipeline, context, and sequence parallelism and related optimized kernels.
   Its upstream reference has the form
   `nvcr.io/nvidia/nemo:<pinned-release>`.

The NeMo container is itself based on NVIDIA's optimized PyTorch stack; these
are two experiment environments, not containers that manage one another. Each
experiment runs in exactly one selected image.

Images are built locally with Docker Buildx or by GitHub Actions and pushed to
GHCR. Runpod pulls the resulting image. Base images and Python dependencies must
be pinned to immutable versions (and preferably image digests); the `latest`
tag is forbidden for recorded experiments.

The local workstation may build `linux/amd64` images, but GPU execution and
validation occur on Runpod. We do not expect an NVIDIA GPU or CUDA driver on the
local workstation.

## Experiment scale and coverage

The proposed experiment inventory and its selection status are maintained in
[EXPERIMENT_CATALOG.md](EXPERIMENT_CATALOG.md). An experiment becomes committed
work only after its `Decision` entry is changed from `TBD` to `accept`. Project
scope and exclusions remain authoritative in this file regardless of catalog
status.

For experiments that train a model, the accepted training mode is
full-parameter continued pretraining with autoregressive next-token
cross-entropy. The runs exercise and validate the complete training path; they
are not intended to train a model to convergence or produce a final model
artifact.

The normal progression is:

| Scale | Primary purpose |
| --- | --- |
| 1 GPU | Correctness, timing baseline, memory baseline, mixed precision, kernels, profiling |
| 2 GPUs | DDP/FSDP, collectives, tensor or pipeline parallelism, communication effects |
| 4 GPUs | Scaling studies and two-dimensional hybrid layouts |

Allowed topics include:

- CPU versus GPU and single-GPU baselines where they clarify acceleration.
- FP32, TF32, BF16, and FP16 behavior on A100. FP8 remains theory-only because
  A100 Tensor Cores do not support native FP8 execution.
- Tensor Core utilization and shape/alignment effects.
- Data parallelism with PyTorch DDP.
- Parameter, gradient, and optimizer-state sharding with PyTorch FSDP.
- Tensor, pipeline, sequence, and context parallelism where supported by
  NeMo/Megatron.
- Selected hybrid parallel layouts that fit on at most four GPUs.
- Gradient accumulation, microbatch size, global batch size, activation
  checkpointing, and optimizer/memory trade-offs.
- NCCL collective behavior within one server over PCIe, NVLink, and/or NVSwitch
  as exposed by the allocated A100 SXM host.
- Data-loading and CPU bottlenecks when they affect GPU utilization.
- Kernel, CUDA, NCCL, CPU, memory, and end-to-end timeline profiling.
- Failure diagnosis, out-of-memory behavior, reproducibility, and distributed
  checkpoint correctness.

## Accepted exclusions

The following are excluded from implementation in this project. They may still
be discussed theoretically when useful for NCP-GENL preparation.

- Multi-node and inter-datacenter training.
- Runpod Instant Clusters for this project.
- Two-Pod distributed training and inter-node versus intra-node benchmarks.
- A complete 5D parallelism layout.
- Any experiment requiring more than four GPUs.
- Kubernetes, Slurm, or another cluster scheduler.
- Production serving, orchestration, RAG, and application monitoring.
- A native FP8 training benchmark on A100; FP8 remains a theory and
  hardware-generation comparison topic.
- A CPU-offload experiment that would require introducing a second, larger
  dense model solely to exceed A100 memory.
- Mixture-of-Experts and expert-parallel training experiments.
- Nanotron, DeepSpeed, Accelerate, Colossal-AI, JAX, TensorFlow, or another
  distributed-framework comparison.
- Full pretraining, training to convergence, or reproduction at paper-scale
  model and cluster sizes.
- MIG partitioning and other resource multi-tenancy experiments.

## Approved tools and libraries

This is the final baseline toolset. A tool being listed does not mean every
experiment must use it. Versions are pinned in the Dockerfiles and dependency
lock files; they are not duplicated here because compatible versions move
together with the selected NGC container release.

### Local development, automation, and storage

| Tool | Role |
| --- | --- |
| Git and GitHub | Version control, source hosting, and review |
| GitHub Actions | Optional reproducible `linux/amd64` image builds and checks |
| GitHub Container Registry (GHCR) | Storage for the two project image families |
| Docker with BuildKit/Buildx | Build and inspect OCI container images |
| Runpod and `runpodctl` | Provision, inspect, connect to, and stop GPU Pods |
| OpenSSH and `rsync` | Interactive access and transfer of code/results when Git or persistent storage is unsuitable |
| GNU Make | Stable local entry points for build, upload, run, collect, and analysis commands |
| Bash | Small launch and collection scripts inside the Linux containers |
| `jq` | Inspection and transformation of CLI/API JSON output |

### Container runtime and NVIDIA compute stack

| Tool/library | Role |
| --- | --- |
| NVIDIA NGC PyTorch container | Base environment for low-level PyTorch experiments |
| NVIDIA NGC NeMo Framework Training container | Base environment for NeMo/Megatron experiments |
| NVIDIA GPU driver | Host-side access to the A100 GPUs; supplied by Runpod |
| CUDA Toolkit/runtime | GPU execution, compilation utilities, and CUDA libraries |
| cuDNN | Optimized deep-learning primitives |
| cuBLAS/cuBLASLt | Dense matrix multiplication and Tensor Core paths |
| NCCL | Single-node multi-GPU collectives and peer communication |
| Python | Experiment implementation and analysis runtime |
| PyTorch | Tensor operations, autograd, compilation, DDP, FSDP, and profiling |
| NeMo Framework | NVIDIA high-level LLM training recipes and configuration |
| Megatron Core | NVIDIA model-parallel LLM building blocks |
| Transformer Engine | Optimized transformer layers and supported low-precision execution |
| NVIDIA Apex | Only when included or required by the selected compatible NVIDIA stack |
| NVTX | Named ranges that connect application phases to profiler timelines |

### Measurement, profiling, and diagnostics

| Tool/library | Role |
| --- | --- |
| `nvidia-smi` | GPU identity, driver, processes, utilization, memory, power, clocks, and topology |
| NVIDIA Management Library (NVML) | Scriptable GPU telemetry when `nvidia-smi` sampling is insufficient |
| NVIDIA DCGM / `dcgmi` | GPU health, diagnostics, and longer-running telemetry when available in the Pod |
| NVIDIA Nsight Systems (`nsys`) | End-to-end CPU, CUDA, kernel, NCCL, and NVTX timeline analysis |
| NVIDIA Nsight Compute (`ncu`) | Detailed kernel metrics and roofline-style kernel analysis |
| PyTorch Profiler | Operator, CPU, CUDA, memory, shape, and distributed traces |
| TensorBoard | Viewing PyTorch profiler traces and experiment scalar series |
| `nccl-tests` | Controlled latency, bandwidth, and collective scaling measurements |
| CUDA Samples (`bandwidthTest`, `p2pBandwidthLatencyTest`) | Host/device and GPU peer-path validation where available |
| `torch.utils.benchmark` | Repeatable Python-level microbenchmarks |
| `psutil` | CPU, process, and host-memory telemetry collected by experiment scripts |

Some low-level counters and diagnostics depend on permissions exposed by the
Runpod host. If DCGM, Nsight Compute hardware counters, or another approved
diagnostic is unavailable inside the container, the experiment must record the
limitation rather than silently omit the intended measurement.

### Experiment configuration, validation, and analysis

| Tool/library | Role |
| --- | --- |
| YAML and PyYAML | Human-readable experiment and expected-result configuration |
| NumPy | Numerical calculations and array processing |
| pandas | Metrics tables, aggregation, and exported CSV/Parquet results |
| Matplotlib and Seaborn | Static plots for reports |
| SciPy | Confidence intervals and statistical comparisons when required |
| pytest | Correctness checks for training, sharding, checkpoint, and analysis code |
| Ruff | Python linting and formatting |

Additional libraries must be justified by a concrete experiment, pinned, and
added to this decision record before becoming shared project infrastructure.
Large hosted experiment platforms are intentionally unnecessary: raw metrics,
profiler artifacts, analysis outputs, and reports remain reproducible from the
repository and project storage.

## Reproducibility requirements

Every recorded run must preserve at least:

- Git commit and whether the worktree was dirty.
- Container image reference and digest.
- Runpod Pod/GPU selection and GPU count.
- GPU name and memory, NVIDIA driver, CUDA, cuDNN, NCCL, PyTorch, NeMo,
  Megatron Core, and Transformer Engine versions as applicable.
- `nvidia-smi topo -m` output and relevant NCCL environment variables.
- Experiment configuration, random seeds, model shape, sequence length,
  precision, microbatch/global batch sizes, and number of warm-up/measured
  iterations.
- Raw timing, throughput, memory, utilization, profiler, and correctness data.
- Estimated Runpod duration and cost.
- Analysis code, expected result, observed result, and conclusion.

Results must be copied to persistent storage before a Pod is stopped. Generated
checkpoints and large profiler traces are not committed to Git; small metrics,
configuration, analysis code, and the final report are.

## Decision-change rule

This file records project-wide constraints. Changing provider, GPU family,
maximum scale, framework set, registry, certification scope, shared training
workload, or accepted exclusion requires an explicit update here before
experiments depend on the change.

## Decision log

The initial entries below reconstruct the accepted choices from the project
planning discussion. They share a recorded date because more precise historical
timestamps were not captured; no earlier chronology is implied by their IDs.

### PD-001 — Limit the project to GPU acceleration and optimization

- **Recorded:** 2026-07-11
- **Status:** Accepted
- **Decision:** Cover only the GPU Acceleration and Optimization portion of
  NCP-GENL through reproducible training experiments.
- **Rationale:** A narrow curriculum permits deeper measurement and explanation
  of distributed-training behavior instead of shallow coverage of the entire
  certification.
- **Consequences:** RAG, serving, safety, general application design, and broad
  model-quality evaluation are outside this repository.

### PD-002 — Use a controlled single-node A100 environment

- **Recorded:** 2026-07-11
- **Status:** Accepted
- **Decision:** Use Runpod A100 SXM Pods on one physical server, with one to four
  GPUs and no complete 5D-parallel run.
- **Rationale:** A single host provides controlled topology and meaningful
  NVLink/NVSwitch and NCCL measurements without depending on an unknown
  inter-node fabric. Four GPUs cover the selected single-dimension and hybrid
  layouts at acceptable rental cost.
- **Consequences:** Multi-node, two-Pod, eight-GPU, inter-datacenter, and full-5D
  experiments are excluded.

### PD-003 — Standardize on PyTorch and NeMo/Megatron

- **Recorded:** 2026-07-11
- **Status:** Accepted
- **Decision:** Implement experiments only with native PyTorch or NVIDIA NeMo
  Framework with Megatron Core.
- **Rationale:** PyTorch exposes low-level distributed behavior for learning and
  debugging, while NeMo/Megatron supplies NVIDIA's optimized model-parallel
  implementation. Together they cover the curriculum without a broad framework
  comparison.
- **Consequences:** Maintain two independent container image families. Use the
  Ultra-Scale Playbook for hypotheses, but do not adopt Nanotron or introduce
  DeepSpeed, Accelerate, Colossal-AI, JAX, or TensorFlow as training frameworks.

### PD-004 — Build reproducible containers before renting GPUs

- **Recorded:** 2026-07-11
- **Status:** Accepted
- **Decision:** Derive pinned images from NVIDIA NGC PyTorch and NeMo containers,
  publish project images to GHCR, and perform development and preparation
  locally whenever possible.
- **Rationale:** Prebuilt images reduce billable setup time and make software
  versions and results reproducible across Pod sessions.
- **Consequences:** Recorded runs may not use floating `latest` tags. Container
  references and digests are part of every result record.

### PD-005 — Use short full-parameter continued-pretraining runs

- **Recorded:** 2026-07-11
- **Status:** Accepted
- **Decision:** Training experiments use full-parameter continued pretraining
  with autoregressive next-token cross-entropy. Evaluation is limited to
  correctness and short loss/perplexity sanity checks.
- **Rationale:** This exercises forward, backward, optimizer, communication,
  sharding, and checkpoint paths without turning the infrastructure curriculum
  into a model-quality or convergence project.
- **Consequences:** Resulting weights are not the project product. SFT, LoRA,
  RLHF, downstream benchmark suites, and claims of model improvement are out of
  scope. The exact model and dataset remain proposals until separately accepted.

### PD-006 — Keep advanced out-of-scope topics theoretical

- **Recorded:** 2026-07-11
- **Status:** Accepted
- **Decision:** Do not add experiments for native FP8 on A100, MoE/expert
  parallelism, CPU offload that requires another model, MIG, cluster schedulers,
  production serving, or paper-scale convergence.
- **Rationale:** These topics require incompatible hardware, additional model or
  framework complexity, or work outside the selected certification domain.
- **Consequences:** Relevant concepts may be documented for exam preparation,
  but they must not be presented as measurements performed by this project.

### PD-007 — Separate prerequisites and shared measurements from experiments

- **Recorded:** 2026-07-11
- **Status:** Accepted
- **Decision:** Treat environment qualification as a mandatory pre-run check and
  memory/OOM anatomy as a shared measurement responsibility, not numbered
  experiments. Maintain experiment definitions and selection status in the
  separate catalog.
- **Rationale:** A numbered experiment should test a falsifiable systems
  hypothesis. Qualification and memory accounting support many experiments but
  do not independently satisfy that standard.
- **Consequences:** Every Pod session runs the qualification script, applicable
  experiments report memory anatomy, and `EXPERIMENT_CATALOG.md` remains the
  authoritative experiment worksheet.

## Primary references

- [NVIDIA NCP-GENL certification and exam blueprint](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
- [NVIDIA NeMo Framework software component versions](https://docs.nvidia.com/nemo-framework/user-guide/latest/softwarecomponentversions.html)
- [NVIDIA Megatron Core installation guidance](https://docs.nvidia.com/megatron-core/developer-guide/nightly/get-started/install.html)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
