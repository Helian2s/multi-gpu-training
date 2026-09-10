# Project decisions

Status: Accepted
Last updated: 2026-07-15

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

The [portfolio overview](README.md) describes the work implemented and run.
The [2026-09-10 validation review](docs/validation-status.md) records gaps
between these accepted decisions and the current implementation, including the
synthetic parallelism workloads. Those findings do not amend the accepted
contract or the decision history below.

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

1. **Supported cloud providers:** AWS EC2 and Runpod. AWS remains the
   PyTorch/Blackwell/PCIe provider for G7e work, and Runpod supplies the A100
   SXM/NVLink environment for the Runpod communication baseline and
   topology-sensitive NeMo/Megatron experiments. EXP-12 is an approved
   AWS-A2-Megatron exception because pipeline scheduling does not hard-require
   NVLink and the AWS PCIe environment is sufficient for the bubble/schedule
   mechanics being tested. AWS capacity failures do not cancel the AWS
   queues; Runpod preparation may proceed in parallel while AWS launch
   configurations, ECR images, and S3 paths remain available for later retry.
   AWS cache volumes can be recreated when another AWS launch is approved. No
   measured run begins until the selected resource passes
   capacity, permission, price, and runtime qualification. VT1 video-transcoding
   accelerators are part of the AWS quota's name but are not CUDA GPUs and are
   not used by this project.
2. **GPU policy:** the project is not restricted to A100. It uses NVIDIA GPUs
   supported by the pinned CUDA, PyTorch, NeMo/Megatron, Transformer Engine, and
   profiler stack. Every result records provider, location, resource type,
   exact GPU architecture/name/memory, driver, and topology. Different GPU
   models or memory sizes are separate environments and are not pooled as
   identical runs.
3. **AWS compute profiles:** `g7e` with RTX PRO 6000 Blackwell Server Edition is
   the planned primary family. `AWS-A1` is `g7e.2xlarge` with one GPU,
   `AWS-A2` is `g7e.12xlarge` with two GPUs, and `AWS-A4` is
   `g7e.24xlarge` with four GPUs. Each GPU has 96 GB; the four-GPU size consumes
   the complete 96-vCPU quota and is not in the current AWS queue. The current
   AWS-A2 queue may run one-visible-GPU and two-visible-GPU phases on one
   `AWS-A2` host; reports must record both the billed physical profile and
   visible GPU count. `g6e` is a contingency that requires an explicit mapping
   change rather than a silent substitution because its GPU, memory, CPU
   allocation, and topology differ.
4. **One physical server only:** every experiment runs on one EC2 instance or
   one Runpod Pod. We will not configure multi-node training or join independent
   hosts.
5. **Active GPU count:** every experiment exposes one to four GPUs, but four
   GPUs are admitted only for the `TP=2 x DP=2` hybrid in the current plan. AWS
   uses exact G7e profiles, with current EXP-02, EXP-07, EXP-08, and EXP-09
   AWS-A2 work batched on one physical `AWS-A2` host by changing visible GPU
   count.
   AWS-A2 is also admitted for the EXP-12 one-/two-visible-GPU Megatron
   pipeline-schedule run. Runpod uses an exact two-GPU A100 SXM Pod for
   NVLink/PyTorch readiness and the remaining one-/two-rank NeMo/Megatron
   work, and an exact four-GPU Pod for the hybrid. Reports record both visible
   GPUs and the complete billed resource. No experiment may use more than four
   GPUs.
6. **No full 5D experiment:** the project will study parallelism dimensions
   separately and in useful smaller combinations, but will not run a complete
   five-dimensional parallelism layout.
7. **Container registries:** private Amazon ECR in `us-west-2` is the primary
   AWS runtime registry. GHCR is a content-identical mirror used by Runpod.
   Images are built once, published to both registries, and recorded by each
   registry's immutable digest. A short-lived ECR authorization token is not
   stored as a persistent Runpod registry credential.
8. **Local-first workflow:** code, configuration, documentation, container build
   definitions, and most data preparation happen in this repository. A billable
   GPU host should run only qualification, staging that requires the target
   environment, profiling, or experiment work.
9. **Runpod profiles and framework placement:** `RUNPOD-A100-SXM2` is one Secure
   Cloud Pod with two `NVIDIA A100-SXM4-80GB` GPUs for the Runpod NVLink/NCCL
   communication baseline and the one-/two-rank NeMo/Megatron experiments whose
   communication pattern benefits materially from NVLink, including TP/SP and
   CP. The current two-GPU Runpod execution queue is `RUNPOD-A2-Megatron`,
   using the NeMo/Megatron image for `QUAL-RUNPOD-A2-Megatron`, EXP-10,
   EXP-11, and EXP-13 so one Pod can run the complete two-GPU Runpod phase.
   EXP-10 remains a model-free CUDA/NCCL communication experiment; it runs in
   the NeMo/Megatron image only to avoid changing images inside a single Pod
   queue. `RUNPOD-A100-SXM4` is a separate four-GPU Pod used only by
   `RUNPOD-A4-Megatron` for EXP-14. `AWS-A2-Megatron` is the approved AWS queue
   for EXP-12. The Runpod `A` number is a queue-level GPU count label, while the
   exact billed Pod resource and visible-device mask are recorded separately in
   every run artifact. `RUNPOD-A2-Megatron` deliberately contains both
   one-visible-GPU and two-visible-GPU phases for its assigned experiments. AWS
   remains the PyTorch/Blackwell/PCIe environment and now also hosts the EXP-12
   Megatron pipeline-schedule exception.
   Qualification must show NVLink between every selected pair; NVSwitch is
   recorded only if observed topology proves it.
10. **One provider per experiment:** an experiment is assigned to AWS or
    Runpod, never both. GPU-count scale points may use sequential instance sizes
    from the same provider when GPU count is the tested variable, but every run
    remains on one physical host. Cross-provider observations live in separate
    numbered experiments.

## Accepted framework decisions

The only training frameworks used to implement experiments are:

- PyTorch, for transparent low-level experiments and direct use of
  `torch.distributed`.
- NVIDIA NeMo Framework with Megatron Core, for NVIDIA's optimized LLM training
  stack and higher-level model-parallel experiments.

Numbered NeMo/Megatron experiments normally run on the accepted Runpod A100 SXM
profiles. EXP-12 is the approved exception and runs on AWS-A2 because pipeline
schedule, bubble, and stage-balance behavior can be tested without NVLink. This
is an operational and topology choice, not permission to compare NeMo/Megatron
performance directly with PyTorch experiments on AWS Blackwell hardware.

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
stack after a billable GPU host starts:

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

AWS-A1-PyTorch and AWS-A2-PyTorch work use the same AWS PyTorch runtime image
when the software stack is identical. The EC2 profile, visible GPU count,
qualification smoke, and measured workload are selected by provider
configuration and run-unit arguments, not by baking separate A1/A2 images.
The current Runpod phase uses the GHCR mirror of the NeMo/Megatron image family
for EXP-10, EXP-11, EXP-13, and EXP-14. The Runpod communication baseline uses
CUDA/NCCL tools inside that image rather than a separate PyTorch Pod image, so
the two-GPU Pod can run one sequential queue. Create a new image only when
source, dependencies, profiler tooling, or runtime contracts change.

Images are built once locally with Docker Buildx or by GitHub Actions and pushed
to private Amazon ECR in `us-west-2` and to GHCR in the same build workflow.
AWS pulls from ECR through a least-privilege instance role; Runpod pulls the
GHCR mirror. Base images and Python dependencies must be pinned to immutable
versions (and preferably image digests); the `latest` tag is forbidden for
recorded experiments. The build record preserves both registry references and
digests so mirrored content can be verified.

The local workstation may build `linux/amd64` images, but GPU execution and
validation occur on the selected provider. We do not expect an NVIDIA GPU or
CUDA driver on the local workstation.

## Provider portability and storage

Experiment logic must not call AWS or Runpod APIs. Provider adapters are limited
to provisioning, connection, storage attachment/staging, instance metadata,
credential injection, and stop/termination safeguards. Inside the container,
the image digest, configuration format, launch command, qualification checks,
metrics, and artifact layout are provider-neutral.

Durable and performance storage have separate roles:

- On AWS, durable source and result artifacts use S3; EBS and EC2 instance-store
  NVMe may stage inputs and outputs for a run. Instance-store data
  is treated as ephemeral.
- On Runpod, a network volume may hold durable prepared inputs and results; Pod
  volume/container storage may stage performance-sensitive files.
- Every run resolves logical input and output locations to provider-specific
  URIs or mount paths and copies its resolved configuration into the run
  artifact directory.

Provider is environmental metadata, not an experimental variable inside a
numbered experiment. Each experiment uses exactly one provider. When an AWS and
a Runpod environment answer related questions, they receive separate IDs and
may be compared only with GPU architecture, memory, topology, image content,
workload, and measurement method stated explicitly.

Before AWS is admitted for measured experiments, qualification must confirm the
96-vCPU On-Demand G/VT quota in `us-west-2`, all planned G7e type offerings and
Availability Zone capacity, physical topology and GPUDirect peer access, any
visibility mask, container execution, profiler/DCGM permissions, storage
staging, artifact upload, current price, and automatic termination.

## Experiment scale and coverage

The proposed experiment inventory and its lifecycle status are maintained in
[EXPERIMENT_CATALOG.md](EXPERIMENT_CATALOG.md). An experiment becomes committed
work only after its `Status` entry is changed from `proposed` to `accepted`.
Project scope and exclusions remain authoritative in this file regardless of
catalog status.

Current canonical experiment IDs use the two-digit form `EXP-NN` and define the
recommended provider-blocked order: EXP-01 through EXP-09 are the AWS PyTorch
phase; EXP-12 is the AWS-A2-Megatron exception; EXP-10, EXP-11, EXP-13, and
EXP-14 are the Runpod phase. The active execution queues are
`AWS-A1-PyTorch`, `AWS-A2-PyTorch`, `AWS-A2-Megatron`,
`RUNPOD-A2-Megatron`, and `RUNPOD-A4-Megatron`; AWS queues stay eligible for retry while Runpod
preparation proceeds. Three-digit
experiment IDs appearing in the decision log are historical identifiers and are
not current catalog IDs. Each provider phase begins with its model-free
topology/P2P/NCCL experiment: EXP-01 on AWS and EXP-10 on Runpod.

AWS launch planning may use run-unit IDs `QUAL-A1`, `QUAL-A2`,
`EXP-NN-A1`, and `EXP-NN-A2V1/A2V2` to queue concrete compute-profile phases.
These are not canonical experiment IDs, do not create experiment directories,
and do not carry separate lifecycle status from the parent catalog row. `V1`
and `V2` denote the number of visible GPUs on the billed `AWS-A2` host.
Sub-runs that are not required to answer the parent hypothesis are omitted from
the current queue instead of being kept as standby experiment work. `QUAL-A4`
or `EXP-NN-A4` require a new project decision before AWS four-GPU launch work
resumes.

Runpod execution planning uses framework-qualified queue labels
`RUNPOD-A2-Megatron` and `RUNPOD-A4-Megatron` plus exact Pod resource profiles. These labels organize
execution and artifacts; they are not canonical experiment IDs and do not
replace the exact GPU type, billed GPU count, visible GPU count, datacenter,
Pod ID, topology, image family, or image digest in run records.

AWS NeMo/Megatron execution planning uses the framework-qualified
`AWS-A2-Megatron` queue for EXP-12 only. It reuses the AWS-A2 G7e host profile,
ECR, S3, and lifecycle guards while recording that the result is a PCIe
pipeline-schedule measurement, not a Runpod NVLink result.

For experiments that train a model, the accepted training mode is
full-parameter continued pretraining with autoregressive next-token
cross-entropy. The runs exercise and validate the complete training path; they
are not intended to train a model to convergence or produce a final model
artifact.

Comparable variants restore the same pinned pretrained checkpoint, input
batches, and seeds. Model, tokenizer, dataset, and preprocessing revisions plus
deterministic sample hashes must be recorded before results are treated as
comparable. The accepted model and tokenizer are
`Qwen/Qwen3-1.7B-Base` at revision
`ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. The accepted dataset is
`Salesforce/wikitext` `wikitext-103-raw-v1` at revision
`b08601e04326c79dfdd32d625aee71d232d685c3`, processed under the
`qwen3-wikitext-v1` contract in `configs/inputs.lock.yaml`. Default precision
and the evaluation boundary remain proposals in the catalog until separately
accepted.

The normal progression is:

| Scale | Primary purpose |
| --- | --- |
| 1 GPU | Correctness, timing baseline, memory baseline, mixed precision, kernels, profiling |
| 2 GPUs | DDP/FSDP, collectives, tensor or pipeline parallelism, communication effects |
| 4 GPUs | Only the TP=2 x DP=2 hybrid in the current plan |

Allowed topics include:

- CPU versus GPU and single-GPU baselines where they clarify acceleration.
- FP32, TF32, BF16, and FP16 behavior on the selected GPU. Standard FP8 through
  Transformer Engine is planned for G7e only after the pinned image proves a
  native path on the RTX PRO 6000 Blackwell Server Edition (SM 12.0). The
  current Transformer Engine documentation limits MXFP8 and NVFP4 training to
  SM 10.0/10.3, so neither format is part of the G7e experiment plan. Hardware
  advertising FP4 operations is not sufficient evidence of a supported
  distributed-training recipe.
- Tensor Core utilization and shape/alignment effects.
- Data parallelism with PyTorch DDP.
- Parameter, gradient, and optimizer-state sharding with PyTorch FSDP.
- Tensor, pipeline, sequence, and context parallelism where supported by
  NeMo/Megatron.
- The selected TP=2 x DP=2 hybrid layout; other four-GPU hybrids are excluded
  unless a distinct hypothesis justifies reopening the decision.
- Gradient accumulation, microbatch size, global batch size, activation
  checkpointing, and optimizer/memory trade-offs.
- NCCL collective behavior within one server over PCIe, NVLink, and/or NVSwitch
  as exposed by the allocated NVIDIA GPU host.
- Data-loading and CPU bottlenecks when they affect GPU utilization.
- Kernel, CUDA, NCCL, CPU, memory, and end-to-end timeline profiling.
- Failure diagnosis, out-of-memory behavior, and reproducibility.

## Accepted exclusions

The following are excluded from implementation in this project. They may still
be discussed theoretically when useful for NCP-GENL preparation.

- Multi-node and inter-datacenter training.
- Runpod Instant Clusters for this project.
- Two-host distributed training and inter-node versus intra-node benchmarks.
- A complete 5D parallelism layout.
- Any experiment requiring more than four GPUs.
- Kubernetes, Slurm, or another cluster scheduler.
- Production serving, orchestration, RAG, and application monitoring.
- Any low-precision result relabeled as native hardware acceleration when the
  selected GPU or pinned software stack actually emulates or falls back from it.
- A CPU-offload experiment that would require introducing a second, larger
  dense model solely to exceed the selected GPU's memory.
- Mixture-of-Experts and expert-parallel training experiments.
- Nanotron, DeepSpeed, Accelerate, Colossal-AI, JAX, TensorFlow, or another
  distributed-framework comparison.
- Full pretraining, training to convergence, or reproduction at paper-scale
  model and cluster sizes.
- MIG partitioning and other resource multi-tenancy experiments.

## Approved tools and libraries

This is the final baseline toolset. A tool being listed does not mean every
experiment must use it. Once an NGC base release is selected, compatible
versions will be pinned together in the Dockerfiles and dependency lock files;
they will not be duplicated here. Installation and authentication status is
recorded separately in
[infra/TOOLING.md](infra/TOOLING.md).

### Local development, automation, and storage

| Tool | Role |
| --- | --- |
| Git and GitHub | Version control, source hosting, and review |
| GitHub Actions | Optional reproducible `linux/amd64` image builds and checks |
| Amazon Elastic Container Registry (ECR) | Primary private runtime registry for AWS in `us-west-2` |
| GitHub Container Registry (GHCR) | Content-identical private mirror pulled by Runpod |
| Docker with BuildKit/Buildx | Build and inspect OCI container images |
| AWS CLI | Provision, inspect, connect to, and terminate the approved EC2 GPU host and transfer artifacts |
| AWS EC2 and IAM roles | AWS compute and scoped machine identity for S3 and management access |
| Amazon S3 | Durable AWS-side model, dataset, result, and profiler artifact storage |
| Amazon EBS and EC2 instance-store NVMe | Persistent or ephemeral AWS staging storage as selected by the run |
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
| NVIDIA GPU driver | Host-side NVIDIA GPU access supplied or installed according to the provider image contract |
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
| NVIDIA DCGM / `dcgmi` | GPU health, diagnostics, and longer-running telemetry when available on the compute host |
| NVIDIA Nsight Systems (`nsys`) | End-to-end CPU, CUDA, kernel, NCCL, and NVTX timeline analysis |
| NVIDIA Nsight Compute (`ncu`) | Detailed kernel metrics and roofline-style kernel analysis |
| PyTorch Profiler | Operator, CPU, CUDA, memory, shape, and distributed traces |
| TensorBoard | Viewing PyTorch profiler traces and experiment scalar series |
| `nccl-tests` | Controlled latency, bandwidth, and collective scaling measurements |
| CUDA Samples (`bandwidthTest`, `p2pBandwidthLatencyTest`) | Host/device and GPU peer-path validation where available |
| `torch.utils.benchmark` | Repeatable Python-level microbenchmarks |
| `psutil` | CPU, process, and host-memory telemetry collected by experiment scripts |

Some low-level counters and diagnostics depend on permissions exposed by the
selected provider and host image. If DCGM, Nsight Compute hardware counters, or
another approved diagnostic is unavailable inside the container, the
experiment must record the limitation rather than silently omit the intended
measurement.

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
| Hugging Face Hub and Datasets | Downloading the pinned public model, tokenizer, and WikiText artifacts |
| Transformers, Tokenizers, and Safetensors | Native PyTorch Qwen3 definition, tokenizer execution, and checkpoint format |
| PyArrow | Deterministic reading of the pinned WikiText Parquet shards |

Additional libraries must be justified by a concrete experiment, pinned, and
added to this decision record before becoming shared project infrastructure.
Large hosted experiment platforms are intentionally unnecessary: raw metrics,
profiler artifacts, analysis outputs, and reports remain reproducible from the
repository and project storage.

## Reproducibility requirements

Every recorded run must preserve at least:

- Git commit and whether the worktree was dirty.
- Container image reference and digest.
- Provider, Region/datacenter, instance or Pod type, purchase option, host ID,
  physical GPU count, visible GPU count, and exact GPU selection.
- GPU name and memory, NVIDIA driver, CUDA, cuDNN, NCCL, PyTorch, NeMo,
  Megatron Core, and Transformer Engine versions as applicable.
- `nvidia-smi topo -m` output and relevant NCCL environment variables.
- Experiment configuration, random seeds, model shape, sequence length,
  precision, microbatch/global batch sizes, and number of warm-up/measured
  iterations.
- Raw timing, throughput, memory, utilization, profiler, and correctness data.
- Estimated provider duration and cost, including the complete billed resource
  whenever only a subset of its GPUs is visible.
- Analysis code, expected result, observed result, and conclusion.

Results must be copied to durable storage before a Pod is stopped or an EC2
instance is stopped/terminated. Generated checkpoints and large profiler traces
are not committed to Git; small metrics, configuration, analysis code, and the
final report are.

## Decision-change rule

This file records project-wide constraints. Changing provider, GPU family,
maximum scale, framework set, registry strategy, certification scope, shared training
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

### PD-008 — Support AWS and Runpod through provider adapters

- **Recorded:** 2026-07-12
- **Status:** Accepted
- **Supersedes:** The Runpod-only provider choice and physical-allocation
  assumption in PD-002. It preserves PD-002's single-node rule, maximum of four
  active experiment GPUs, and exclusion of a full 5D run.
- **Decision:** Support AWS EC2 and Runpod, qualify AWS first, and keep provider
  operations outside experiment logic. An AWS P4 host may contain eight physical
  A100s, but each experiment exposes only one to four of them.
- **Rationale:** The approved 96-vCPU AWS P-instance quota creates a viable
  initial path, while provider-neutral containers, configurations, artifacts,
  and launch behavior reduce the risk of becoming unnecessarily locked to AWS.
- **Consequences:** AWS qualification must distinguish `p4d.24xlarge` 40 GB
  A100s from `p4de.24xlarge` 80 GB A100s and record the full-instance cost.
  Provider-specific provisioning and storage adapters are allowed; experiment
  implementations and measured contracts remain shared. Cross-provider results
  are not treated as interchangeable without matching hardware and topology.

### PD-009 — Use the AWS G quota and make hardware capability-driven

- **Recorded:** 2026-07-12
- **Status:** Accepted
- **Supersedes:** PD-008's assumption that the AWS approval covered P instances,
  PD-002's A100-only hardware choice, and PD-006's practical exclusion of
  measured FP8. It preserves provider portability, single-node execution, and
  the one-to-four active-GPU limit.
- **Decision:** Interpret the approved quota as 96 vCPUs of Running On-Demand G
  and VT instances in `us-west-2`. Evaluate G7e/Blackwell first and G6e/Ada as
  the fallback. Select experiments by recorded hardware capability rather than
  assuming A100-specific precision or interconnect behavior.
- **Rationale:** G7e supplies one-, two-, and four-GPU sizes with 96 GB per GPU,
  GPUDirect P2P over PCIe, Blackwell precision support, and a four-GPU size that
  exactly fits the quota. G6e supplies up to four 48 GB L40S GPUs within the
  quota and may be more economical. Both preserve the NVIDIA software focus.
- **Consequences:** Native FP8 becomes measurable on supported Ada/Blackwell
  configurations, while Blackwell adds MXFP8/NVFP4 opportunities. PCIe P2P is
  not treated as NVLink/NVSwitch; topology-sensitive expectations must follow
  qualification evidence. G5/A10G and G6/L4 remain possible fallback hardware,
  but their 24 GB per-GPU memory may invalidate the planned one-GPU full-training
  baseline and therefore requires an explicit workload review before use.

### PD-010 — Assign exact provider profiles to the experiment catalog

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** PD-009's treatment of G7e and G6e as equally unresolved
  experiment candidates, and the general Runpod allocation implied by PD-002.
  It preserves support for both providers, the single-node rule, and the
  one-to-four visible-GPU limit.
- **Decision:** Use `AWS-G7E-1` (`g7e.2xlarge`), `AWS-G7E-2`
  (`g7e.12xlarge`), and `AWS-G7E-4` (`g7e.24xlarge`) as the primary compute
  profiles in `us-west-2`. Use one `RUNPOD-A100-SXM4` Secure Cloud Pod with four
  `NVIDIA A100-SXM4-80GB` GPUs for EXP-007, EXP-008, and selected EXP-009 and
  EXP-011 comparison points. The Runpod host must expose a qualified NVLink
  topology; NVSwitch is claimed only when observed.
- **Rationale:** G7e supplies exact one-, two-, and four-GPU allocations under
  the approved quota and is suitable for the common workload. A single
  controlled Runpod A100 SXM session supplies the scale-up interconnect that is
  absent from G7e and lets topology, NCCL, DDP, and TP observations be connected
  without turning the project into a general provider benchmark.
- **Consequences:** The catalog records a compute profile for every experiment.
  The Runpod comparison Pod is billed as four GPUs even when visibility is
  restricted to one or two. Results from RTX PRO 6000 and A100 are separate
  hardware environments; no cross-provider result is used as an interchangeable
  baseline. G6e or another resource requires an explicit decision and catalog
  update before measured use.

### PD-011 — Limit G7e low-precision training to qualified standard FP8

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** PD-009's statement that G7e creates an immediate measured
  MXFP8/NVFP4 opportunity. It preserves PD-009's capability-driven precision
  policy and authorization to measure native standard FP8.
- **Decision:** Qualify Transformer Engine standard FP8 Current or Delayed
  Scaling on G7e's SM 12.0 RTX PRO 6000 GPU. Do not include MXFP8 or NVFP4
  training in the current experiment plan because the selected Transformer
  Engine documentation lists their training support for SM 10.0/10.3 rather
  than SM 12.0.
- **Rationale:** A Blackwell product name and advertised FP4 Tensor Core
  operations do not prove that a specific distributed-training recipe is
  implemented for every Blackwell compute capability.
- **Consequences:** EXP-001 compares BF16 and other conventional modes with
  standard FP8 only when runtime evidence proves the native recipe and kernels.
  Unsupported, emulated, or fallback paths are recorded but are not performance
  variants. MXFP8/NVFP4 can be reconsidered only after the pinned stack documents
  and passes SM 12.0 training qualification.

### PD-012 — Use ECR for AWS and mirror the same images to GHCR

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** PD-004's GHCR-only publication decision. It preserves the
  build-once, immutable-image, and local-first requirements.
- **Decision:** Publish each PyTorch and NeMo/Megatron image to private Amazon
  ECR in `us-west-2` and to GHCR in one build workflow. AWS pulls from ECR;
  Runpod pulls the GHCR mirror. GitHub Actions should use OIDC to assume a
  least-privilege AWS publishing role and its repository token for GHCR.
- **Rationale:** ECR integrates naturally with AWS instance roles and same-Region
  EC2 pulls. GHCR remains convenient for Runpod, while an ECR Docker
  authorization token is short-lived and is a poor persistent Pod credential.
- **Consequences:** Images are not rebuilt per provider. Every build records
  both immutable references and verifies the mirrored content. EC2 gets ECR
  pull-only permission; Runpod gets GHCR pull-only credentials. Mutable
  `latest` tags and long-lived AWS keys in GitHub or Runpod remain forbidden.

### PD-013 — Give every experiment one provider and restrict four-GPU runs

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** PD-010's cross-provider EXP-007/008/009/011 matrices and its
  plan to rent one four-GPU Runpod Pod for masked one- and two-GPU work.
- **Decision:** Assign each numbered experiment to exactly one provider. Split
  PCIe and NVLink topology/NCCL work into separate AWS and Runpod experiment
  IDs. Use `RUNPOD-A100-SXM2` for two-GPU NVLink, TP, and CP work and
  `RUNPOD-A100-SXM4` only for TP=2 x DP=2. Retain four GPUs elsewhere only for
  AWS DDP's 1/2/4 scaling curve. Remove the optional TP=2 x CP=2 experiment.
- **Rationale:** A provider change also changes GPU architecture and host
  environment, so hiding it inside one experiment makes the hypothesis
  ambiguous. Two GPUs answer topology, collective, FSDP, TP, PP, CP,
  checkpoint, and troubleshooting questions. Four ranks add unique evidence
  only for a second DDP scaling step and for two non-trivial 2 x 2 process-group
  dimensions.
- **Consequences:** The catalog is renumbered continuously through EXP-019.
  Sequential scale sub-runs may use different instance sizes from the same
  provider, but every run uses one host. Adding a four-GPU point requires an
  explicit hypothesis and decision update.

### PD-014 — Keep the G/VT quota; do not require AWS P quota

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Decision:** Treat the approved 96-vCPU On-Demand G/VT quota in `us-west-2`
  as useful and aligned with the current curriculum. Do not request P quota for
  the accepted plan. Reconsider P only if a future experiment specifically
  requires AWS-hosted NVSwitch or another P-family capability.
- **Rationale:** G7e supplies exact one-, two-, and four-GPU sizes and the
  Blackwell/PCIe environment used by most experiments. AWS P-family quota is a
  separate quota, but the relevant NVSwitch training instances commonly expose
  eight GPUs, which conflicts with the project's one-to-four-GPU cost model and
  would bill unused accelerators. Runpod supplies exact two- and four-GPU A100
  SXM environments for the accepted NVLink work.
- **Consequences:** The G/VT request was not a mistake; it simply does not grant
  P-family access. No result may describe G7e PCIe P2P as NVLink/NVSwitch. A
  future P request must include a new cost, physical-GPU-count, and experiment
  justification before it changes the catalog.

### PD-015 — Reduce the exam-focused core to 14 experiments

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** PD-013's continuously numbered 19-experiment catalog. It
  preserves one provider per experiment and the two admitted four-GPU cases.
- **Decision:** Maintain 14 numbered core experiments. Merge topology/P2P and
  NCCL into one communication experiment per provider; fold input-pipeline
  starvation into controlled troubleshooting; make `torch.compile` a secondary
  variant of the attention/fusion experiment; move distributed checkpointing
  to a non-numbered optional operational extension; and make the capstone a
  report and exam decision worksheet with no dedicated GPU run.
- **Rationale:** The NCP-GENL GPU Acceleration and Optimization domain rewards
  understanding multi-GPU setup, parallelism, memory/batch optimization,
  profiling, and troubleshooting. Separate topology and NCCL experiments on the
  same host repeated one causal communication lesson, while input starvation
  repeated a troubleshooting mechanism. Checkpoint recovery is useful but maps
  more directly to reliability, and evidence synthesis is a deliverable rather
  than a falsifiable experiment.
- **Consequences:** The core is renumbered continuously from EXP-001 through
  EXP-014 with an estimated 19.5-38 measured GPU-hours. The four-GPU cases are
  now EXP-008 DDP scaling and EXP-013 TP=2 x DP=2. Individual experiment
  `Decision` values remain `TBD` until execution scope and budget are approved.
  The final report explicitly records that inference optimization is a
  theory-only coverage gap in this distributed-training project.

### PD-016 — Consolidate all NeMo/Megatron experiments on Runpod

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** The catalog placement of EXP-011 pipeline parallelism on
  `AWS-G7E-2`. It preserves the one-provider-per-experiment rule, the two
  admitted four-GPU cases, and the ECR/GHCR image strategy.
- **Decision:** Run every numbered NeMo/Megatron experiment on Runpod A100 SXM.
  Use `RUNPOD-A100-SXM2` for EXP-010 tensor/sequence parallelism, EXP-011
  pipeline parallelism, and EXP-012 context parallelism. Keep
  `RUNPOD-A100-SXM4` exclusively for EXP-013 TP=2 x DP=2. If the optional
  distributed-checkpoint extension is selected, run its PyTorch and
  NeMo/Megatron jobs sequentially on `RUNPOD-A100-SXM2`.
- **Rationale:** The same pinned NeMo image, converted model, dataset, and A100
  SXM/NVLink host can be staged once and reused across all one-/two-rank model-
  parallel experiments. Pipeline scheduling and bubble behavior remain visible
  on NVLink, while keeping it with TP and CP removes an unnecessary second
  NeMo provider environment. The project does not use these runs for a direct
  PyTorch-versus-Megatron performance comparison.
- **Consequences:** The Runpod R2 session now groups EXP-007 and EXP-010 through
  EXP-012; the R4 session remains EXP-013 only. AWS sessions contain PyTorch and
  NVIDIA-tool experiments but no NeMo/Megatron training. Framework and provider
  are correlated in the measured catalog, so reports must not attribute an
  AWS/Runpod or Blackwell/A100 difference solely to the framework. Both image
  families remain mirrored to ECR and GHCR under the accepted registry policy.

### PD-017 — Make canonical experiment IDs reflect provider execution order

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** The three-digit canonical numbering and mixed-provider order
  established by PD-015 and subsequently updated by PD-016. Historical decision
  entries retain their original identifiers.
- **Decision:** Use two-digit canonical IDs `EXP-01` through `EXP-14`. Order the
  current catalog with all AWS work first and all Runpod work second. The exact
  mapping from the immediately preceding catalog is:
  - EXP-001 through EXP-006 become EXP-01 through EXP-06 without changing topic.
  - EXP-008 DDP becomes EXP-07; EXP-009 FSDP becomes EXP-08; and EXP-014
    troubleshooting becomes EXP-09.
  - EXP-007 Runpod NVLink/NCCL becomes EXP-10.
  - EXP-010 TP/SP becomes EXP-11; EXP-011 PP becomes EXP-12; EXP-012 CP becomes
    EXP-13; and EXP-013 TP=2 x DP=2 becomes EXP-14.
- **Rationale:** A provider-blocked curriculum avoids switching between AWS and
  Runpod and lets the Runpod communication baseline immediately precede all
  NeMo/Megatron work. Changing the identifier width distinguishes current
  references from superseded three-digit references preserved in this
  append-only history.
- **Consequences:** Current documents, templates, automation, artifact examples,
  and provider session plans use `EXP-NN`. IDs define the recommended high-level
  execution order, while one-/two-/four-GPU sub-runs may still be batched by
  compute profile for cost efficiency. Project-decision IDs retain their
  independent `PD-NNN` format.

### PD-018 — Start the AWS phase with its communication baseline

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** PD-017's ordering of the first six two-digit AWS experiment
  IDs. It preserves the AWS-then-Runpod provider phases and leaves EXP-07 through
  EXP-14 unchanged.
- **Decision:** Move the AWS PCIe P2P/NCCL experiment from EXP-06 to EXP-01.
  Shift the former EXP-01 through EXP-05 to EXP-02 through EXP-06 respectively:
  mixed precision, batch geometry, activation checkpointing, attention/fusion,
  and profiler triangulation. Keep DDP, FSDP, troubleshooting, and every Runpod
  experiment at EXP-07 through EXP-14.
- **Rationale:** Both provider phases should begin with a model-free
  communication experiment that validates and characterizes the fabric before
  model-dependent distributed work. Profiler triangulation remains after its
  attention/fusion input case and before the more complex DDP/FSDP experiments;
  strict grouping by stack label would weaken that dependency order.
- **Consequences:** The current catalog order is AWS communication, AWS execution
  fundamentals/profiling, AWS data parallelism/troubleshooting, Runpod
  communication, and Runpod NeMo/Megatron. Historical references in PD-017
  retain the immediately preceding mapping; active documents use the PD-018
  order.

### PD-019 — Simplify experiment selection to one lifecycle status

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** The per-row `Recommendation` and `Decision` fields retained by
  PD-015 and referenced by later catalog-order decisions. It does not accept the
  proposed experiments or shared workload choices.
- **Decision:** Remove the redundant `Recommendation` column from the numbered
  catalog and replace `Decision` with `Status`. Use `proposed`, `accepted`,
  `deferred`, and `completed` as the supported experiment lifecycle values. Keep
  every numbered experiment at `proposed` until it is explicitly accepted.
- **Rationale:** Every numbered row is already part of the recommended core, so
  repeating `Core` conveys no information. A lifecycle status expresses both
  selection and later progress more clearly than a one-time decision field.
- **Consequences:** Experiment scaffolding requires `Status=accepted`; proposed
  or deferred rows cannot create implementation directories, and `completed`
  denotes a finished report rather than initial authorization. Historical
  `Decision`/`TBD` wording remains unchanged in prior log entries.

### PD-020 — Clarify the shared workload contract and its unresolved choices

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** The ten-row shared-workload worksheet that mixed independent
  choices with redundant model, tokenizer, data-preparation, and evaluation
  rows. It preserves PD-005's accepted training task and its explicit deferral of
  the exact model and dataset.
- **Decision:** Consolidate the workload worksheet into eight non-overlapping
  areas. Accept the methodology that comparable variants restore the same
  pinned checkpoint, batches, and seeds; that full-parameter continued
  pretraining uses next-token cross-entropy for short infrastructure runs; that
  resulting weights are discarded except for the optional restart extension;
  and that model/tokenizer/dataset/preprocessing revisions and sample hashes are
  recorded. Keep the proposed Qwen3-1.7B-Base model, WikiText/tokenization
  pipeline, BF16 default, and correctness-only evaluation boundary unaccepted.
- **Rationale:** Exact artifacts can remain open while reproducibility and
  benchmark-purpose rules are settled. Combining choices that must be accepted
  together makes their status visible and removes duplicated inverse rows such
  as evaluation scope versus excluded model-quality benchmarks.
- **Consequences:** Model-free EXP-01 and EXP-10 can be designed without the
  shared model/data contract. Model-dependent experiments remain `proposed`
  until the unresolved workload rows and compatibility gates are accepted. No
  row in the experiment-status table is changed by this decision.

### PD-021 — Accept and pin the shared model and input dataset

- **Recorded:** 2026-07-13
- **Status:** Accepted
- **Supersedes:** PD-005's deferral of the exact model and dataset and PD-020's
  `proposed` status for the dense workload model and dataset/tokenizer pipeline.
  It leaves the BF16 default and correctness-only evaluation boundary proposed.
- **Decision:** Use `Qwen/Qwen3-1.7B-Base` and its tokenizer at immutable Hub
  revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`. Use
  `Salesforce/wikitext` `wikitext-103-raw-v1` at immutable Hub revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`. Apply the tracked
  `qwen3-wikitext-v1` preprocessing contract: preserve source split and row
  order, tokenize without added special tokens, append EOS to every non-empty
  record, retain empty records as zero-token offset/hash entries, and create
  canonical per-split token streams with document offsets and hashes.
  Fixed-length training samples are deterministic slices of those streams.
- **Rationale:** The dense model has a public Apache-2.0 base checkpoint,
  dimensions compatible with the selected parallel degrees, and an NVIDIA
  Megatron Bridge recipe. WikiText-103 is public, compact, and sufficient for
  infrastructure-focused continued-pretraining measurements without creating a
  data-engineering project. Immutable revisions and one tokenization contract
  make PyTorch and Megatron comparisons reproducible.
- **Consequences:** `transformers`, `tokenizers`, `safetensors`,
  `huggingface_hub`, `datasets`, and PyArrow become approved preparation/runtime
  libraries and are pinned in `requirements-preparation.txt`. Downloaded model,
  dataset, and processed files remain outside Git. Their generated manifest and
  hashes govern uploads to S3 and the Runpod network volume. Model-dependent
  experiment rows remain `proposed` until the remaining shared-workload choices
  and experiment-specific compatibility gates are accepted.

### PD-022 — Remove standby experiment work and add AWS run-unit IDs

- **Recorded:** 2026-07-14
- **Status:** Accepted
- **Supersedes:** The optional operational extension placement in PD-015 and
  PD-016, and PD-020's exception for keeping resulting weights for an optional
  restart extension. It preserves the 14 numbered core experiments, the
  AWS-then-Runpod provider order, and the two admitted four-GPU cases.
- **Decision:** Keep the active catalog limited to the 14 numbered experiments
  plus the final synthesis deliverable. Remove the non-numbered distributed
  checkpoint/restart extension from the current plan. Use AWS run-unit IDs
  `QUAL-A1`, `QUAL-A2`, `QUAL-A4`, and `EXP-NN-A1/A2/A4` for concrete AWS
  compute-profile queues and artifacts. Remove AWS sub-runs that are not needed
  to answer their parent experiment's hypothesis; specifically, EXP-03 remains
  a one-GPU AWS experiment and no longer has a two-GPU batch-geometry repeat.
- **Rationale:** Standby experiment work made the AWS queue ambiguous at the
  same time that `AWS-G7E-2` capacity is scarce. Concrete run units let one
  acquired host execute the required two-GPU work in a known order without
  changing canonical experiment IDs. Checkpoint/restart practice is useful
  operationally but is less central to the GPU Acceleration and Optimization
  curriculum than the existing 14 experiments.
- **Consequences:** Resulting training weights are discarded after metrics are
  collected. Future checkpoint/restart work or a removed AWS sub-run requires a
  new explicit decision and catalog update before implementation. The A2 queue
  begins with `QUAL-A2` and accepted `EXP-01-A2`; later A2 run units remain
  blocked until their parent experiments are accepted and implemented.

### PD-023 — Batch current AWS distributed work on AWS-A2 visible-GPU phases

- **Recorded:** 2026-07-14
- **Status:** Accepted
- **Supersedes:** PD-013, PD-015, PD-017, PD-018, and PD-022 only where they
  preserved a current AWS four-GPU DDP scaling point or used unsuffixed
  `EXP-NN-A2` run units for experiments that need both one- and two-visible-GPU
  phases. It preserves the 14 canonical experiment IDs, the AWS-then-Runpod
  provider order, and the Runpod TP=2 x DP=2 four-GPU hybrid.
- **Decision:** Make the current AWS-A2 queue the authoritative AWS distributed
  work queue. Run EXP-02, EXP-07, and EXP-09 one-rank or one-GPU baseline work
  on `AWS-A2` with one visible GPU, not on a separate `AWS-A1` host. Use
  run-unit suffixes `A2V1` and `A2V2` for one- and two-visible-GPU phases on
  the same billed `AWS-A2` profile. Keep EXP-08 as a two-visible-GPU AWS-A2 run.
  Remove the current EXP-07 AWS-A4 run; any future AWS four-GPU DDP point requires
  a new decision, updated queue, and lifecycle guard.
- **Rationale:** `AWS-A2` capacity has been scarce. Once acquired, it is more
  useful to execute all required AWS-A2 work and its one-visible-GPU baselines than
  to wait separately for AWS-A1 or AWS-A4 capacity. The one-to-two-rank DDP comparison
  answers the near-term communication-overlap question, while the removed AWS-A4
  point mainly supplied a second scaling doubling at materially higher capacity
  risk.
- **Consequences:** A one-visible-GPU run on AWS-A2 is not an `AWS-A1` result.
  Reports must record `physical_profile=AWS-A2`, visible GPU count,
  visibility mask, CPU/RAM shape, and billed resource. The catalog GPU-hour
  estimate drops to 17.25-33.5 measured GPU-hours, while provider-billed cost
  for masked A2 phases still charges the full two-GPU instance. EXP-02,
  EXP-07, EXP-08, and EXP-09 remain `proposed` until explicitly accepted and
  implemented.

### PD-024 — Rename current AWS G7e profile aliases to AWS-A names

- **Recorded:** 2026-07-14
- **Status:** Accepted
- **Supersedes:** The active profile aliases `AWS-G7E-1`, `AWS-G7E-2`, and
  `AWS-G7E-4` used in the current snapshot and operational files. Historical
  decision-log entries retain their original wording.
- **Decision:** Use `AWS-A1`, `AWS-A2`, and `AWS-A4` as the current project
  aliases for AWS G7e `g7e.2xlarge`, `g7e.12xlarge`, and `g7e.24xlarge`.
  The `A` prefix denotes the AWS profile group and the number denotes the
  billed physical GPU count. Visible-GPU phases on the two-GPU host use
  run-unit suffixes such as `A2V1` and `A2V2`.
- **Rationale:** The old aliases encoded the current EC2 family in the profile
  name and made queue discussions harder to read. The new aliases keep the
  operational profile stable while the exact instance type remains recorded in
  the profile table and run artifacts.
- **Consequences:** Active configs, tests, catalog rows, run-unit tables, and
  AWS operator docs use `AWS-A*` names. A one-visible-GPU phase on `AWS-A2`
  remains a two-GPU billed host and must not be reported as an `AWS-A1` result.

### PD-025 — Share the AWS PyTorch image across AWS-A1 and AWS-A2

- **Recorded:** 2026-07-14
- **Status:** Accepted
- **Supersedes:** No prior decision; clarifies the active container strategy
  for AWS G7e PyTorch work.
- **Decision:** Use one immutable AWS PyTorch image for AWS-A1 and AWS-A2 runs
  whenever the software stack is the same. Select A1 versus A2 behavior through
  provider configuration, run units, visible-GPU masks, and container command
  mode. Do not create separate A1/A2 images merely because the EC2 instance
  type or visible GPU count differs.
- **Rationale:** A shared image keeps software content identical across AWS
  one-GPU and two-GPU runs, reduces rebuilds and registry churn, and makes
  differences attributable to hardware profile, GPU visibility, and workload
  configuration rather than accidental image drift.
- **Consequences:** Image digests remain tied to software state, not compute
  profile. Qualification and experiment commands may have separate modes inside
  the same image. A new PyTorch image is required only when source,
  dependencies, profiler tooling, or runtime contracts change.

### PD-026 — Use five active execution queues while AWS capacity is scarce

- **Recorded:** 2026-07-14
- **Status:** Accepted
- **Supersedes:** PD-016 only where it used the temporary Runpod `R2` and `R4`
  session labels, and any earlier workflow wording that implied Runpod
  preparation must wait for AWS capacity. It preserves the canonical
  EXP-01 through EXP-14 experiment IDs, one-provider-per-experiment rule, AWS
  G7e assignments, and Runpod A100 SXM assignments.
- **Decision:** Use five active execution queues: `AWS-A1`, `AWS-A2`,
  `RUNPOD-A1`, `RUNPOD-A2`, and `RUNPOD-A4`. Keep AWS launch configurations,
  ECR images, S3 artifact prefixes, and retained cache volumes available for
  later AWS retries. Start Runpod preparation now, but do not launch a paid
  Runpod Pod until the exposed API key is rotated, local Runpod tooling passes
  readiness checks, registry pull access is configured, durable storage is
  planned, and the launch receives explicit approval.
- **Rationale:** Current AWS G7e capacity is unreliable even though AWS quota,
  permissions, subnets, images, and dry-runs are valid. A queue model lets the
  project make progress on Runpod without deleting or de-prioritizing the AWS
  path.
- **Consequences:** `RUNPOD-A1` means one-visible-GPU A100 SXM baselines,
  `RUNPOD-A2` means two-GPU A100 SXM readiness, communication, and
  NeMo/Megatron work, and `RUNPOD-A4` means the four-GPU A100 SXM hybrid queue.
  The Runpod `A` number is the visible GPU count for the queue. Because
  one-visible-GPU baselines may run on the same billed two-GPU Pod as their
  two-GPU comparisons, every run still records billed GPU count, visible GPU
  count, Pod resource profile, datacenter, topology, visible mask, image digest,
  and billed resource.

### PD-027 — Split active execution queues by provider, GPU count, and framework

- **Recorded:** 2026-07-14
- **Status:** Accepted
- **Supersedes:** PD-026 where it named the active execution queues. It
  preserves the decision to keep AWS retryable while Runpod preparation
  proceeds, the one-provider-per-experiment rule, the AWS G7e assignments, and
  the Runpod A100 SXM assignments.
- **Decision:** Use six active execution queues:
  `AWS-A1-PyTorch`, `AWS-A2-PyTorch`, `RUNPOD-A1-PyTorch`,
  `RUNPOD-A2-PyTorch`, `RUNPOD-A2-Megatron`, and `RUNPOD-A4-Megatron`.
  `RUNPOD-A2-Megatron` includes both one-visible-GPU and two-visible-GPU
  EXP-11 through EXP-13 phases on the two-GPU A100 SXM Pod. Do not create a
  separate `RUNPOD-A1-Megatron` queue unless a later decision admits a distinct
  one-GPU Megatron resource strategy. `RUNPOD-A4-Megatron` is currently
  relevant only to EXP-14.
- **Rationale:** GPU-count-only queue names made the image/runtime expectation
  ambiguous because Runpod has both PyTorch/NCCL readiness work and
  NeMo/Megatron training work. Framework-qualified queues make it clear which
  image family, launch contract, and validation path applies before a paid Pod
  is started.
- **Consequences:** Queue names appear in launch plans, artifact manifests, and
  reports. Resource profiles such as `AWS-A2`, `RUNPOD-A100-SXM2`, and
  `RUNPOD-A100-SXM4` remain separate from queue labels. Every run still records
  provider, billed resource, visible GPU count, image family, image digest,
  topology, datacenter/Region, and cost guard metadata.

### PD-028 — Limit the immediate AWS-A2 PyTorch launch queue to EXP-01/02/07/09

- **Recorded:** 2026-07-15
- **Status:** Accepted
- **Supersedes:** PD-023 only where it kept `EXP-08-A2V2` in the current
  AWS-A2 launch queue.
- **Decision:** The immediate `AWS-A2-PyTorch` queue contains `EXP-01-A2`,
  `EXP-02-A2V1`, `EXP-02-A2V2`, `EXP-07-A2V1`, `EXP-07-A2V2`,
  `EXP-09-A2V1`, and `EXP-09-A2V2`. `EXP-08` remains accepted as an AWS
  PyTorch experiment, but it is not part of this queued launch and needs an
  explicit later scheduling step before implementation or execution.
- **Rationale:** The first acquired two-GPU AWS capacity should focus on the
  communication baseline, precision checks, DDP scaling/overlap, and controlled
  troubleshooting queue that was requested for immediate execution. Removing
  EXP-08 from this launch keeps the queue shorter and avoids starting FSDP work
  before the priority A2 results exist.
- **Consequences:** Current AWS-A2 queue tooling, catalog run-unit maps, and
  launch documentation must exclude `EXP-08-A2V2`. Reports must not imply that
  an AWS-A2 run completed or attempted EXP-08 unless it is added back by a
  later decision and launch plan.

### PD-029 — Restore EXP-08 to the AWS-A2 completion queue

- **Recorded:** 2026-07-15
- **Status:** Accepted
- **Supersedes:** PD-028.
- **Decision:** The AWS-A2 PyTorch completion queue for finishing AWS
  experiments EXP-01 through EXP-09 contains `EXP-01-A2`, `EXP-02-A2V1`,
  `EXP-02-A2V2`, `EXP-07-A2V1`, `EXP-07-A2V2`, `EXP-08-A2V2`,
  `EXP-09-A2V1`, and `EXP-09-A2V2`. The queue stops the instance after a fully
  successful run but leaves the instance running after a queue failure for
  manual inspection, while retaining the hard maximum-lifetime safety shutdown.
- **Rationale:** The current operator goal is to complete all AWS experiments
  EXP-01 through EXP-09, not only the shorter communication/precision/DDP/fault
  subset. EXP-08 is accepted, has runnable PyTorch scaffolding, and is the
  missing two-GPU AWS memory-sharding experiment.
- **Consequences:** Current AWS-A2 queue tooling, catalog run-unit maps, and
  launch documentation include `EXP-08-A2V2`. Any failed queue retry may leave
  billable compute running until manual action or the hard safety shutdown, so
  monitoring and follow-up are required.

### PD-030 — Move EXP-12 pipeline scheduling to AWS-A2-Megatron

- **Recorded:** 2026-07-15
- **Status:** Accepted
- **Supersedes:** PD-016 and PD-027 only where they assigned EXP-12 pipeline
  parallelism to Runpod A100 SXM. It preserves Runpod placement for EXP-10,
  EXP-11, EXP-13, and EXP-14.
- **Decision:** Run EXP-12 on AWS using a new `AWS-A2-Megatron` queue on the
  existing AWS-A2 `g7e.12xlarge` profile. EXP-12 keeps the NeMo/Megatron image
  family and uses one-visible-GPU PP=1 and two-visible-GPU PP=2 phases on the
  same billed AWS-A2 host. The result must record PCIe topology explicitly and
  must not be described as an NVLink measurement.
- **Rationale:** EXP-12 studies pipeline bubble size, microbatch scheduling,
  and stage balance. Unlike EXP-10, EXP-11, EXP-13, and EXP-14, its core
  hypothesis does not require NVLink. AWS-A2 is already qualified operationally
  for two-GPU single-node runs and can answer the pipeline-schedule mechanics
  while Runpod remains blocked by registry/storage/API-key readiness.
- **Consequences:** Add an `AWS-A2-Megatron` queue, scoped S3 artifact
  permissions for `artifacts/EXP-12/` and `artifacts/AWS-A2-Megatron/`, an
  EXP-12-capable NeMo image in ECR, and an EXP-12 implementation directory.
  Runpod remains the required provider for EXP-10, EXP-11, EXP-13, and EXP-14
  unless another explicit decision changes those placements.

### PD-031 — Use one NeMo/Megatron image for the remaining Runpod phase

- **Recorded:** 2026-07-15
- **Status:** Accepted
- **Supersedes:** PD-027 where it split Runpod EXP-10 communication readiness
  into `RUNPOD-A1-PyTorch` and `RUNPOD-A2-PyTorch`. It preserves the
  one-provider-per-experiment rule, Runpod placement for EXP-10, EXP-11,
  EXP-13, and EXP-14, and the separate four-GPU EXP-14 queue.
- **Decision:** Use the NeMo/Megatron image family for the complete remaining
  Runpod phase. `RUNPOD-A2-Megatron` runs the two-GPU Pod queue containing
  `QUAL-RUNPOD-A2-Megatron`, EXP-10, EXP-11 one-/two-visible-GPU phases, and
  EXP-13 one-/two-visible-GPU phases. `RUNPOD-A4-Megatron` runs the separate
  four-GPU Pod queue for EXP-14. EXP-10 remains a model-free CUDA/NCCL
  communication experiment; the NeMo image must include the required NCCL tests
  and `p2pBandwidthLatencyTest`.
- **Rationale:** A Runpod Pod boots one container image. Using the same
  NeMo/Megatron image for EXP-10, EXP-11, and EXP-13 allows one acquired
  two-GPU A100 SXM Pod to run the complete two-GPU Runpod queue without nested
  Docker or image switching. The NeMo container already includes NVIDIA's
  optimized PyTorch and NCCL stack, so adding the CUDA P2P sample covers the
  communication baseline without changing the measured provider or topology.
- **Consequences:** The current Runpod blockers are now the NeMo GHCR digest,
  a pull-only Runpod GHCR registry auth, a Runpod network volume for durable
  artifacts, and explicit paid Pod approval. A separate Runpod PyTorch image is
  not required for the current EXP-10 launch path, though the PyTorch image
  family remains available for AWS and any future explicitly accepted Runpod
  PyTorch work.

## Primary references

- [NVIDIA NCP-GENL certification and exam blueprint](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
- [NVIDIA NeMo Framework software component versions](https://docs.nvidia.com/nemo-framework/user-guide/latest/softwarecomponentversions.html)
- [NVIDIA Megatron Core installation guidance](https://docs.nvidia.com/megatron-core/developer-guide/nightly/get-started/install.html)
- [NVIDIA Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [AWS EC2 G7e instance specifications](https://aws.amazon.com/ec2/instance-types/g7e/)
- [AWS EC2 G6e instance specifications](https://aws.amazon.com/ec2/instance-types/g6e/)
- [AWS EC2 accelerated-computing instance specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [AWS EC2 instance-type quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)
- [AWS EC2 P4 NVSwitch instance specifications](https://aws.amazon.com/ec2/instance-types/p4/)
- [Amazon ECR registry authentication](https://docs.aws.amazon.com/AmazonECR/latest/userguide/registry_auth.html)
- [Amazon ECR pricing](https://aws.amazon.com/ecr/pricing/)
- [GitHub Container Registry authentication](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Runpod network volumes](https://docs.runpod.io/storage/network-volumes)
- [Runpod A100 SXM GPU type](https://docs.runpod.io/references/gpu-types)
- [NVIDIA CUDA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus)
- [NVIDIA Transformer Engine supported hardware](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)
- [Transformer Engine FP8 Delayed Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)
- [Transformer Engine MXFP8 support](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)
- [Transformer Engine NVFP4 support](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
