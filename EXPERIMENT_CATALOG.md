# Proposed experiment catalog

Status: Proposed experiments and selection worksheet
Last updated: 2026-07-11
Governing decisions: [PROJECT_DECISIONS.md](PROJECT_DECISIONS.md)

## Document role and authority

This file owns experiment IDs, definitions, recommendations, selection status,
hypotheses, measurements, estimated GPU-hours, and implementation order. It may
also contain proposed project-wide choices, such as a model or dataset, while
those choices are under review.

Accepted scope, infrastructure, frameworks, exclusions, and tools are
maintained only in [PROJECT_DECISIONS.md](PROJECT_DECISIONS.md). If a
project-wide proposal here is accepted, it must be promoted to that decision
record. If the files conflict on a project constraint, `PROJECT_DECISIONS.md`
takes precedence.

## Purpose

This document is the decision worksheet for selecting the experiments to
implement. It translates the Ultra-Scale Playbook and the GPU Acceleration and
Optimization portion of NCP-GENL into experiments that fit the governing
project decisions.

The recommendations below are proposals, not accepted decisions. Change the
`Decision` column to `accept`, `defer`, or `reject` during review. Accepted
experiments will later receive their own directory and detailed design.

## Proposed shared workload contract

These decisions must be accepted before individual experiments are designed.
The recommendation is to use one exact dense model and one existing dataset for
all comparable training experiments. Changing either is allowed only when the
experiment's hypothesis makes it unavoidable.

| Decision area | Proposed choice | Status |
| --- | --- | --- |
| Primary model | `Qwen/Qwen3-1.7B-Base` | TBD |
| Number of dense models | One exact model/checkpoint for all dense experiments | TBD |
| Initialization | Start every comparable variant from the same pretrained base checkpoint | TBD |
| Training type | Full-parameter continued pretraining, not SFT, LoRA, or training from scratch | accepted |
| Training objective | Autoregressive causal language modeling with next-token cross-entropy | accepted |
| Dataset | `Salesforce/wikitext`, configuration `wikitext-103-raw-v1` | TBD |
| Tokenizer | Tokenizer shipped with `Qwen/Qwen3-1.7B-Base` | TBD |
| Data preparation | Tokenize and pack once; no project-specific cleaning, curation, or deduplication | TBD |
| Evaluation scope | Correctness and short loss/perplexity sanity checks only | TBD |
| Model-quality benchmarks | Exclude downstream benchmark suites and training-to-convergence | TBD |

Rows marked `accepted` summarize workload rules already governed by
[PROJECT_DECISIONS.md](PROJECT_DECISIONS.md); they are repeated here only to
make the proposed workload readable. Rows marked `TBD` remain proposals and do
not become project decisions until they are promoted to that file.

### Why this model

`Qwen/Qwen3-1.7B-Base` is proposed because it is:

- A decoder-only causal language model suitable for the selected training
  objective.
- Small enough for full training steps on one A100 SXM, making one-GPU baselines
  possible.
- Large enough to produce meaningful compute, memory, and communication traces
  on two and four GPUs.
- Structurally suitable for the proposed parallel degrees: 28 transformer
  layers, 16 query heads, and 8 key/value heads permit the principal TP and PP
  divisions used in this catalog.
- Available under Apache-2.0 without a gated-license workflow.
- Supported by NVIDIA Megatron Bridge through an existing Qwen3 1.7B pretraining
  recipe, avoiding a new NeMo/Megatron model implementation.

The project uses the base model, not the instruction/chat-tuned model. The
pretrained checkpoint is restored before each comparable variant so that
precision or parallelism changes are measured from identical weights.

For native PyTorch experiments, Hugging Face Transformers may supply the Qwen3
model definition and checkpoint loader, but the training loop, optimizer, and
all distributed behavior remain native PyTorch. Hugging Face Trainer and
Accelerate remain excluded. For NeMo/Megatron experiments, the checkpoint is
converted once through Megatron Bridge and reused.

### Compatibility review

No known blocker prevents Qwen3-1.7B-Base from serving as the primary workload,
but not every combination has model-specific vendor certification. The project
must distinguish documented support from generic compatibility and empirical
validation.

| Component or experiment feature | Compatibility assessment | Required action |
| --- | --- | --- |
| Docker/OCI, GHCR, Runpod, Git, SSH, and storage tooling | Model-agnostic | Verify image pull, artifact paths, and available disk space |
| Hugging Face model, tokenizer, and Safetensors | Officially supported; Qwen3 requires Transformers 4.51 or newer | Pin the exact model revision and a compatible Transformers version |
| Hugging Face Datasets/Hub and WikiText-103 | Model-agnostic download/preprocessing path | Pin dataset revision and preserve license/attribution metadata |
| Native PyTorch forward/backward and AdamW | Official model implementation is a PyTorch `nn.Module` | Compare a fixed-batch loss and update with the published checkpoint |
| PyTorch DDP and NCCL | Model-agnostic distributed wrappers/collectives | Two-GPU correctness smoke test |
| BF16, FP16, FP32, and TF32 on A100 | Compatible with native PyTorch; BF16 is the default | Validate numerical tolerances and Tensor Core use in EXP-001 |
| PyTorch activation checkpointing | Generic compatibility | Check Qwen block wrapping and loss agreement |
| PyTorch FSDP2 | Expected generic compatibility, but no Qwen3-specific FSDP2 recipe was found | Validate wrapping, tied/shared parameters, state dict, and resume before EXP-010 |
| PyTorch SDPA/Flash Attention backends | Qwen3 uses GQA; optimized backends are available but backend selection is shape/version dependent | Log the selected kernel/backend and retain a math-backend correctness baseline |
| `torch.compile`/Inductor | Transformers supports compiled training generally; no Qwen3 full-training guarantee was found | Treat compilation as an EXP-004 hypothesis and record graph breaks/fallbacks rather than assuming success |
| PyTorch Profiler and TensorBoard | Model-agnostic | Smoke test trace export |
| Nsight Systems, Nsight Compute, NVTX, DCGM, and `nvidia-smi` | Model-agnostic; low-level counters depend on Runpod permissions | Check permissions in the mandatory pre-run qualification |
| CUDA, cuBLAS/cuBLASLt, cuDNN, NCCL, `nccl-tests`, and CUDA Samples | Model-agnostic; used directly or beneath PyTorch/Transformer Engine | Pin via the NGC image and run topology/collective smoke tests |
| NeMo Framework, Megatron Core, and Megatron Bridge | Official Qwen3 support, Qwen3-1.7B pretraining recipe, and HF/Megatron conversion | Pin the NeMo container and perform one round-trip logit/checkpoint comparison |
| Transformer Engine and optimized/fused attention | Megatron Bridge exposes TE attention backends and Qwen3 maps to Megatron model modules | Smoke test the selected A100 backend and log fallbacks |
| NVIDIA Apex | Not required by Qwen itself; use only if bundled and required by the pinned NVIDIA stack | Do not add a separate Apex dependency without a demonstrated need |
| TP=2/4 and sequence parallelism | Framework-supported and Qwen dimensions are divisible; the official 1.7B recipe recommends TP=1 for efficiency | Validate TP=2/4 correctness; treat poor performance as a valid result |
| PP=2/4 | Framework-supported; 28 layers are divisible by 2 and 4 | Validate stage assignment, embedding/loss placement, and balance |
| CP=2/4 | Framework-supported; use sequence lengths divisible by CP and no more than the model's 32K context | Validate CP attention backend and fixed-token loss agreement |
| TP=2 x DP=2, TP=2 x CP=2, and TP=2 x PP=2 x DP=2 | Megatron Bridge supports combined process groups and the dimensions are structurally valid | Each hybrid needs a short rank-map and correctness smoke test before profiling |
| Distributed checkpoints | PyTorch and Megatron Bridge support their respective formats | Do not assume arbitrary TP/PP reshaping; test only the layout changes explicitly supported |
| WikiText token stream | Model-independent once tokenized with the pinned Qwen tokenizer | Generate once and verify identical sample hashes in PyTorch and Megatron loaders |
| NumPy, pandas, SciPy, Matplotlib, Seaborn, PyYAML, pytest, Ruff, and `psutil` | Operate on configurations, tests, telemetry, or saved metrics; model-agnostic | Pin versions and validate the analysis pipeline locally |

The verdict is therefore **suitable with targeted smoke tests**, not
"automatically supported by every possible combination." The container build is
not accepted until a compatibility job passes the required actions above on an
A100 Pod.

### One model versus multiple models

The default is **one exact dense model**, not a collection of convenient models.
This keeps tokens/second, memory, numerical behavior, and loss comparisons
interpretable across DDP, FSDP, TP, PP, CP, and hybrid layouts.

No other experiment may silently change model size, layer count, vocabulary,
attention type, or checkpoint. If the primary model proves technically
incompatible with an accepted parallel layout, that is a project-level decision
to revisit the model—not permission to substitute a model inside one result.

### Dataset and tokenization

The proposed dataset is the existing `Salesforce/wikitext` dataset with the
`wikitext-103-raw-v1` configuration. It contains train, validation, and test
splits and is small enough to download and preprocess once without making data
engineering a project of its own.

The preparation pipeline is deliberately minimal:

1. Pin the dataset revision and download the existing splits.
2. Tokenize with the pinned Qwen3 tokenizer.
3. Insert the model's EOS token between documents.
4. Concatenate and pack tokens into deterministic fixed-length sequences.
5. Materialize one canonical token stream plus adapters/indexes needed by
   native PyTorch and Megatron.
6. Preserve the existing train/validation/test split; do not curate, clean,
   deduplicate, or create a custom corpus.

The same token IDs and sample order must be used across framework and
parallelism comparisons. Sequence length may change only when it is the
independent variable, such as context-parallel or memory experiments. Packing
across document boundaries is acceptable because model quality is not the
research target.

Deterministic synthetic token IDs remain allowed for pure compute/communication
microbenchmarks and fault injection, but they cannot replace WikiText in an
end-to-end training comparison.

If the shared workload contract is accepted, every end-to-end training
experiment uses WikiText-103 and the pinned Qwen tokenizer. EXP-007/EXP-008 do
not train a model, and isolated kernel/collective sub-benchmarks may use
synthetic tensors.

### Training type and objective

The shared training workload is **full-parameter continued pretraining** from
the Qwen3 base checkpoint. Every trainable parameter participates in forward,
backward, gradient synchronization/sharding, optimizer state, and checkpointing.
This exercises the complete distributed-training path without the cost of
pretraining a useful model from random initialization.

The objective is standard next-token prediction:

```text
input  = tokens[0 : sequence_length]
target = tokens[1 : sequence_length + 1]
loss   = mean cross_entropy(model(input), target)
```

SFT, RLHF, preference optimization, LoRA/PEFT, instruction templates, and
generation objectives are excluded. They introduce training behaviors that are
not necessary for the infrastructure questions in this catalog.

Unless the experiment studies one of these variables, optimizer, learning rate,
weight decay, precision, sequence length, and effective global batch remain
fixed in a shared workload configuration. BF16 is the proposed default
precision on A100; EXP-001 is the explicit exception.

### Expected effect of continued pretraining

If continued for enough well-tuned steps, WikiText training would move the base
model's probability distribution toward English Wikipedia-style prose and the
topics present in WikiText. Validation loss/perplexity on WikiText might improve,
but that does not establish that the model became a useful subject-matter
expert. The corpus is small relative to Qwen's original pretraining, and our
benchmark windows are intentionally short.

For this project, the updated model is not the product. Each comparison starts
again from the identical base checkpoint, and resulting weights are discarded
after metrics are collected except when a checkpoint/restart experiment needs
them. We care that the loss and updates are valid and equivalent across
configurations—not that the short run improves general knowledge or downstream
answers.

Longer continued pretraining could overfit WikiText or shift performance away
from other domains (catastrophic forgetting). Studying those effects would
require broader evaluation and belongs to model training/evaluation, not this
infrastructure project.

### Run profiles

Experiments use standardized profiles instead of inventing a different training
duration each time:

| Profile | Purpose | Proposed minimum |
| --- | --- | --- |
| Smoke | Detect configuration, launch, and immediate OOM failures | 3 complete optimizer steps |
| Correctness | Compare loss/gradients/updates from a fixed checkpoint and fixed batches | 5 deterministic optimizer steps |
| Benchmark | Measure steady-state performance | 20 warm-up + 100 measured steps, repeated when variance requires |
| Resume | Validate checkpoint continuation | Save, reload, and compare at least the next 3 steps |

These are measurement windows, not attempts to train the model to convergence.
An experiment may shorten a profile when profiler overhead is extreme, but must
justify the change and still collect enough iterations for a stable conclusion.

### Evaluation policy

The project remains infrastructure-focused, but evaluation cannot be removed
entirely: a faster run is meaningless if it computes a different update.

Required evaluation is limited to:

- Finite loss and gradients.
- Initial loss/logit agreement for implementations expected to be equivalent.
- Loss, selected gradient, and parameter-update comparison over the deterministic
  correctness profile, using precision-appropriate tolerances.
- Training loss during the measured run.
- Validation cross-entropy and perplexity on a small fixed WikiText validation
  slice before and after experiments that perform optimizer updates.
- Exact or tolerance-based post-resume agreement for checkpoint experiments.

Not required:

- Training to convergence.
- MMLU, HellaSwag, HumanEval, or another downstream benchmark suite.
- Human evaluation, generation-quality scoring, or comparison with published
  Qwen quality numbers.
- A claim that the short continued-pretraining run improves the model.

Thus evaluation acts as a **correctness guardrail**, not as a separate model
evaluation research program.

### Proposed supporting libraries

Accepting this workload contract adds the following narrowly scoped libraries
to the approved toolset:

- `transformers` and `safetensors` for the Qwen model definition/checkpoint in
  native PyTorch experiments.
- `datasets` and `huggingface_hub` for downloading the pinned public dataset,
  tokenizer, and model artifacts.
- Megatron Bridge, supplied by the selected NeMo container, for the NVIDIA
  implementation and one-time checkpoint conversion.

These libraries do not authorize Hugging Face Trainer, Accelerate, or another
distributed-training framework.

## Selection vocabulary

- **Core:** recommended for the main curriculum. Removing it leaves a material
  gap in GPU optimization, profiling, distributed training, or parallelism.
- **Optional:** useful, but overlaps another experiment, has a narrower use
  case, or has a relatively high implementation/cost burden.
- **GPU-hours:** an initial target for the final measured run, after code is
  working. It excludes image pulls, development, debugging, and failed runs.

## Proposed selection summary

| ID | Experiment | Tags | Stack | GPUs | Target GPU-hours | Recommendation | Decision |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| EXP-001 | Mixed precision and Tensor Cores in distributed training | `precision` `tensor-cores` `ddp` | PyTorch | 1, 2 | 1.0-2.0 | Core | TBD |
| EXP-002 | Microbatch, global batch, and gradient accumulation | `batching` `memory` `ddp` | PyTorch | 1, 2 | 1.0-2.0 | Core | TBD |
| EXP-003 | Activation checkpointing/recomputation | `memory` `recompute` | PyTorch | 1 | 0.5-1.0 | Core | TBD |
| EXP-004 | PyTorch SDPA/FlashAttention, fusion, and `torch.compile` | `kernels` `attention` `flash-attention` `compile` | PyTorch | 1 | 0.75-1.5 | Core | TBD |
| EXP-005 | Profiler triangulation | `profiling` `nsys` `ncu` | PyTorch/NVIDIA tools | 1 | 0.75-1.5 | Core | TBD |
| EXP-006 | Input-pipeline starvation | `data-pipeline` `cpu` `ddp` | PyTorch | 1, 4 | 1.0-2.5 | Optional | TBD |
| EXP-007 | GPU topology and peer-to-peer paths | `topology` `p2p` `nvlink` | NVIDIA tools | 2, 4 | 1.0-2.0 | Core | TBD |
| EXP-008 | NCCL collective benchmark | `nccl` `collectives` `topology` | NCCL/NVIDIA tools | 2, 4 | 2.0-4.0 | Core | TBD |
| EXP-009 | DDP scaling and communication overlap | `ddp` `scaling` `overlap` | PyTorch | 1, 2, 4 | 3.0-6.0 | Core | TBD |
| EXP-010 | FSDP sharding and ZeRO-style memory trade-offs | `fsdp` `sharding` `memory` | PyTorch | 2, 4 | 2.0-4.0 | Core | TBD |
| EXP-011 | Tensor plus sequence parallelism | `megatron` `tp` `sp` | NeMo/Megatron | 1, 2, 4 | 2.0-4.0 | Core | TBD |
| EXP-012 | Pipeline schedules and bubble size | `megatron` `pp` `scheduling` | NeMo/Megatron | 2, 4 | 2.0-4.0 | Core | TBD |
| EXP-013 | Context parallelism for long sequences | `megatron` `cp` `long-context` | NeMo/Megatron | 1, 2, 4 | 2.0-4.0 | Core | TBD |
| EXP-014 | TP=2 x DP=2 for model width and throughput | `megatron` `hybrid` `tp` `dp` | NeMo/Megatron | 4 | 2.0-3.0 | Core | TBD |
| EXP-015 | TP=2 x CP=2 for model width and long context | `megatron` `hybrid` `tp` `cp` | NeMo/Megatron | 4 | 2.0-3.0 | Optional | TBD |
| EXP-016 | Distributed checkpoint and restart | `checkpointing` `recovery` `sharding` | PyTorch and NeMo/Megatron | 2, 4 | 2.0-4.0 | Core | TBD |
| EXP-017 | Controlled troubleshooting and failure diagnosis | `troubleshooting` `nccl` `oom` | PyTorch | 2 | 1.0-2.0 | Core | TBD |
| EXP-018 | End-to-end configuration-selection capstone | `capstone` `cost` `comparison` | Both, separate runs | 1, 2, 4 | 3.0-6.0 | Core | TBD |

If all 16 core experiments are accepted, their row estimates sum to 26-51
measured GPU-hours. Including both optional experiments gives 29-56.5 measured
GPU-hours. First-time debugging and profiler setup can make the billable total
materially higher.

## Tag index

- Foundations: `precision` (EXP-001), `batching` (EXP-002), `memory`
  (EXP-002, EXP-003, EXP-010), `kernels`/`flash-attention` (EXP-004),
  `profiling` (EXP-005), and
  `data-pipeline` (EXP-006).
- Communication: `topology` (EXP-007, EXP-008), `p2p` (EXP-007), `nccl`
  (EXP-008, EXP-017), and `collectives` (EXP-008).
- PyTorch distributed: `ddp` (EXP-001, EXP-002, EXP-006, EXP-009), `fsdp`
  (EXP-010), and `sharding` (EXP-010, EXP-016).
- NeMo/Megatron parallelism: `tp` (EXP-011, EXP-014, EXP-015), `sp`
  (EXP-011), `pp` (EXP-012), and `cp` (EXP-013, EXP-015).
- Integration: `hybrid` (EXP-014, EXP-015), `checkpointing` (EXP-016),
  `troubleshooting` (EXP-017), and `capstone` (EXP-018).

## Mandatory pre-run qualification

Environment and topology qualification is a prerequisite check, not an
experiment. A shared script must run at the start of every Pod session and its
output must be attached to every experiment executed in that session.

The check captures:

- Exact GPU model and memory, GPU count, UUIDs, MIG state, clocks, and power
  limits.
- Driver, CUDA, cuDNN, NCCL, PyTorch, NeMo, Megatron Core, and Transformer
  Engine versions as applicable.
- `nvidia-smi topo -m`, peer-access capability, and visible devices.
- Container image reference/digest, Git commit, CPU/RAM, mounted storage, and
  relevant environment variables.
- Availability and permissions for DCGM, Nsight Systems, Nsight Compute, and
  hardware counters.
- A short CUDA/PyTorch/NCCL smoke test appropriate to the allocated GPU count.

An experiment is invalid if this check shows the wrong GPU family, missing
devices, an unexplained topology change, or a software mismatch.

## Common experimental contract

Unless an experiment explicitly tests one of these variables, comparisons must
hold constant:

- Model architecture, parameter count, sequence length, token count, optimizer,
  precision, and effective global batch size.
- Data order or deterministic synthetic token generation.
- Warm-up policy and number of measured iterations.
- Container digest, software versions, GPU type, and topology.
- Correctness gates: finite loss, comparable initial loss, expected parameter
  updates, and appropriate numerical tolerance.

Every performance experiment should report the applicable subset of:

- Step latency distribution, tokens/second, samples/second, and scaling
  efficiency.
- Peak allocated/reserved GPU memory and estimated model-state/activation
  memory.
- SM activity, Tensor Core use, kernel time, GPU idle time, CPU utilization,
  power, and achieved memory bandwidth.
- Time and bytes attributable to NCCL collectives or point-to-point transfers.
- Model FLOP utilization (MFU) only when the FLOP estimate is defined and
  documented; MFU must not be compared across incompatible formulas.
- Loss/gradient agreement and whether the optimization changed numerical
  behavior.

Memory anatomy and OOM analysis are also shared measurement tasks rather than a
standalone experiment. Whenever an experiment changes precision, batch shape,
sequence length, sharding, checkpointing, or model parallelism, its report must:

- Estimate parameter, gradient, optimizer-state, and activation memory.
- Record per-rank allocated, reserved, and peak GPU memory at meaningful phase
  boundaries.
- Explain important differences between the analytical estimate and measured
  memory, including temporary buffers, caching, and fragmentation where
  relevant.
- Record the OOM boundary only when finding or moving that boundary is part of
  the experiment's hypothesis; experiments need not intentionally crash merely
  to produce an OOM result.

For scaling from one GPU to `N` GPUs, report both speedup and efficiency:

```text
speedup(N)    = throughput(N) / throughput(1)
efficiency(N) = speedup(N) / N
```

Synthetic tensors/tokens are preferred for isolated compute and communication
microbenchmarks. End-to-end training comparisons use the canonical packed
WikiText token stream defined above.

### Experiment admission and measurement protocol

Before GPU time is authorized, an accepted experiment must define:

- One falsifiable hypothesis and its expected direction of change.
- The baseline, independent variables, fixed variables, and correctness
  tolerance.
- The minimum metrics needed to confirm or reject the hypothesis.
- The planned GPU configurations and GPU-hour limit.
- A local test plus the smallest applicable GPU smoke test.
- The analysis script and expected plots/tables; analysis must not be invented
  after seeing the result.

Benchmark timing must exclude image download, model loading, compilation, and
warm-up unless one of those costs is the subject of the experiment. Distributed
end-to-end timing uses barriers around the measured window; CUDA kernel timing
uses CUDA events or profiler timestamps rather than unsynchronized CPU clocks.

Each final configuration should provide at least three measured windows and
report median throughput, step-time percentiles, and variability. A profiler
run is collected separately because profiler overhead can distort the primary
benchmark. A previously collected baseline may be reused only when its
container digest, hardware topology, workload configuration, and measurement
method are identical.

## Detailed proposals

The company situations below are realistic synthetic scenarios written in the
style of certification questions. They explain practical motivation without
claiming that a named real company disclosed the incident.

### EXP-001: Mixed precision and Tensor Cores in distributed training

Tags: `precision` `tensor-cores` `ddp` `pytorch` `1-gpu` `2-gpu`

**Scenario (exam style):** A financial-services company moves LLM training to
A100 GPUs. FP32 training is stable but expensive, while an FP16 trial produces
non-finite gradients and two-GPU scaling is weaker than expected. Which
precision mode should the team use, and how should it verify that Tensor Cores
are active without sacrificing acceptable numerical behavior?

**Question:** How do FP32, TF32, BF16, and FP16 affect Tensor Core utilization,
numerical behavior, and one-to-two-GPU DDP scaling on A100?

**Sweep:** Precision, aligned versus deliberately misaligned matrix dimensions,
and automatic mixed precision/gradient scaling where applicable. Include a GEMM
microbenchmark, one-GPU transformer steps, and the same workload under two-GPU
DDP with constant effective global batch.

**Measurements:** Throughput, kernel selection, Tensor Core activity, memory,
loss/gradient difference from the FP32 reference, and FP16 overflow behavior.

**Expected result:** Tensor-Core-compatible shapes and reduced precision improve
throughput; BF16 is normally more numerically robust than FP16 because of its
wider exponent range. TF32 accelerates eligible FP32 matrix operations while
retaining FP32 storage.

**Boundary:** A100 has no native FP8 Tensor Cores, so FP8 is not included as an
execution variant.

**Scope note:** This is GPU execution optimization, not model optimization. The
model architecture and parameter count remain unchanged; the experiment studies
how A100 executes the same training computation and how precision changes the
compute-to-communication balance in DDP.

### EXP-002: Microbatch, global batch, and gradient accumulation

Tags: `batching` `memory` `ddp` `pytorch` `1-gpu` `2-gpu`

**Scenario (exam style):** A retailer doubles its training GPU count but keeps
the old microbatch and accumulation settings. Throughput improves, yet the
effective global batch doubles and the loss curve no longer matches the
baseline. Which batch terms must be changed to make the scaling comparison
valid, and which setting maximizes throughput within memory limits?

**Question:** How do microbatch size and accumulation steps trade memory,
utilization, optimizer frequency, and throughput while preserving effective
global batch size?

**Sweep:** Several `(microbatch, accumulation_steps)` pairs with constant global
batch. Optionally repeat on two GPUs to demonstrate the DP term in
`global_batch = microbatch x accumulation_steps x data_parallel_size`.

**Measurements:** Peak memory, tokens/second, step time per optimizer update,
GPU utilization, number of synchronization operations, loss, and gradient
agreement.

**Expected result:** Larger microbatches generally improve arithmetic intensity
until memory pressure or kernel behavior reverses the gain; accumulation permits
a larger effective batch but does not reproduce every property of one physically
large batch unless loss normalization and synchronization are correct.

### EXP-003: Activation checkpointing/recomputation

Tags: `memory` `recompute` `pytorch` `1-gpu`

**Scenario (exam style):** A legal-technology company can train its model at a
4K-token context, but an 8K-token run OOMs. Buying more GPUs is possible but
expensive. Which activations should be recomputed, how much memory should be
recovered, and what throughput penalty would justify the change?

**Question:** How much memory does recomputation save, and what compute penalty
does it impose?

**Sweep:** No checkpointing, selective/block checkpointing, and full/uniform
checkpointing where supported, with identical model and batch.

**Measurements:** Peak memory, forward/backward time, tokens/second, extra kernel
work, and maximum model/sequence/microbatch that fits.

**Expected result:** Checkpointing reduces retained activation memory while
increasing backward computation. Selective policies should offer a better
trade-off than recomputing everything for many workloads.

### EXP-004: PyTorch SDPA/FlashAttention, fusion, and `torch.compile`

Tags: `kernels` `attention` `sdpa` `flash-attention` `compile` `pytorch`
`1-gpu`

**Scenario (exam style):** An AI startup's profiler shows thousands of short
CUDA kernels separated by launch gaps, and eager attention materializes a large
score matrix. The GPU has free compute capacity. Should the team enable
FlashAttention, compile the graph, or do both—and how can it isolate the
contribution of each change?

**Question:** Which memory and throughput changes come from PyTorch's attention
backend, and which come from general graph compilation/fusion?

**Sweep:** Use fixed shapes and compare, in order:

1. Eager/manual attention or the PyTorch SDPA math backend as the correctness
   baseline.
2. PyTorch SDPA automatic backend selection, while recording the selected
   backend.
3. PyTorch SDPA with `SDPBackend.FLASH_ATTENTION` forced when Qwen3's GQA shape,
   dtype, and A100 support it; record a skip/fallback rather than relabeling
   another kernel as FlashAttention.
4. `torch.compile` on the math/eager path to isolate compilation.
5. `torch.compile` combined with the validated FlashAttention path to measure
   whether the optimizations are complementary.

Efficient-attention or cuDNN SDPA backends may be recorded as secondary variants
if supported by the pinned PyTorch build. Do not install a separate attention
package unless the integrated PyTorch backend is unavailable and the project
decision is explicitly revisited.

**Measurements:** End-to-end step time, attention time, kernel count, launch
gaps, selected SDPA backend/kernel, attention-score materialization, peak memory,
compilation/warm-up cost, graph breaks, and numerical agreement.

**Expected result:** FlashAttention should reduce attention memory traffic and
peak activation memory by avoiding materialization of the full score matrix.
`torch.compile` should primarily reduce general launch/graph overhead and fuse
eligible operations. Either may help without the other. Compilation may not win
for tiny or shape-changing workloads, and its one-time cost must be excluded
from steady-state timing but reported separately.

**NeMo/Megatron boundary:** NeMo/Megatron runs use Transformer Engine attention
with its pinned backend policy (normally automatic selection). They must record
the actual Transformer Engine backend, but they do not repeat this PyTorch
backend sweep unless attention itself is the independent variable.

### EXP-005: Profiler triangulation

Tags: `profiling` `nsys` `ncu` `pytorch-profiler` `1-gpu`

**Scenario (exam style):** A media company sees only 35% average GPU utilization
during LLM training. One engineer suspects slow Python launches, another
suspects an inefficient CUDA kernel, and a third suspects synchronization.
Which profiler should answer each question, and what evidence distinguishes
these bottlenecks?

**Question:** Can the same bottleneck be identified at framework, system
timeline, and individual-kernel levels?

**Procedure:** Profile one controlled slow variant and one optimized variant
from EXP-004 with PyTorch Profiler, Nsight Systems, and Nsight Compute. Add NVTX
ranges for data loading, forward, backward, communication, and optimizer work.

**Measurements:** Compare operator attribution, CPU/GPU overlap, CUDA launch
gaps, synchronization, memory transfers, kernel occupancy, and memory/compute
limits. Record unavailable hardware counters rather than substituting guesses.

**Why core:** Performance profiling and troubleshooting are explicit NCP-GENL
objectives; this experiment teaches when each profiler is appropriate.

### EXP-006: Input-pipeline starvation

Tags: `data-pipeline` `cpu` `ddp` `pytorch` `1-gpu` `4-gpu`

**Scenario (exam style):** An e-commerce company scales from one to four GPUs,
but aggregate throughput barely changes and Nsight Systems shows long gaps
before each forward pass. Synthetic tokens remove the gaps. Which DataLoader,
host-memory, and storage changes should the team test before blaming NCCL?

**Question:** When does data preparation prevent expensive GPUs from reaching
useful utilization?

**Sweep:** Synthetic pre-generated tokens versus a fixed tokenized dataset;
DataLoader worker count, pinned memory, prefetching, and persistent workers.
Repeat the important cases with four DDP ranks.

**Measurements:** Data wait time, CPU/RAM, host-to-device copy time, GPU idle
gaps, tokens/second, and storage throughput.

**Why optional:** Important operationally, but less central to the Playbook's
parallelism story and partially separable from GPU training.

### EXP-007: GPU topology and peer-to-peer paths

Tags: `topology` `p2p` `nvlink` `nvidia-tools` `2-gpu` `4-gpu`

**Scenario (exam style):** A research team sees different scaling from two
apparently identical four-GPU A100 allocations. The software and batch sizes
match, but the selected GPU pairs have different paths in `nvidia-smi topo -m`.
Which peer-to-peer test would establish whether topology explains the result?

**Question:** Do observed GPU-to-GPU transfer characteristics agree with the
reported NVLink/NVSwitch/PCIe topology?

**Sweep:** Relevant GPU pairs and buffer sizes with CUDA
`p2pBandwidthLatencyTest`; compare peer access and host-staged behavior where a
safe toggle is available. Run on every GPU count that will be used later.

**Measurements:** Unidirectional/bidirectional bandwidth, latency, peer-access
matrix, topology labels, and link counters when accessible.

**Expected result:** Link placement and transfer size materially affect achieved
bandwidth and latency; later process groups should be interpreted using the
measured topology rather than GPU count alone.

### EXP-008: NCCL collective benchmark

Tags: `nccl` `collectives` `topology` `2-gpu` `4-gpu`

**Scenario (exam style):** A cloud team observes that DDP performs well for a
large model but poorly for a small model on the same four GPUs. Before changing
the training code, it needs to know whether gradient messages lie in NCCL's
latency-bound or bandwidth-bound regime. Which collective and message-size
sweep should it run, and which bandwidth figure should it report?

**Question:** How do collective type, message size, and world size affect
single-node communication cost?

**Sweep:** `all_reduce`, `reduce_scatter`, `all_gather`, `broadcast`, and
`all_to_all` with 2 and 4 GPUs over small-to-large messages.
Default NCCL settings are the primary result; environment tuning is a small
diagnostic appendix, not an open-ended search for a lucky configuration.

**Measurements:** Algorithm and bus bandwidth, latency, scaling, topology,
NCCL debug output, and profiler timeline.

**Expected result:** Small messages are latency-bound, large messages approach
the fabric's bandwidth regime, and collective algorithms have different
communication volumes. These results explain later DDP, FSDP, TP, PP, CP, and
hybrid traces.

### EXP-009: DDP scaling and communication overlap

Tags: `ddp` `scaling` `overlap` `pytorch` `1-gpu` `2-gpu` `4-gpu`

**Scenario (exam style):** A software company expects four A100s to train nearly
four times faster than one, but measures only 1.9x speedup. GPU timelines show
all-reduces extending beyond backward computation. Should it increase local
work, change bucket behavior, accumulate gradients locally, or conclude the
model is too communication-heavy for DDP?

**Question:** When does replicated data parallelism scale well within one
server, and when do gradient synchronization and small local batches dominate?

**Sweep:** 1, 2, and 4 GPUs with fixed per-GPU batch (weak-scaling
view) and fixed global batch (strong-scaling view). Test a small, justified set
of bucket sizes and gradient accumulation with `no_sync`.

**Measurements:** Throughput, speedup, efficiency, peak memory per GPU,
all-reduce time, backward/communication overlap, bucket readiness, and loss
equivalence.

**Expected result:** Scaling is better when each GPU has enough computation to
hide all-reduce. Fixed-global-batch scaling eventually loses efficiency as work
per rank shrinks. Accumulating locally should reduce synchronization frequency
when implemented correctly.

### EXP-010: FSDP sharding and ZeRO-style memory trade-offs

Tags: `fsdp` `sharding` `memory` `pytorch` `2-gpu` `4-gpu`

**Scenario (exam style):** A healthcare company can train its model with DDP on
four A100s, but assumes full sharding must be better because it uses less memory.
Which states does FSDP shard, which extra collectives appear, and why can the
memory-saving configuration be slower when the unsharded model already fits?

**Question:** How do replicated DDP and optimizer/gradient/parameter sharding
trade memory for additional collectives?

**Sweep:** DDP baseline and the supported FSDP2 equivalents of sharded model
states, including full sharding. Compare wrapping/prefetch choices only after a
correct baseline. Prefer FSDP2 for new code if the pinned PyTorch release is
compatible; otherwise record the reason for using FSDP1.

**Measurements:** Per-rank model-state and peak memory, analytical capacity
projection, all-gather/reduce-scatter traffic, throughput, overlap,
initialization time, checkpoint size, and loss agreement. Do not change model
size to manufacture an OOM result.

**Expected result:** More complete sharding lowers persistent per-rank memory but
adds parameter materialization and communication. It may be slower than DDP for
a model that already fits comfortably.

### EXP-011: Tensor plus sequence parallelism

Tags: `megatron` `tp` `sp` `nvlink` `1-gpu` `2-gpu` `4-gpu`

**Scenario (exam style):** An enterprise enables TP=4 on a model that already
fits on one A100 and expects a fourfold speedup. Instead, per-rank GEMMs shrink
and collective time rises. When is tensor parallelism justified, should
sequence parallelism be enabled, and where does additional TP hurt throughput?

**Question:** When does sharding transformer layer tensors reduce memory or
enable model size, and what communication cost appears?

**Sweep:** TP=1, 2, and 4 on the same Qwen3 model; for TP>1 compare sequence
parallelism disabled/enabled where supported and valid. After the baseline is
correct, compare TP communication overlap disabled/enabled for one representative
case. Use model shapes divisible by every TP degree.

**Measurements:** Per-GPU parameter/activation memory, throughput, GEMM sizes,
all-reduce/all-gather/reduce-scatter time, scaling efficiency, and loss
equivalence.

**Expected result:** TP shards large layers and permits larger models, but
communication and smaller per-rank GEMMs can make excessive TP slower. Sequence
parallelism should reduce duplicated activation memory and changes the
collective pattern; it does not consume another multiplicative GPU dimension.

### EXP-012: Pipeline schedules and bubble size

Tags: `megatron` `pp` `scheduling` `microbatch` `2-gpu` `4-gpu`

**Scenario (exam style):** A pharmaceutical company partitions a deep model
across four GPUs, yet the trace shows some stages idle while others work. The
global batch cannot grow without limit. How should the team choose microbatch
count, 1F1B scheduling, and layer placement to reduce the bubble without
creating excessive activation memory?

**Question:** How do microbatch count and schedule determine pipeline bubbles,
memory, and throughput?

**Sweep:** PP=1, 2, and 4; several microbatch counts; a flush/GPipe-style
schedule and 1F1B where exposed by the pinned NeMo/Megatron release. Add virtual
pipeline stages only as a final variant if the basic result is clear.

**Measurements:** Stage utilization, idle/bubble fraction, activation memory,
point-to-point time, throughput, load balance, and timeline shape.

**Expected result:** More microbatches amortize the pipeline bubble but alter
memory and batch geometry. Imbalanced layer assignment makes the slowest stage
the throughput limit.

### EXP-013: Context parallelism for long sequences

Tags: `megatron` `cp` `long-context` `attention` `1-gpu` `2-gpu` `4-gpu`

**Scenario (exam style):** A document-intelligence company increases context
length from 4K to 32K tokens. Parameters still fit, but attention activations
OOM even with a small microbatch. Should it use recomputation, tensor
parallelism, or context parallelism, and at what sequence length does CP's
communication become worthwhile?

**Question:** At what sequence lengths does context parallelism's activation
memory reduction justify its attention communication?

**Sweep:** CP=1, 2, and 4 over increasing sequence lengths, with constant model
and documented global batch/token conventions. Include a direct comparison with
activation checkpointing near the single-GPU memory boundary.

**Measurements:** Peak activation memory, maximum sequence that fits,
tokens/second, attention communication, recomputation time, and numerical
agreement.

**Expected result:** CP reduces per-GPU activation pressure and enables longer
contexts. For short contexts its communication/setup cost may lose to CP=1;
the crossover is the important result.

### EXP-014: TP=2 x DP=2 for model width and throughput

Tags: `megatron` `hybrid` `tp` `dp` `4-gpu`

**Scenario (exam style):** A company has four tightly connected A100s and a
medium-size LLM. DP=4 gives high replica throughput but high per-GPU state
memory; TP=4 reduces layer memory but spends more time in collectives. Would two
TP ranks replicated across two DP groups provide a better balance for the
company's fixed global batch?

**Question:** When is a hybrid layout preferable to DP=4 or TP=4 on the same
four GPUs?

**Sweep:** DP=4, TP=4, and TP=2 x DP=2 for the same model, precision, effective
global batch, and measured tokens.

**Measurements:** Memory, throughput, scaling efficiency, collective timeline,
process-group/rank map, and loss agreement.

**Expected result:** No layout wins universally. The hybrid should expose the
trade-off between replica throughput and model/activation sharding and is the
smallest experiment that tests interacting parallel process groups.

**Process-group interpretation:** TP groups `[0,1]` and `[2,3]` each execute one
model replica, while DP groups `[0,2]` and `[1,3]` synchronize corresponding TP
shards. The two replicas consume different samples. TP reduces per-GPU layer
state; DP increases aggregate batch throughput but does not partition a sample's
context.

### EXP-015: TP=2 x CP=2 for model width and long context

Tags: `megatron` `hybrid` `tp` `cp` `long-context` `4-gpu`

**Scenario (exam style):** A contract-analysis company must fit a wide model and
a long context on four GPUs. TP=4 solves the weight problem but makes each GEMM
small; CP=4 solves activation pressure but duplicates all weights. Would
TP=2 x CP=2 better match the two independent constraints?

**Question:** For a long-context model, is it better to spend four GPUs entirely
on TP or to divide them between tensor and context parallelism?

**Sweep:** TP=4, CP=4 where valid, and TP=2 x CP=2 at one short and one
memory-demanding sequence length.

**Measurements:** Weight and activation memory, communication by process group,
attention versus linear-layer time, throughput, and maximum sequence length.

**Process-group interpretation:** TP groups `[0,1]` and `[2,3]` shard layer
weights. CP groups `[0,2]` and `[1,3]` split each sequence and exchange attention
context; they are not independent data replicas. TP reduces layer state, while
CP reduces per-GPU sequence activation memory. With DP=1, CP does not multiply
the global batch as DP does in EXP-014.

**Why optional:** It is a strong conceptual hybrid exercise but overlaps the
separate TP and CP experiments and needs careful batch/sequence normalization.

### EXP-016: Distributed checkpoint and restart

Tags: `checkpointing` `recovery` `sharding` `ddp` `fsdp` `megatron`

**Scenario (exam style):** A team uses interruptible cloud capacity and loses a
four-GPU training Pod after several hours. The model reloads, but its next loss
differs because optimizer state, RNG state, or data position was not restored.
What must a distributed checkpoint preserve, and which world-size changes are
actually supported?

**Question:** Can replicated and sharded jobs resume without changing the next
loss/update, and which checkpoint forms depend on world size or layout?

**Sweep:** Save and resume DDP, FSDP, and one NeMo/Megatron model-parallel job.
If supported, test loading with one compatible changed data-parallel degree;
do not promise arbitrary TP/PP reshaping.

**Measurements:** Save/load time, checkpoint size and file count, CPU/GPU memory
during save, restored optimizer/RNG/data position, and post-resume loss/update
agreement.

**Why core:** Sharded-state correctness is operationally important and reveals
whether the experiment can actually be reproduced after interruption.

### EXP-017: Controlled troubleshooting and failure diagnosis

Tags: `troubleshooting` `nccl` `oom` `numerics` `straggler` `2-gpu`

**Scenario (exam style):** A two-GPU job alternates between hanging in a
collective, OOMing during backward, and producing NaNs after enabling FP16. A
single "training failed" alert does not identify the cause. Which logs,
timeouts, memory evidence, and profiler patterns distinguish the three failure
classes and prove the corrective action worked?

**Question:** Can common distributed failures be recognized from their symptoms
and diagnosed with the correct PyTorch, NCCL, and NVIDIA evidence?

**Fault cases:** One OOM/fragmentation case, one mismatched collective or rank
configuration caught with a short timeout, one artificial straggler, and one
numerical overflow/non-finite-gradient case. Faults must be bounded so a Pod is
not left hanging or consuming money unnoticed.

**Measurements:** Error/log signature, timeline symptom, utilization pattern,
debug variables used, root cause, corrective action, and proof of recovery.

**Expected result:** Each failure produces a distinct evidence pattern; the
report becomes a practical diagnostic playbook rather than merely a collection
of successful runs.

### EXP-018: End-to-end configuration-selection capstone

Tags: `capstone` `cost` `comparison` `ddp` `fsdp` `tp` `pp` `hybrid`

**Scenario (exam style):** A CTO asks for the cheapest way to process a fixed
number of training tokens on one to four A100s while respecting a memory limit
and delivery deadline. Teams advocate DDP, FSDP, TP, and PP using results from
different workloads. How should the options be normalized, and which decision
rule selects a configuration without claiming one strategy is universally
best?

**Question:** Given a fixed model, sequence length, effective global batch, and
one-to-four-GPU budget, which configuration best satisfies a stated objective?

**Candidate configurations:** Single GPU, DDP, FSDP, TP, PP, and TP=2 x DP=2.
Only configurations already validated in earlier experiments are eligible.

**Objectives:** Evaluate at least maximum throughput under a memory limit and
minimum memory under a throughput floor. Include estimated cost per fixed token
count and energy per token where power telemetry is available, rather than
comparing tokens/second alone.

**Measurements:** Correctness, memory, tokens/second, scaling efficiency, MFU
where valid, communication fraction, implementation complexity, and projected
GPU-hours/cost and energy for a fixed workload.

**Expected result:** The best parallel strategy depends on the binding
constraint. This report should state a decision rule, not declare one framework
or parallelism strategy universally fastest.

## Scope boundary

Project-wide exclusions are maintained only in
[PROJECT_DECISIONS.md](PROJECT_DECISIONS.md#accepted-exclusions) and are not
duplicated here. A candidate that conflicts with those decisions is omitted
from the numbered catalog.

## Proposed implementation order

Dependencies matter more than experiment numbering:

1. **Preflight:** implement the mandatory qualification script.
2. **GPU execution fundamentals:** EXP-001 through EXP-005, and optionally
   EXP-006.
3. **Communication baseline:** EXP-007 and EXP-008.
4. **Replicated and sharded data parallelism:** EXP-009 and EXP-010.
5. **Model-parallel dimensions:** EXP-011 through EXP-013.
6. **Hybrid layouts:** EXP-014 and optionally EXP-015.
7. **Operational correctness:** EXP-016 and EXP-017.
8. **Synthesis:** EXP-018.

An experiment should not be implemented merely because it is next in the list.
Its prerequisite correctness and measurement tools must already be validated.

## Rental-session strategy

To reduce idle cloud cost, group final runs after local implementation and
non-GPU tests pass:

| Session | Pod size | Candidate work |
| --- | ---: | --- |
| A | 1 x A100 SXM | EXP-001 through EXP-005 and one-GPU parts of later experiments |
| B | 2 x A100 SXM | Two-GPU communication, DDP/FSDP, TP/PP/CP, checkpoint, and troubleshooting points |
| C | 4 x A100 SXM | Four-GPU scaling, hybrid, and model-parallel points |

Each session should begin with the shared qualification script and a short smoke
test, and end only after raw results and profiler artifacts are copied to
persistent storage.

## Selection questions

The following decisions should be made before implementation begins:

1. Accept the proposed shared workload contract, including Qwen3-1.7B-Base,
   WikiText-103, full-parameter continued pretraining, and correctness-only
   evaluation?
2. Accept the proposed core set as-is, or set an overall GPU-hour/budget cap?
3. Is a long-context hybrid worth EXP-015 after separate TP and CP results?
4. Should input-pipeline work (EXP-006) be retained or deferred until all core
   distributed experiments are complete?

## Primary sources

- [NVIDIA NCP-GENL certification and exam blueprint](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Megatron Core parallelism strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron Core context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [PyTorch FSDP2 documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [NVIDIA A100 Tensor Core precisions](https://www.nvidia.com/en-eu/data-center/tensorcore/)
- [PyTorch SDPA backend selection](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
- [NVIDIA FlashAttention and Transformer Engine guidance](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/attention_optimizations.html)
- [Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [NVIDIA Megatron Bridge Qwen recipes](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/qwen/qwen.html)
- [WikiText-103 dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
