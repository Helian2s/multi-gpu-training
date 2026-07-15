# Proposed experiment catalog

Document status: Planning worksheet; numbered experiments use per-row lifecycle status
Last updated: 2026-07-15
Governing decisions: [PROJECT_DECISIONS.md](PROJECT_DECISIONS.md)

## Document role and authority

This file owns experiment IDs, definitions, lifecycle status, hypotheses,
measurements, estimated GPU-hours, and implementation order. It may
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

The numbered rows begin with `Status=proposed`. Change an experiment to
`accepted` or `deferred` during review; use `completed` only after its report is
finished. Only accepted experiments receive an implementation directory.
Canonical IDs `EXP-01` through `EXP-14` define the learning order. Execution is
organized by the active queues `AWS-A1-PyTorch`, `AWS-A2-PyTorch`,
`AWS-A2-Megatron`, `RUNPOD-A2-Megatron`, and `RUNPOD-A4-Megatron`; Runpod preparation can proceed while AWS capacity is
unavailable, and AWS remains available for later retry.

## Proposed shared workload contract

Proposed rows must be resolved before a model-dependent experiment is accepted.
The model-free EXP-01 and EXP-10 communication experiments do not depend on this
contract. The accepted input contract uses one exact dense model and one
existing dataset for all comparable training experiments. Changing either is
allowed only when the experiment's hypothesis makes it unavoidable.

| Decision area | Choice | Status |
| --- | --- | --- |
| Dense workload model | Use `Qwen/Qwen3-1.7B-Base` at the revision pinned in `configs/inputs.lock.yaml` for every model-dependent experiment | accepted |
| Comparable initialization | Restore every comparable variant from the same pinned pretrained checkpoint, batches, and seeds | accepted |
| Training task and objective | Full-parameter continued pretraining with autoregressive next-token cross-entropy; exclude SFT, LoRA, and training from scratch | accepted |
| Benchmark purpose and output | Exercise the complete training path for short infrastructure measurements, not convergence; discard resulting weights after metrics are collected | accepted |
| Dataset, tokenizer, and preparation | Use pinned `Salesforce/wikitext` `wikitext-103-raw-v1`, the pinned model tokenizer, and deterministic canonical token streams without custom cleaning or curation | accepted |
| Revision and sample identity | Pin model, tokenizer, dataset, and preprocessing revisions and preserve sample hashes before comparable runs | accepted |
| Default benchmark precision | Use BF16 except when precision is the independent variable in EXP-02 or a compatibility requirement dictates otherwise | proposed |
| Evaluation boundary | Use correctness gates and short loss/perplexity sanity checks; exclude downstream model-quality suites and training-to-convergence claims | proposed |

Rows marked `accepted` summarize workload rules already governed by
[PROJECT_DECISIONS.md](PROJECT_DECISIONS.md); they are repeated here only to
make the shared workload readable. Rows marked `proposed` remain proposals and
do not become project decisions until they are promoted to that file.

### Why this model

`Qwen/Qwen3-1.7B-Base` was selected because it is:

- A decoder-only causal language model suitable for the selected training
  objective.
- Small enough for full training steps on one qualified 48-96 GB primary GPU,
  making one-GPU baselines possible.
- Large enough to produce meaningful compute, memory, and communication traces
  on two visible GPUs and the four-rank Runpod hybrid.
- Structurally suitable for the proposed parallel degrees: 28 transformer
  layers, 16 query heads, and 8 key/value heads permit the principal TP and PP
  divisions used in this catalog.
- Available under Apache-2.0 without a gated-license workflow.
- Supported by NVIDIA Megatron Bridge through an existing Qwen3 1.7B pretraining
  recipe, avoiding a new NeMo/Megatron model implementation.

The accepted workload uses the base model, not the instruction/chat-tuned model.
The pretrained checkpoint is restored before each comparable variant so that
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
| Docker/OCI, Amazon ECR, GHCR, AWS, Runpod, Git, SSH, and storage tooling | Model-agnostic | Verify both image pulls, provider metadata, artifact paths, staging, and available disk space |
| Hugging Face model, tokenizer, and Safetensors | Officially supported; Qwen3 requires Transformers 4.51 or newer | Pin the exact model revision and a compatible Transformers version |
| Hugging Face Datasets/Hub and WikiText-103 | Model-agnostic download/preprocessing path | Pin dataset revision and preserve license/attribution metadata |
| Native PyTorch forward/backward and AdamW | Official model implementation is a PyTorch `nn.Module` | Compare a fixed-batch loss and update with the published checkpoint |
| PyTorch DDP and NCCL | Model-agnostic distributed wrappers/collectives | Two-GPU correctness smoke test |
| BF16, FP16, FP32, and TF32 | Compatible on the candidate NVIDIA GPUs; BF16 is the proposed default | Validate numerical tolerances and Tensor Core use in EXP-02 |
| FP8 and Blackwell lower-precision formats | Transformer Engine standard FP8 supports Ada and later, including the G7e GPU's SM 12.0 capability; current MXFP8/NVFP4 training documentation lists SM 10.0/10.3 rather than SM 12.0 | Qualify standard FP8 on the pinned G7e stack; exclude MXFP8/NVFP4 from the current plan and never relabel emulation or fallback as native execution |
| PyTorch activation checkpointing | Generic compatibility | Check Qwen block wrapping and loss agreement |
| PyTorch FSDP2 | Expected generic compatibility, but no Qwen3-specific FSDP2 recipe was found | Validate wrapping, tied/shared parameters, state dict, and resume before EXP-08 |
| PyTorch SDPA/Flash Attention backends | Qwen3 uses GQA; optimized backends are available but backend selection is shape/version dependent | Log the selected kernel/backend and retain a math-backend correctness baseline |
| `torch.compile`/Inductor | Transformers supports compiled training generally; no Qwen3 full-training guarantee was found | Treat compilation as a secondary EXP-05 variant and record graph breaks/fallbacks rather than assuming success |
| PyTorch Profiler and TensorBoard | Model-agnostic | Smoke test trace export |
| Nsight Systems, Nsight Compute, NVTX, DCGM, and `nvidia-smi` | Model-agnostic; low-level counters depend on provider and host permissions | Check permissions in the mandatory pre-run qualification |
| CUDA, cuBLAS/cuBLASLt, cuDNN, NCCL, `nccl-tests`, and CUDA Samples | Model-agnostic; used directly or beneath PyTorch/Transformer Engine | Pin via the NGC image and run topology/collective smoke tests |
| NeMo Framework, Megatron Core, and Megatron Bridge | Official Qwen3 support, Qwen3-1.7B pretraining recipe, and HF/Megatron conversion | Pin the NeMo container and perform one round-trip logit/checkpoint comparison |
| Transformer Engine and optimized/fused attention | Megatron Bridge exposes TE attention backends and Qwen3 maps to Megatron model modules | Smoke test the selected GPU backend and log fallbacks |
| NVIDIA Apex | Not required by Qwen itself; use only if bundled and required by the pinned NVIDIA stack | Do not add a separate Apex dependency without a demonstrated need |
| TP=2 and sequence parallelism | Framework-supported and Qwen dimensions are divisible; the official 1.7B recipe recommends TP=1 for efficiency | Validate TP=2 correctness; treat poor performance as a valid result |
| PP=2 | Framework-supported; 28 layers are divisible by 2 | Validate stage assignment, embedding/loss placement, and balance |
| CP=2 | Framework-supported; use sequence lengths divisible by 2 and no more than the model's 32K context | Validate CP attention backend and fixed-token loss agreement |
| TP=2 x DP=2 | Megatron Bridge supports the combined process groups and the dimensions are structurally valid | Run a rank-map and correctness smoke test before profiling |
| Distributed checkpoints | PyTorch and Megatron Bridge support their respective formats | Do not assume arbitrary TP/PP reshaping; test only the layout changes explicitly supported |
| WikiText token stream | Model-independent once tokenized with the pinned Qwen tokenizer | Generate once and verify identical sample hashes in PyTorch and Megatron loaders |
| NumPy, pandas, SciPy, Matplotlib, Seaborn, PyYAML, pytest, Ruff, and `psutil` | Operate on configurations, tests, telemetry, or saved metrics; model-agnostic | Pin versions and validate the analysis pipeline locally |

The verdict is therefore **suitable with targeted smoke tests**, not
"automatically supported by every possible combination." The workload/image
combination is not considered compatible until a job passes the required
actions above on a qualified NVIDIA compute host.

### One model versus multiple models

The accepted default is **one exact dense model**, not a collection of convenient
models. This keeps tokens/second, memory, numerical behavior, and loss
comparisons interpretable across DDP, FSDP, TP, PP, CP, and hybrid layouts.

No other experiment may silently change model size, layer count, vocabulary,
attention type, or checkpoint. If the primary model proves technically
incompatible with an accepted parallel layout, that is a project-level decision
to revisit the model—not permission to substitute a model inside one result.

### Dataset and tokenization

The selected dataset is the existing `Salesforce/wikitext` dataset with the
`wikitext-103-raw-v1` configuration. It contains train, validation, and test
splits and is small enough to download and preprocess once without making data
engineering a project of its own.

The preparation pipeline is deliberately minimal:

1. Pin the dataset revision and download the existing splits.
2. Tokenize with the pinned Qwen3 tokenizer.
3. Insert the model's EOS token between documents.
4. Concatenate each source split into a deterministic canonical token stream.
5. Derive fixed-length samples as deterministic slices through loader adapters
   for native PyTorch and Megatron, without retokenizing.
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

Every end-to-end training experiment uses WikiText-103 and the pinned Qwen
tokenizer. EXP-01 and EXP-10 do not train a model, and isolated
kernel/collective sub-benchmarks may use synthetic tensors.

### Training type and objective

The accepted training task is **full-parameter continued pretraining** from the
selected Qwen3 base checkpoint. Every trainable
parameter participates in forward, backward, gradient synchronization/sharding,
optimizer state, and checkpointing. This exercises the complete
distributed-training path without the cost of pretraining a useful model from
random initialization.

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
fixed in a shared workload configuration. BF16 is the proposed portable default
precision; EXP-02 is the explicit hardware-capability sweep.

### Expected effect of continued pretraining

If continued for enough well-tuned steps, WikiText training would move the base
model's probability distribution toward English Wikipedia-style prose and the
topics present in WikiText. Validation loss/perplexity on WikiText might improve,
but that does not establish that the model became a useful subject-matter
expert. The corpus is small relative to Qwen's original pretraining, and our
benchmark windows are intentionally short.

For this project, the updated model is not the product. Each comparison starts
again from the identical base checkpoint, and resulting weights are discarded
after metrics are collected. We care that the loss and updates are valid and
equivalent across configurations—not that the short run improves general
knowledge or downstream answers.

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

These are measurement windows, not attempts to train the model to convergence.
An experiment may shorten a profile when profiler overhead is extreme, but must
justify the change and still collect enough iterations for a stable conclusion.

### Proposed evaluation policy

The proposed boundary remains infrastructure-focused, but it retains enough
evaluation to show that a faster run did not compute a materially different
update.

If accepted, required evaluation is limited to:

- Finite loss and gradients.
- Initial loss/logit agreement for implementations expected to be equivalent.
- Loss, selected gradient, and parameter-update comparison over the deterministic
  correctness profile, using precision-appropriate tolerances.
- Training loss during the measured run.
- Validation cross-entropy and perplexity on a small fixed WikiText validation
  slice before and after experiments that perform optimizer updates.

Not required:

- Training to convergence.
- MMLU, HellaSwag, HumanEval, or another downstream benchmark suite.
- Human evaluation, generation-quality scoring, or comparison with published
  Qwen quality numbers.
- A claim that the short continued-pretraining run improves the model.

Thus evaluation acts as a **correctness guardrail**, not as a separate model
evaluation research program.

### Supporting libraries

The accepted input workload adds the following narrowly scoped libraries to the
approved toolset:

- `transformers` and `safetensors` for the Qwen model definition/checkpoint in
  native PyTorch experiments.
- `datasets` and `huggingface_hub` for downloading the pinned public dataset,
  tokenizer, and model artifacts.
- Megatron Bridge, supplied by the selected NeMo container, for the NVIDIA
  implementation and one-time checkpoint conversion.

These libraries do not authorize Hugging Face Trainer, Accelerate, or another
distributed-training framework.

## Proposed experiment summary

### Planned compute profiles

These are accepted placements, not evidence that capacity is available. Every
session must still pass the mandatory qualification and cost gate.

AWS profile names are project aliases, not AWS product names. `AWS-A1`,
`AWS-A2`, and `AWS-A4` mean AWS G7e profiles with one, two, and four physical
GPUs respectively. The digit records the billed physical GPU count; a run-unit
suffix such as `A2V1` or `A2V2` records how many GPUs are visible inside a
two-GPU `AWS-A2` host.

Execution queue names are operational provider/framework labels, not EC2-style
instance types. `AWS-A2-Megatron`, `RUNPOD-A2-Megatron`, and
`RUNPOD-A4-Megatron` describe the selected provider, visible-GPU count, and
framework path. `RUNPOD-A2-Megatron` deliberately includes the two-GPU Runpod
communication baseline plus both one-visible-GPU and two-visible-GPU Megatron
phases on the same two-GPU A100 SXM resource profile. The exact billed
resource, GPU count, and GPU model remain recorded in the resource profile and
every run artifact.

| Profile | Provider resource | Physical GPUs | Normal visible GPUs | Purpose |
| --- | --- | ---: | ---: | --- |
| `AWS-A1` | AWS `us-west-2`, On-Demand `g7e.2xlarge` | 1 x RTX PRO 6000 Blackwell Server Edition 96 GB | 1 | One-GPU correctness, kernel, memory, and profiling runs that are not batched into AWS-A2 |
| `AWS-A2` | AWS `us-west-2`, On-Demand `g7e.12xlarge` | 2 x RTX PRO 6000 Blackwell Server Edition 96 GB | 1 or 2 by visibility mask | Current AWS-A2 PyTorch queue plus the AWS-A2-Megatron EXP-12 pipeline-schedule exception |
| `AWS-A4` | AWS `us-west-2`, On-Demand `g7e.24xlarge` | 4 x RTX PRO 6000 Blackwell Server Edition 96 GB | 4 | Not in the current AWS queue; consumes all 96 approved vCPUs and requires a new decision before use |
| `RUNPOD-A100-SXM2` | Runpod Secure Cloud Pod, 2 x `NVIDIA A100-SXM4-80GB` | 2 x A100 80 GB SXM | 1 or 2 by visibility mask | EXP-10 NVLink/NCCL measurements plus EXP-11 and EXP-13 one-/two-rank NeMo/Megatron work |
| `RUNPOD-A100-SXM4` | Runpod Secure Cloud Pod, 4 x `NVIDIA A100-SXM4-80GB` | 4 x A100 80 GB SXM | 4 | The single four-rank hybrid TP=2 x DP=2 experiment |

Runpod does not expose an EC2-style standardized instance type. The cloud class,
exact GPU type ID, GPU count, datacenter, Pod ID, and observed topology together
identify the resource. A Runpod profile is admissible only when qualification
shows the intended GPUs on one physical host and NVLink between every selected
pair. NVSwitch is recorded only if `nvidia-smi topo -m` proves it.

### Queue map by compute profile

The table below is the static launch map. Queues identify the framework/image
path and visible-GPU phase; compute profiles identify the billed resource.
Dynamic workstation, registry, credential, quota, and capacity readiness is
tracked in [infra/TOOLING.md](infra/TOOLING.md).

| Queue | Compute profile | Image family and registry side | Experiment run units | Current launch posture |
| --- | --- | --- | --- | --- |
| `AWS-A1-PyTorch` | `AWS-A1` | PyTorch image from ECR | `QUAL-A1`; `EXP-03-A1`, `EXP-04-A1`, `EXP-05-A1`, `EXP-06-A1` | Raw AWS measurements collected under run `aws-a1-pytorch-20260715T003637Z`; reports and completed status are pending validation |
| `AWS-A2-PyTorch` | `AWS-A2` | PyTorch image from ECR | `QUAL-A2`; `EXP-01-A2`; `EXP-02-A2V1/A2V2`; `EXP-07-A2V1/A2V2`; `EXP-08-A2V2`; `EXP-09-A2V1/A2V2` | Raw AWS measurements collected under run `aws-a2-full-fp16fix-20260715T023252Z`; reports and completed status are pending validation |
| `AWS-A2-Megatron` | `AWS-A2` | NeMo/Megatron image from ECR | `QUAL-A2`; `EXP-12-A2V1`, `EXP-12-A2V2` | Completed EXP-12 on run `aws-a2-megatron-exp12-20260715T0340Z`; qualification ran inside the NeMo image |
| `RUNPOD-A2-Megatron` | `RUNPOD-A100-SXM2` | NeMo/Megatron SSH-start image from GHCR digest `sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b` | `QUAL-RUNPOD-A2-Megatron`; `EXP-10-RUNPOD-A2-Megatron`; EXP-11 and EXP-13 V1/V2 Megatron units | Raw Runpod measurements collected under run `runpod-a2-megatron-20260715T203057Z`; reports and completed status are pending validation |
| `RUNPOD-A4-Megatron` | `RUNPOD-A100-SXM4` | NeMo/Megatron SSH-start image from GHCR digest `sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b` | `QUAL-RUNPOD-A4-Megatron`; `EXP-14-RUNPOD-A4-Megatron` | Completed EXP-14 on run `runpod-a4-megatron-20260715T212017Z`; artifacts mirrored locally; future reusable image should include the NCCL cleanup hotfix |

There is no current queue on `AWS-A4`; using it requires a new decision before
AWS four-GPU work.

As of 2026-07-15, the required AWS run units for EXP-01 through EXP-09 have
all finished with exit status 0. This is complete **raw execution**, not final
catalog completion: the report files still need analysis, validation, and
written conclusions before the lifecycle status can change to `completed`.

| ID | Experiment | Framework/tools needed | GPUs | Provider and planned compute | Docker image family | All queues and run units | Execution state | Target GPU-hours | Status |
| --- | --- | --- | ---: | --- | --- | --- | --- | ---: | --- |
| EXP-01 | AWS PCIe P2P and NCCL communication | CUDA/NCCL/NVIDIA tools, no training framework | 2 | AWS `AWS-A2-PyTorch` queue on `AWS-A2` | PyTorch image, AWS ECR reference `multi-gpu-training-pytorch` | `AWS-A2-PyTorch`: `EXP-01-A2` | Raw execution complete: run `aws-a2-full-fp16fix-20260715T023252Z`, required exit status 0; report validation pending | 1.5-3.0 | accepted |
| EXP-02 | Mixed precision and Tensor Cores in distributed training | PyTorch | 1, 2 | AWS `AWS-A2-PyTorch` queue on `AWS-A2` with `V1` and `V2` phases | PyTorch image, AWS ECR reference `multi-gpu-training-pytorch` | `AWS-A2-PyTorch`: `EXP-02-A2V1`, `EXP-02-A2V2` | Raw execution complete: run `aws-a2-full-fp16fix-20260715T023252Z`, both required exit statuses 0; FP16 AMP parameter fix applied; report validation pending | 1.0-2.0 | accepted |
| EXP-03 | Microbatch, global batch, and gradient accumulation | PyTorch | 1 | AWS `AWS-A1-PyTorch` queue on `AWS-A1` | PyTorch image, AWS ECR reference `multi-gpu-training-pytorch` | `AWS-A1-PyTorch`: `EXP-03-A1` | Raw execution complete: run `aws-a1-pytorch-20260715T003637Z`, required exit status 0; report validation pending | 0.75-1.5 | accepted |
| EXP-04 | Activation checkpointing/recomputation | PyTorch | 1 | AWS `AWS-A1-PyTorch` queue on `AWS-A1` | PyTorch image, AWS ECR reference `multi-gpu-training-pytorch` | `AWS-A1-PyTorch`: `EXP-04-A1` | Raw execution complete: run `aws-a1-pytorch-20260715T003637Z`, required exit status 0; report validation pending | 0.5-1.0 | accepted |
| EXP-05 | PyTorch SDPA/FlashAttention and operator fusion | PyTorch | 1 | AWS `AWS-A1-PyTorch` queue on `AWS-A1` | PyTorch image, AWS ECR reference `multi-gpu-training-pytorch` | `AWS-A1-PyTorch`: `EXP-05-A1` | Raw execution complete: run `aws-a1-pytorch-20260715T003637Z`, required exit status 0; report validation pending | 0.75-1.5 | accepted |
| EXP-06 | Profiler triangulation | PyTorch and NVIDIA profiler tools | 1 | AWS `AWS-A1-PyTorch` queue on `AWS-A1` | PyTorch image, AWS ECR reference `multi-gpu-training-pytorch` | `AWS-A1-PyTorch`: `EXP-06-A1` | Raw execution complete: run `aws-a1-pytorch-20260715T003637Z`, required exit status 0; report validation pending | 0.75-1.5 | accepted |
| EXP-07 | DDP scaling and communication overlap | PyTorch DDP and NCCL | 1, 2 | AWS `AWS-A2-PyTorch` queue on `AWS-A2` with `V1` and `V2` phases | PyTorch image, AWS ECR reference `multi-gpu-training-pytorch` | `AWS-A2-PyTorch`: `EXP-07-A2V1`, `EXP-07-A2V2` | Raw execution complete: run `aws-a2-full-fp16fix-20260715T023252Z`, both required exit statuses 0; report validation pending | 2.0-4.0 | accepted |
| EXP-08 | FSDP sharding and ZeRO-style memory trade-offs | PyTorch FSDP and DDP | 2 | AWS `AWS-A2-PyTorch` queue on `AWS-A2` | PyTorch image, AWS ECR reference `multi-gpu-training-pytorch` | `AWS-A2-PyTorch`: `EXP-08-A2V2` | Raw execution complete: run `aws-a2-full-fp16fix-20260715T023252Z`, required exit status 0; report validation pending | 1.0-2.0 | accepted |
| EXP-09 | Controlled troubleshooting and failure diagnosis | PyTorch, NCCL, and NVIDIA tools | 1, 2 | AWS `AWS-A2-PyTorch` queue on `AWS-A2` with `V1` and `V2` phases | PyTorch image, AWS ECR reference `multi-gpu-training-pytorch` | `AWS-A2-PyTorch`: `EXP-09-A2V1`, `EXP-09-A2V2` | Raw execution complete: run `aws-a2-full-fp16fix-20260715T023252Z`, both required exit statuses 0; report validation pending | 1.5-3.0 | accepted |
| EXP-10 | Runpod NVLink P2P and NCCL communication | CUDA/NCCL/NVIDIA tools, no training framework | 2 | Runpod `RUNPOD-A2-Megatron` queue on `RUNPOD-A100-SXM2` | NeMo/Megatron image, GHCR digest `sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b` | `RUNPOD-A2-Megatron`: `EXP-10-RUNPOD-A2-Megatron` | Raw execution complete: run `runpod-a2-megatron-20260715T203057Z`, required exit status 0; report validation pending | 1.5-3.0 | accepted |
| EXP-11 | Tensor plus sequence parallelism | NeMo Framework with Megatron Core and Megatron Bridge | 1, 2 | Runpod `RUNPOD-A2-Megatron` queue on `RUNPOD-A100-SXM2` with one- and two-visible-GPU phases | NeMo/Megatron image, GHCR digest `sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b` | `RUNPOD-A2-Megatron`: `EXP-11-RUNPOD-A2-Megatron-V1`, `EXP-11-RUNPOD-A2-Megatron-V2` | Raw execution complete: run `runpod-a2-megatron-20260715T203057Z`, both required exit statuses 0; report validation pending | 2.0-4.0 | accepted |
| EXP-12 | Pipeline schedules and bubble size | NeMo Framework with Megatron Core and Megatron Bridge | 1, 2 | AWS `AWS-A2-Megatron` queue on `AWS-A2` with one- and two-visible-GPU phases | NeMo/Megatron image, AWS ECR digest `sha256:ea7616a570d7e271eff25b4f3c0655a9910024e119171e2ead7569f6714b35fb` | `AWS-A2-Megatron`: `EXP-12-A2V1`, `EXP-12-A2V2` | Completed on AWS run `aws-a2-megatron-exp12-20260715T0340Z`; artifacts mirrored locally | 1.0-2.0 | completed |
| EXP-13 | Context parallelism for long sequences | NeMo Framework with Megatron Core and Megatron Bridge | 1, 2 | Runpod `RUNPOD-A2-Megatron` queue on `RUNPOD-A100-SXM2` with one- and two-visible-GPU phases | NeMo/Megatron image, GHCR digest `sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b` | `RUNPOD-A2-Megatron`: `EXP-13-RUNPOD-A2-Megatron-V1`, `EXP-13-RUNPOD-A2-Megatron-V2` | Raw execution complete: run `runpod-a2-megatron-20260715T203057Z`, both required exit statuses 0; report validation pending | 1.0-2.0 | accepted |
| EXP-14 | TP=2 x DP=2 for model width and throughput | NeMo Framework with Megatron Core and Megatron Bridge | 4 | Runpod `RUNPOD-A4-Megatron` queue on `RUNPOD-A100-SXM4` | NeMo/Megatron image, GHCR digest `sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b` | `RUNPOD-A4-Megatron`: `EXP-14-RUNPOD-A4-Megatron` | Completed on Runpod run `runpod-a4-megatron-20260715T212017Z`; artifacts mirrored locally; report written with the cleanup hotfix anomaly disclosed | 2.0-3.0 | completed |

The 14 core row estimates sum to 17.25-33.5 measured GPU-hours. First-time
debugging and profiler setup can make the billable total materially higher.

These are **active experiment GPU-hours**, not necessarily provider-billed
accelerator hours. Cost uses the complete EC2 instance or Runpod Pod. A
one-visible-GPU phase on `AWS-A2` still bills the complete two-GPU instance
and is not equivalent to `AWS-A1`; reports must record both the physical
profile and visible GPU count.

The AWS phase intentionally keeps G7e as the normal family so precision,
topology, memory, profiler, and communication observations stay within one GPU
generation. Cost control should come from batching required one- and two-visible
GPU phases on acquired `AWS-A2` capacity, deferring non-required follow-up to
a separate decision, and keeping the only current four-GPU run on Runpod
EXP-14, not from silently switching to G6e, G6, G5, or another family. A cheaper
family may be proposed only as an explicit contingency because it changes GPU
architecture, memory size, interconnect behavior, supported precision paths,
and often the exact GPU-count shape.

Each numbered experiment has exactly one provider. A scale experiment may use
sequential instance sizes from that provider, but every sub-run still uses one
physical host and records its own profile. Cross-provider relationships are
comparisons between separately numbered experiments, never additional rows
hidden inside one experiment.

### Four-GPU admission gate

Four GPUs are admitted only where two GPUs cannot test the hypothesis:

- EXP-14 needs four ranks because non-trivial TP and DP groups of size two
  require `2 x 2 = 4` ranks.

No AWS catalog experiment currently has a four-GPU run. The former EXP-07 AWS-A4
DDP point is outside the current plan because the AWS-A2 session answers the
near-term communication-overlap question with less capacity risk. Adding an AWS
four-GPU run requires an explicit hypothesis that cannot be answered with one
or two visible GPUs and a project-decision update.

### Provider queue and run-unit ID system

Canonical experiment IDs remain `EXP-NN` and are the only IDs used for
experiment directories, reports, and lifecycle status. AWS execution planning
uses run-unit IDs to name concrete compute-profile sub-runs without renumbering
the catalog:

- `QUAL-A1` and `QUAL-A2` are shared qualification run units for current AWS
  sessions. They are prerequisites, not experiments.
- `EXP-NN-A1` names a run unit on `AWS-A1`.
- `EXP-NN-A2V1` and `EXP-NN-A2V2` name one- and two-visible-GPU phases on the
  same physical `AWS-A2` profile. `EXP-01-A2` keeps the shorter form because
  it is a two-GPU communication baseline with no one-visible-GPU phase.
- `QUAL-A4` and `EXP-NN-A4` are not current queue IDs. They require a new
  project decision before any AWS four-GPU launch work resumes.
- A run unit appears in launch queues, artifact manifests, and report
  subsections. It does not create a separate experiment directory or change the
  parent experiment's `Status`.
- If a proposed sub-run is not needed to answer the parent experiment's
  hypothesis, it is removed from the current queue instead of being kept as
  standby work.

The queue map by compute profile is the authoritative static list of current
run units. The per-experiment summary table repeats each experiment's queue
membership from the experiment point of view. Runpod queue labels appear in
launch queues, artifact manifests, and report subsections; they do not create
separate experiment directories or replace the exact Pod ID, datacenter, GPU
type, GPU count, topology, visible mask, image family, or image digest recorded
for each run. There is no current `RUNPOD-A1-Megatron` queue; one-visible-GPU
Megatron baselines for EXP-11 and EXP-13 are phases in `RUNPOD-A2-Megatron` so
they stay paired with their two-GPU comparisons on the same two-GPU A100 SXM
resource profile.

## Mandatory pre-run qualification

Environment and topology qualification is a prerequisite check, not an
experiment. A shared script must run at the start of every compute session and
its output must be attached to every experiment executed in that session.

The check captures:

- Provider, Region/datacenter, instance or Pod type, purchase option, host ID,
  exact GPU model/memory, physical and visible GPU counts, UUIDs, visibility
  mask, MIG state, clocks, and power limits.
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
- Provider, instance/Pod type, container digest, software versions, GPU memory,
  visible GPU selection, and topology.
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
microbenchmarks. If the shared workload is accepted, end-to-end training
comparisons use the canonical packed WikiText token stream defined above.

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

### EXP-01: AWS PCIe P2P and NCCL communication

**Planned compute:** AWS `AWS-A2-PyTorch` queue on `AWS-A2` only. Run
`EXP-01-A2` with two visible GPUs as the AWS PCIe communication baseline before
the training experiments in the same queue.

**Educational goal:** Learn how to turn raw GPU topology, peer-access, P2P
bandwidth/latency, and NCCL collective measurements into a concrete explanation
of two-GPU communication behavior on one PCIe server.

**Scenario (exam style):** A two-GPU EC2 training job scales poorly. The team
must determine whether the GPUs have a working GPUDirect P2P path, establish the
path's latency and bandwidth, and then decide whether the job's NCCL messages
are latency-bound or bandwidth-bound before changing training code.

**Question:** Does NCCL behavior agree with the measured G7e PCIe topology and
peer-to-peer transfer characteristics?

**Procedure:** First capture `nvidia-smi topo -m`, peer-access capability, and
CUDA `p2pBandwidthLatencyTest` over representative buffer sizes and both
directions. Then run two-rank `all_reduce`, `reduce_scatter`, `all_gather`,
`broadcast`, and `all_to_all` from small to large messages. Default NCCL
settings are the primary result; tuning is a bounded diagnostic appendix.

**Measurements:** Peer-access matrix, unidirectional/bidirectional P2P
bandwidth and latency, NCCL algorithm and bus bandwidth, collective latency,
NCCL debug output, topology, available PCIe counters, and profiler timeline.

**Expected result:** Small collectives are latency-bound while large messages
approach the qualified PCIe path's bandwidth regime. The causal sequence from
topology to P2P measurement to collective behavior becomes the AWS baseline for
EXP-07 and EXP-08.

### EXP-02: Mixed precision and Tensor Cores in distributed training

**Planned compute:** AWS `AWS-A2-PyTorch` queue on `AWS-A2` only. Run
`EXP-02-A2V1` with one visible GPU for the full precision and Tensor Core
sweep, then `EXP-02-A2V2` with two visible GPUs for a bounded DDP check after
the AWS communication baseline is qualified. `EXP-02-A2V1` is billed as an
AWS-A2 host and must not be reported as an exact one-GPU-instance measurement.

**Educational goal:** Learn how precision modes, Tensor Core eligibility,
numerical stability, and DDP communication interact so a faster precision choice
is backed by both performance and correctness evidence.

**Scenario (exam style):** A financial-services company moves LLM training to a
new NVIDIA GPU generation. FP32 training is stable but expensive, while an FP16
trial produces non-finite gradients and an FP8 trial may silently fall back.
Which precision mode should the team use, and how should it verify native Tensor
Core execution without sacrificing acceptable numerical behavior?

**Question:** How do the selected GPU's supported training precisions affect
Tensor Core utilization, numerical behavior, memory, and one-to-two-GPU DDP
scaling?

**Sweep:** FP32, TF32, BF16, and FP16; add standard FP8 Current or Delayed
Scaling through Transformer Engine only after the pinned G7e image proves a
native SM 12.0 path. Do not add MXFP8 or NVFP4 to the current G7e sweep. Also
vary aligned versus deliberately misaligned matrix dimensions and use automatic
mixed precision/gradient scaling where applicable. Include a GEMM
microbenchmark, one-visible-GPU transformer steps, and the same workload under
two-visible-GPU DDP with constant effective global batch. Record FP8 amax
synchronization and the actual NCCL communication datatype/bytes rather than
assuming FP8 compute automatically makes DDP gradient communication FP8.

**Measurements:** Throughput, kernel selection, Tensor Core activity, memory,
loss/gradient difference from the FP32 reference, FP16 overflow behavior, DDP
scaling efficiency, FP8 scaling-reduction overhead, and NCCL time, datatype,
and bytes.

**Expected result:** Tensor-Core-compatible shapes and reduced precision improve
throughput; BF16 is normally more numerically robust than FP16 because of its
wider exponent range. TF32 accelerates eligible FP32 matrix operations while
retaining FP32 storage. Standard FP8 improves performance only when the actual
layer, shape, recipe, and kernel take a supported native path. Faster FP8
computation may increase the fraction of step time spent in unchanged DDP
communication.

**Boundary:** Unsupported formats are recorded as `unsupported` or `fallback`,
not included as measured native-precision variants. Cross-generation numbers
are an appendix unless hardware generation is explicitly the independent
variable.

**Scope note:** This is GPU execution optimization, not model optimization. The
model architecture and parameter count remain unchanged; the experiment studies
how the selected GPU executes the same training computation and how precision
changes the compute-to-communication balance in DDP.

### EXP-03: Microbatch, global batch, and gradient accumulation

**Planned compute:** AWS `AWS-A1-PyTorch` queue on `AWS-A1` only.

**Educational goal:** Learn how microbatch size, accumulation steps, and
effective global batch are related, and how to improve throughput without
accidentally changing the optimization problem.

**Scenario (exam style):** A retailer doubles its training GPU count but keeps
the old microbatch and accumulation settings. Throughput improves, yet the
effective global batch doubles and the loss curve no longer matches the
baseline. Which batch terms must be changed to make the scaling comparison
valid, and which setting maximizes throughput within memory limits?

**Question:** How do microbatch size and accumulation steps trade memory,
utilization, optimizer frequency, and throughput while preserving effective
global batch size?

**Sweep:** Several `(microbatch, accumulation_steps)` pairs with constant global
batch on one GPU. Record the formula
`global_batch = microbatch x accumulation_steps x data_parallel_size` and the
settings needed to keep global batch fixed when data-parallel size changes.
The measured data-parallel scaling evidence comes from EXP-07 instead of a
separate EXP-03 two-GPU sub-run.

**Measurements:** Peak memory, tokens/second, step time per optimizer update,
GPU utilization, number of synchronization operations, loss, and gradient
agreement.

**Expected result:** Larger microbatches generally improve arithmetic intensity
until memory pressure or kernel behavior reverses the gain; accumulation permits
a larger effective batch but does not reproduce every property of one physically
large batch unless loss normalization and synchronization are correct.

### EXP-04: Activation checkpointing/recomputation

**Planned compute:** AWS `AWS-A1-PyTorch` queue on `AWS-A1` only.

**Educational goal:** Learn when recomputing activations is a good memory trade,
how much memory it saves, and how to measure the added compute cost.

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

### EXP-05: PyTorch SDPA/FlashAttention and operator fusion

**Planned compute:** AWS `AWS-A1-PyTorch` queue on `AWS-A1` only.

**Educational goal:** Learn how to distinguish attention-backend improvements
from general graph/operator fusion effects using kernel evidence, memory
measurements, and correctness checks.

**Scenario (exam style):** An AI startup's profiler shows thousands of short
CUDA kernels separated by launch gaps, and eager attention materializes a large
score matrix. The GPU has free compute capacity. Should the team select a fused
attention backend, enable broader operator fusion, or do both—and how can it
verify which kernel actually executed?

**Question:** Which memory and throughput changes come from the attention
backend, and which come from more general operator fusion?

**Sweep:** Use fixed shapes and compare, in order:

1. Eager/manual attention or the PyTorch SDPA math backend as the correctness
   baseline.
2. PyTorch SDPA automatic backend selection, while recording the selected
   backend.
3. PyTorch SDPA with `SDPBackend.FLASH_ATTENTION` forced when Qwen3's GQA shape,
   dtype, and selected GPU support it; record a skip/fallback rather than relabeling
   another kernel as FlashAttention.
4. As a secondary implementation variant, apply `torch.compile` to the
   math/eager path and then to the validated FlashAttention path. This checks
   whether broader graph fusion is complementary without turning compilation
   into a separate curriculum objective.

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

### EXP-06: Profiler triangulation

**Planned compute:** AWS `AWS-A1-PyTorch` queue on `AWS-A1` only.

**Educational goal:** Learn which profiler answers which performance question,
and how to connect framework-level, system-timeline, and kernel-level evidence
into one bottleneck diagnosis.

**Scenario (exam style):** A media company sees only 35% average GPU utilization
during LLM training. One engineer suspects slow Python launches, another
suspects an inefficient CUDA kernel, and a third suspects synchronization.
Which profiler should answer each question, and what evidence distinguishes
these bottlenecks?

**Question:** Can the same bottleneck be identified at framework, system
timeline, and individual-kernel levels?

**Procedure:** Profile one controlled slow variant and one optimized variant
from EXP-05 with PyTorch Profiler, Nsight Systems, and Nsight Compute. Add NVTX
ranges for data loading, forward, backward, communication, and optimizer work.

**Measurements:** Compare operator attribution, CPU/GPU overlap, CUDA launch
gaps, synchronization, memory transfers, kernel occupancy, and memory/compute
limits. Record unavailable hardware counters rather than substituting guesses.

**Why core:** Performance profiling and troubleshooting are explicit NCP-GENL
objectives; this experiment teaches when each profiler is appropriate.

### EXP-07: DDP scaling and communication overlap

**Planned compute:** AWS `AWS-A2-PyTorch` queue on `AWS-A2` only. Run
`EXP-07-A2V1` with one visible GPU and `EXP-07-A2V2` with two visible GPUs on
the same physical AWS-A2 profile. No distributed job spans instances, and
there is no current four-GPU AWS DDP run.

**Educational goal:** Learn how to evaluate DDP speedup, communication overlap,
bucket behavior, and local-batch effects when moving from one rank to two ranks
on one server.

**Scenario (exam style):** A software company expects two GPUs to train nearly
twice as fast as one, but measures a weak speedup. GPU timelines show
all-reduces extending beyond backward computation. Should it increase local
work, change bucket behavior, accumulate gradients locally, or conclude the
model is too communication-heavy for DDP?

**Question:** When does replicated data parallelism scale well within one
server, and when do gradient synchronization and small local batches dominate?

**Sweep:** One and two visible GPUs on `AWS-A2` with fixed per-GPU batch
(weak-scaling view) and fixed global batch (strong-scaling view). Test a small,
justified set of bucket sizes and gradient accumulation with `no_sync`.

**Measurements:** Throughput, speedup, efficiency, peak memory per GPU,
all-reduce time, backward/communication overlap, bucket readiness, and loss
equivalence.

**Expected result:** Scaling is better when each GPU has enough computation to
hide all-reduce. Fixed-global-batch scaling loses efficiency as work per rank
shrinks. Accumulating locally should reduce synchronization frequency when
implemented correctly.

**AWS-A2 boundary:** The current run answers the immediate one-to-two-rank DDP
communication-overlap question while avoiding scarce four-GPU AWS capacity. It
does not claim to measure non-linear degradation from two to four ranks. Adding
that point later requires a new decision and a distinct AWS-A4 queue.

### EXP-08: FSDP sharding and ZeRO-style memory trade-offs

**Planned compute:** AWS `AWS-A2-PyTorch` queue on `AWS-A2` only. Run
`EXP-08-A2V2` after the DDP overlap experiment in the same AWS-A2 completion
queue.

**Educational goal:** Learn what FSDP shards, which extra collectives it adds,
and how to decide whether memory savings are worth the throughput and
complexity cost when the model already fits.

**Scenario (exam style):** A healthcare company can train its model with DDP on
two GPUs, but assumes full sharding must be better because it uses less memory.
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

### EXP-09: Controlled troubleshooting and failure diagnosis

**Planned compute:** AWS `AWS-A2-PyTorch` queue on `AWS-A2` only. Run
`EXP-09-A2V1` with one visible GPU for one-rank failure and input-pipeline
cases, then `EXP-09-A2V2` with two visible GPUs for distributed faults such as
mismatched collectives, rank stragglers, and NCCL timeout evidence.

**Educational goal:** Learn to identify common training and distributed failure
classes from logs, utilization, memory, timeout, numerical, and timeline
evidence, then verify that the corrective action actually fixes the root cause.

**Scenario (exam style):** A two-GPU job alternates between hanging in a
collective, OOMing during backward, producing NaNs after enabling FP16, and
leaving GPUs idle while workers prepare data. A single "training failed or
slow" alert does not identify the cause. Which logs, timeouts, memory evidence,
and profiler patterns distinguish the failure classes and prove the corrective
action worked?

**Question:** Can common training and distributed failures be recognized from
their symptoms and diagnosed with the correct PyTorch, NCCL, and NVIDIA
evidence?

**Fault cases:** One OOM/fragmentation case, one mismatched collective or rank
configuration caught with a short timeout, one distributed artificial straggler,
one numerical overflow/non-finite-gradient case, and one input-starvation case.
For input starvation, compare synthetic pre-generated tokens with the fixed
tokenized dataset while varying DataLoader workers, pinned memory, prefetching,
and persistent workers. Faults must be bounded so a compute host is not left
hanging or consuming money unnoticed.

**Measurements:** Error/log signature, timeline symptom, utilization pattern,
CPU/RAM and data-wait time where applicable, debug variables used, root cause,
corrective action, and proof of recovery.

**Expected result:** Each failure produces a distinct evidence pattern; the
report becomes a practical diagnostic playbook rather than merely a collection
of successful runs. The input case should show GPU idle gaps disappearing when
the pipeline rather than NCCL is corrected.

### EXP-10: Runpod NVLink P2P and NCCL communication

**Planned compute:** Runpod `RUNPOD-A2-Megatron` queue on
`RUNPOD-A100-SXM2` only.

**Educational goal:** Learn how to qualify an A100 SXM/NVLink host and explain
how NVLink topology changes P2P and collective behavior compared with the AWS
PCIe baseline.

**Scenario (exam style):** A team rents an A100 SXM Pod for model-parallel
training, but the product name alone does not prove the selected pair is linked
or that NCCL uses the intended path. It needs one communication qualification
that connects the observed topology to P2P and collective measurements.

**Question:** Does NCCL behavior agree with the measured A100 SXM NVLink
topology and peer-to-peer transfer characteristics?

**Procedure:** Repeat EXP-01's topology, P2P buffer-size, collective-type, and
message-size method on the qualified two-GPU Pod. Record NVLink counters when
accessible. Keep AWS and Runpod results separate and treat GPU architecture and
fabric as co-varying environment differences, not as a provider ranking.

**Measurements:** Peer-access matrix, unidirectional/bidirectional P2P
bandwidth and latency, NCCL algorithm and bus bandwidth, collective latency,
NCCL debug output, topology, NVLink counters, and profiler timeline.

**Expected result:** A qualified NVLink path should show materially different
P2P and collective regimes from the AWS PCIe environment. These measurements
set communication expectations for EXP-11, EXP-13, and EXP-14.

### EXP-11: Tensor plus sequence parallelism

**Planned compute:** Runpod `RUNPOD-A2-Megatron` queue on
`RUNPOD-A100-SXM2` only. Use one and then two visible GPUs on the same billed
two-GPU Pod so the TP=1 baseline and TP=2 run share the exact GPU type, image,
and host environment.

**Educational goal:** Learn which parts of a transformer layer tensor
parallelism shards, how sequence parallelism reduces activation pressure, and
which collectives those choices introduce.

**Scenario (exam style):** An enterprise enables TP=2 on a model that already
fits on one GPU and expects a twofold speedup. Instead, per-rank GEMMs shrink
and collective time rises. When is tensor parallelism justified, should
sequence parallelism be enabled, and why can TP hurt throughput?

**Question:** When does sharding transformer layer tensors reduce memory or
enable model size, and what communication cost appears?

**Sweep:** TP=1 and 2 on the same Qwen3 model; for TP=2 compare sequence
parallelism disabled/enabled where supported and valid. After the baseline is
correct, compare TP communication overlap disabled/enabled for one representative
case. Use model shapes divisible by TP=2.

**Measurements:** Per-GPU parameter/activation memory, throughput, GEMM sizes,
all-reduce/all-gather/reduce-scatter time, scaling efficiency, and loss
equivalence.

**Expected result:** TP shards large layers and permits larger models, but
communication and smaller per-rank GEMMs can make excessive TP slower. Sequence
parallelism should reduce duplicated activation memory and changes the
collective pattern; it does not consume another multiplicative GPU dimension.

### EXP-12: Pipeline schedules and bubble size

**Planned compute:** AWS `AWS-A2-Megatron` queue on `AWS-A2` only. Use one and
then two visible GPUs on the same billed `g7e.12xlarge` host so PP=1 and PP=2
share the image, GPU type, Region, and host environment. This is a PCIe
pipeline-schedule experiment, not a Runpod NVLink measurement.

**Run state:** Completed on AWS run `aws-a2-megatron-exp12-20260715T0340Z`.
`QUAL-A2`, `EXP-12-A2V1`, and `EXP-12-A2V2` all exited `0`; S3 artifacts and
the local ignored mirror contain the raw logs, rank metrics, queue log, and
report inputs.

**Educational goal:** Learn how pipeline stage balance, microbatch count, and
schedule choice determine bubble overhead, activation memory, and throughput.

**Scenario (exam style):** A pharmaceutical company partitions a deep model
across two GPUs, yet the trace shows one stage idle while the other works. The
global batch cannot grow without limit. How should the team choose microbatch
count, 1F1B scheduling, and layer placement to reduce the bubble without
creating excessive activation memory?

**Question:** How do microbatch count and schedule determine pipeline bubbles,
memory, and throughput?

**Sweep:** PP=1 and 2; several microbatch counts; a flush/GPipe-style
schedule and 1F1B where exposed by the pinned NeMo/Megatron release. Add virtual
pipeline stages only as a final variant if the basic result is clear.

**Measurements:** Stage utilization, idle/bubble fraction, activation memory,
point-to-point time, throughput, load balance, and timeline shape.

**Expected result:** More microbatches amortize the pipeline bubble but alter
memory and batch geometry. Imbalanced layer assignment makes the slowest stage
the throughput limit.

### EXP-13: Context parallelism for long sequences

**Planned compute:** Runpod `RUNPOD-A2-Megatron` queue on
`RUNPOD-A100-SXM2` only. Use one and then two visible GPUs on the same billed
Pod.

**Educational goal:** Learn when context parallelism becomes useful for long
sequence training, and how to compare its activation-memory savings against
attention communication cost.

**Scenario (exam style):** A document-intelligence company increases context
length from 4K to 32K tokens. Parameters still fit, but attention activations
OOM even with a small microbatch. Should it use recomputation, tensor
parallelism, or context parallelism, and at what sequence length does CP's
communication become worthwhile?

**Question:** At what sequence lengths does context parallelism's activation
memory reduction justify its attention communication?

**Sweep:** CP=1 and 2 over increasing sequence lengths, with constant model
and documented global batch/token conventions. Include a direct comparison with
activation checkpointing near the single-GPU memory boundary.

**Measurements:** Peak activation memory, maximum sequence that fits,
tokens/second, attention communication, recomputation time, and numerical
agreement.

**Expected result:** CP reduces per-GPU activation pressure and enables longer
contexts. For short contexts its communication/setup cost may lose to CP=1;
the crossover is the important result.

### EXP-14: TP=2 x DP=2 for model width and throughput

**Planned compute:** Runpod `RUNPOD-A4-Megatron` queue on
`RUNPOD-A100-SXM4` only.

**Educational goal:** Learn how tensor-parallel and data-parallel process
groups compose in a four-rank hybrid, and how to reason about memory,
throughput, and communication trade-offs across DP=4, TP=4, and TP=2 x DP=2.

**Scenario (exam style):** A company has four peer-accessible NVIDIA GPUs and a
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

**Why four GPUs:** This is the minimum rank count that gives both TP and DP
non-trivial group sizes: `TP=2 x DP=2 = 4`. Two GPUs can test either dimension,
but not their interaction.

**Process-group interpretation:** TP groups `[0,1]` and `[2,3]` each execute one
model replica, while DP groups `[0,2]` and `[1,3]` synchronize corresponding TP
shards. The two replicas consume different samples. TP reduces per-GPU layer
state; DP increases aggregate batch throughput but does not partition a sample's
context.

## Final curriculum deliverable

The project ends with a configuration-selection report and exam decision
worksheet, not another numbered experiment or dedicated GPU run. It synthesizes
the accepted experiments, links each result to an NCP-GENL concept, and records
the evidence that distinguishes plausible exam answers.

The report also records one deliberate coverage boundary: the official GPU
Acceleration and Optimization domain includes inference, while this repository
implements distributed-training experiments only. Inference batching,
TensorRT-LLM/Triton execution, and serving latency remain study-note topics and
are not silently claimed as hands-on coverage.

**Scenario (exam style):** A CTO asks for the cheapest way to process a fixed
number of training tokens on a two-GPU server while respecting a memory limit
and delivery deadline. Teams advocate DDP, FSDP, and PP using results from
different workloads. How should the options be normalized, and which decision
rule selects a configuration without claiming one strategy is universally
best?

**Question:** Given a fixed model, sequence length, effective global batch, and
two-GPU budget, which configuration best satisfies a stated objective?

**Candidate configurations:** Use only configurations already validated by the
numbered experiments. Keep AWS and Runpod evidence in separate hardware tables;
do not normalize unlike GPUs into a false provider ranking.

**Objectives:** Evaluate at least maximum throughput under a memory limit and
minimum memory under a throughput floor. Include estimated cost per fixed token
count and energy per token where power telemetry is available, rather than
comparing tokens/second alone.

**Measurements:** Correctness, memory, tokens/second, scaling efficiency, MFU
where valid, communication fraction, implementation complexity, and projected
GPU-hours/cost and energy for a fixed workload.

**Expected result:** The best parallel strategy depends on the binding
constraint. The report states decision rules, not one universally fastest
framework or layout, and adds short exam-style questions for every core result.

## Scope boundary

Project-wide exclusions are maintained only in
[PROJECT_DECISIONS.md](PROJECT_DECISIONS.md#accepted-exclusions) and are not
duplicated here. A candidate that conflicts with those decisions is omitted
from the numbered catalog.

## Proposed implementation order

Experiment numbering defines the recommended learning order. Execution now uses
five provider/framework queues so Runpod preparation can proceed while AWS G7e
capacity is unavailable. A later experiment still starts only after its
explicit prerequisites pass:

1. **Preflight:** implement the mandatory qualification script.
2. **AWS communication baseline:** EXP-01.
3. **AWS execution fundamentals and profiling:** EXP-02 through EXP-06.
4. **AWS data parallelism:** EXP-07 and EXP-08.
5. **AWS troubleshooting:** EXP-09.
6. **RUNPOD-A2-Megatron:** Runpod NeMo/Megatron image, storage, registry, and
   EXP-10 communication readiness.
7. **AWS-A2-Megatron:** EXP-12 pipeline schedule and bubble-size exception.
8. **RUNPOD-A2-Megatron:** EXP-11 and EXP-13 one- and two-GPU
   model-parallel dimensions.
9. **RUNPOD-A4-Megatron:** EXP-14 hybrid layout.
10. **Synthesis:** write the final curriculum report and exam decision worksheet.

Compute-profile sub-runs may be batched for cost efficiency, but their reports
retain this logical order. AWS queues remain available for retry when capacity
appears; Runpod work does not overwrite AWS state. An experiment should not
start merely because it is next in the list; prerequisite correctness and
measurement tools must already be validated.

## Compute-session strategy

To reduce idle cloud cost, group final runs only after local implementation and
non-GPU tests pass. Every session begins with shared qualification and a short
smoke test and ends only after artifacts reach durable storage.

Use the queue map by compute profile above as the static launch plan. Within
any paid session, run qualification first, then only the accepted or explicitly
approved run units whose prerequisites are already satisfied. Dynamic provider
readiness, exact image digests, credential state, and recent capacity attempts
belong in [infra/TOOLING.md](infra/TOOLING.md), not in repeated catalog tables.

There is no current AWS-A4 session. `AWS-A4` consumes the complete 96-vCPU
G/VT quota, so reviving a four-GPU AWS DDP point requires a new decision,
updated queue entry, and lifecycle guard verifying that no other G or VT
instance is running in the Region. G6e, G5, G6, or another AWS family is not a
silent fallback: changing the GPU or instance type changes the environment and
requires updating the planned compute profile and expected results.

The intended batching order is:

1. `AWS-A2-PyTorch`: qualification, EXP-01, then accepted EXP-02/07/08/09
   phases when capacity and scan disposition allow.
2. `AWS-A1-PyTorch`: qualification, then EXP-03 through EXP-06 as one
   sequential one-GPU queue on an AWS-A1 host after queue smoke and ECR scan
   review.
3. `AWS-A2-Megatron`: EXP-12 one- and two-visible-GPU pipeline-schedule phases
   after the EXP-12-capable NeMo image, S3 permissions, and dry-run pass.
4. `RUNPOD-A2-Megatron`: provider readiness, NeMo image/storage smoke tests,
   EXP-10 communication, and one-/two-visible-GPU EXP-11 and EXP-13 phases.
5. `RUNPOD-A4-Megatron`: EXP-14 only after the two-GPU Megatron path and local
   TP=2 x DP=2 rank-map check pass.

AWS and Runpod results remain separate because GPU generation, memory
technology, CPU allocation, and interconnect all differ.

## Selection questions

The following decisions should be made before implementation begins:

1. Accept the remaining proposed workload choices: BF16 as the default
   benchmark precision and the correctness-only evaluation boundary?
2. Accept the proposed core set as-is, or set an overall GPU-hour/budget cap?

## Primary sources

- [NVIDIA NCP-GENL certification and exam blueprint](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-professional/)
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [Megatron Core parallelism strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
- [Megatron Core context parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html)
- [PyTorch FSDP2 documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
- [NVIDIA A100 Tensor Core precisions](https://www.nvidia.com/en-eu/data-center/tensorcore/)
- [NVIDIA Transformer Engine supported hardware and precision formats](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)
- [NVIDIA CUDA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus)
- [Transformer Engine FP8 Delayed Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)
- [Transformer Engine MXFP8 supported devices](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)
- [Transformer Engine NVFP4 supported devices](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
- [NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
- [NVIDIA L40S specifications](https://www.nvidia.com/en-us/data-center/l40s/)
- [AWS G7e instance specifications](https://aws.amazon.com/ec2/instance-types/g7e/)
- [AWS G6e instance specifications](https://aws.amazon.com/ec2/instance-types/g6e/)
- [Runpod GPU type IDs](https://docs.runpod.io/references/gpu-types)
- [Runpod A100 SXM specifications](https://www.runpod.io/gpu-models/a100-sxm)
- [PyTorch SDPA backend selection](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
- [NVIDIA FlashAttention and Transformer Engine guidance](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/optimizations/attention_optimizations.html)
- [Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [NVIDIA Megatron Bridge Qwen3 recipes](https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.recipes.qwen.qwen3.html)
- [WikiText-103 dataset card](https://huggingface.co/datasets/Salesforce/wikitext)
