# Validation status

Source review: **2026-09-10**, against code at `2e5ddc0`.

The project has recorded GPU executions for all 14 experiments. The catalog
marks EXP-12 and EXP-14 completed and leaves the other 12 accepted, with formal
report validation pending. This review records implementation gaps that limit
the conclusions. It does not change those lifecycle entries or approve a new
workload contract.

## Evidence available

- Tracked reports contain July 2026 results, run IDs, environment descriptions,
  and artifact locations.
- The review ran on macOS ARM64. All 41 existing tests and repository validation
  passed using `make check PYTHON=/usr/local/bin/python3.12`.
- Small CPU probes checked sample indexing and the gradient path through the
  context-parallel gather helper. The latter used two local Gloo processes.
- The review checkout had no prepared inputs or raw artifact mirrors. Historical
  GPU metrics were read from tracked reports, not independently recalculated.
  No new CUDA run was performed.

The local suite primarily validates preparation, configuration, and launch
contracts. A passing suite does not establish distributed numerical equivalence
or validate the performance hypotheses.

## Training objective and sample identity

### Labels are shifted twice

In [the PyTorch executor](../common/pytorch_executor.py), `token_batch` returns
input tokens with labels already shifted one position. `train_steps` supplies
those labels to `AutoModelForCausalLM`, whose
[causal LM loss shifts labels internally](https://github.com/huggingface/transformers/blob/main/src/transformers/loss/loss_utils.py).

For a simple token stream, the current composition produces:

```text
Input IDs:                 0  1  2  3
Labels supplied:           1  2  3  4
Targets after model shift: 2  3  4  ignored
```

This changes the intended next-token objective in the training paths used by
EXP-02, EXP-03, EXP-04, EXP-07, and EXP-08. Historical timing and allocation
numbers remain records of the executed code, but its losses do not validate
the intended objective. Correct the alignment, test it with known tokens, and
rerun affected correctness comparisons.

### Accumulated batches overlap across ranks

`train_steps` indexes batches with `step * accumulation + microstep + rank`.
For two ranks and four accumulation steps, a CPU probe observed:

```text
Rank 0 batch indices: 0 1 2 3
Rank 1 batch indices: 1 2 3 4
```

Eight sample slots contain five distinct sequences. EXP-07's accumulation
comparison needs a global sample-index scheme that partitions each optimizer
step across ranks and preserves sample identity across comparable variants.

## Synthetic parallelism workloads

### Context parallelism drops K/V gradients

In [the parallelism executor](../common/megatron_executor.py),
`all_gather_first_dim` gathers into `empty_like` buffers and concatenates them.
The gathered keys and values are disconnected from their originating autograd
graphs. A two-rank CPU reproduction of the attention path found:

| Parameter projection | Gradient present on each rank |
| --- | --- |
| Query | Yes |
| Key | No |
| Value | No |
| Output | Yes |

EXP-13's CP=1 and CP=2 paths therefore perform unequal backward work. Its
memory and timing differences cannot establish a correct CP training tradeoff.
The repair needs gradient-preserving exchange, synchronization of replicated
parameters, and comparison with a single-rank reference.

### TP/SP and hybrid variants need equivalence checks

The TP and hybrid workers seed inputs with the global rank, then sum partial
outputs within tensor-parallel groups. Those partial outputs describe different
samples. Parameters are generated at shard shape rather than partitioned from
one reference model. The SP path shortens local sequences without implementing
the full sequence redistribution needed to preserve that reference computation.

EXP-11 and EXP-14 demonstrate group creation, collectives, and allocation
behavior, but do not establish equivalent TP/SP/DP training. A reference model,
consistent inputs within each TP group, and output/gradient/update comparisons
are needed before interpreting their performance as correct parallel training.

### The implemented workload differs from the accepted plan

EXP-11–14 use synthetic MLP, attention, or pipeline stages. They import and
record Megatron availability but use custom PyTorch distributed operations;
they do not execute a full Qwen NeMo/Megatron recipe. EXP-12's bubble fraction
is a formula-based estimate, not a measured idle-time fraction from a trace.

The reports disclose synthetic workloads. However, the
[accepted project contract](../PROJECT_DECISIONS.md) specifies pinned Qwen
continued pretraining for training experiments. Review the EXP-12/14 completion
records against that contract. Retaining a narrower synthetic scope requires
an explicit project decision; this documentation update does not make one.

## Analysis and profiling gaps

- Twelve experiment `analyze.py` entry points still raise a template error.
  EXP-12 and EXP-14 have table summarizers, but these do not enforce every
  declared correctness gate.
- The executors mark manifests completed after worker execution without
  evaluating `expected_results.yaml`. Recording `finite_loss` is not the same
  as rejecting an invalid run.
- Configured repetition counts are not implemented by the shared execution
  loop. Training timing arrays are reduced to means rather than preserved as
  per-step samples for later variance analysis.
- EXP-06 captured PyTorch Profiler output. Its Nsight variants only check
  executable availability and print a command; that command routes back to the
  availability check rather than an attention measurement. Nsight capture and
  actual cross-profiler analysis remain unfinished.

Consequently, “raw execution complete” means execution artifacts were recorded.
It does not mean all planned measurements or decision rules were satisfied.

## Reproduction and lifecycle gaps

The [Runpod queue generator](../infra/runpod/runpod_queue.py) includes a hard
termination deadline in its Pod creation command. The container script prints
an operator instruction when `stop_on_success=true`; it does not stop the Pod
itself. The checked-in queues use Pod storage and depend on manual artifact
copy-out before deletion. A deadline limits runtime but does not verify durable
artifact preservation.

The [July handoff](../HANDOFF.md) records deletion of AWS ECR images and a
container-side cleanup hotpatch in the successful EXP-14 run. That patch is now
in source but was not in the published Runpod digest. Historical image names
and readiness records are not sufficient to reproduce an unmodified run today.

## Next validation work

1. Fix label alignment and rank/sample partitioning, with deterministic CPU
   regression checks.
2. Validate distributed outputs, gradients, and optimizer updates against an
   equivalent single-rank workload; repair CP and TP/SP/hybrid semantics.
3. Resolve the synthetic-workload scope explicitly and review completion status.
4. Implement analysis gates, missing profiler captures, repetition handling,
   and preservation of timing samples.
5. Restore raw artifacts, build corrected images, and rerun affected comparisons
   when GPU execution is authorized.

Keep historical measurements intact and attach corrected runs with new IDs.
Promote reports only after the required evidence and analysis are validated.
