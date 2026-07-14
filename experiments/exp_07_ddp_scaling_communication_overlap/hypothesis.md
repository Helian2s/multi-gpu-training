# EXP-07: DDP scaling and communication overlap

Design status: Draft

Catalog entry: [EXPERIMENT_CATALOG.md](../../EXPERIMENT_CATALOG.md)

## Scenario

A two-GPU PyTorch job does not approach a 2x speedup. The team needs to
distinguish between insufficient per-rank computation, expensive all-reduce,
poor overlap, bucket behavior, and synchronization frequency before changing
the model or requesting larger hardware.

## Question

When does replicated DDP scale well from one visible GPU to two visible GPUs on
one `AWS-A2` host, and when do gradient synchronization and small local batches
dominate the step?

## Hypothesis

The two-rank DDP run will scale better when each rank has enough local work to
hide all-reduce. Fixed-global-batch scaling should lose efficiency because work
per rank is smaller. Local accumulation with `no_sync` should reduce
synchronization frequency when gradients are synchronized before each optimizer
step.

## Rationale

DDP overlaps gradient all-reduce with the backward pass, but that overlap
depends on bucket readiness, local compute duration, model structure, and
communication bandwidth. `EXP-01` supplies the communication baseline; this
experiment tests whether actual training steps have enough computation to make
that communication less visible.

## Variables

- Independent variables: visible GPU count, per-rank versus effective global
  batch view, DDP bucket size, and local accumulation strategy.
- Fixed variables: one AWS `g7e.12xlarge`, pinned inputs, restored checkpoint
  per variant, fixed precision candidate, fixed sequence length, and one
  immutable PyTorch image.
- Dependent metrics: tokens per second, step time, speedup, scaling efficiency,
  all-reduce time, overlap percentage, peak memory, bucket readiness, and loss
  agreement.
- Baseline: `EXP-07-A2V1` one-rank training baseline on the billed AWS-A2 host.

## Correctness gates

- AWS-A2 qualification and input manifest verification pass.
- DDP world size exactly matches visible GPU count.
- All ranks complete the same number of optimizer steps.
- Loss is finite on every rank.
- Comparable runs use the same model, inputs, optimizer settings, and seeds.
- `no_sync` variants synchronize gradients before every optimizer step.

## Decision rule

Confirm the hypothesis when two-rank efficiency tracks the amount of local work
per rank and profiler evidence shows whether all-reduce overlaps backward work
or extends the critical path.

Reject it when scaling behavior cannot be explained by local work,
communication timing, bucket readiness, or correctness-preserving accumulation.

Leave it inconclusive when profiling timelines, rank logs, loss checks, input
identity, or image identity are incomplete.

## Invalidating conditions

- The run is not on one `AWS-A2` host.
- A two-rank run spans hosts or uses a different GPU topology.
- DDP rank count, visible device count, or optimizer step count differs between
  compared variants.
- The chosen precision candidate is later invalidated by EXP-02 and the run is
  not repeated with an accepted precision.
- Artifacts are not durably staged out.
