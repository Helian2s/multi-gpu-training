# EXP-08: FSDP sharding and ZeRO-style memory trade-offs

Design status: Draft

Catalog entry: [EXPERIMENT_CATALOG.md](../../EXPERIMENT_CATALOG.md)

## Scenario

A model fits with two-rank DDP, but the team assumes FSDP must be better because
it shards model state. The experiment must show which memory categories are
reduced, which collectives are added, and whether those trade-offs help when
the unsharded model already fits.

## Question

How do replicated DDP and FSDP sharding trade persistent memory savings for
additional communication, initialization complexity, and checkpoint handling on
two visible GPUs?

## Hypothesis

FSDP full sharding will reduce persistent per-rank model-state memory relative
to DDP, but may be slower for this fixed model because parameter materialization
and extra collectives add overhead. Partial sharding should land between DDP
and full sharding on memory and communication cost.

## Rationale

DDP replicates model states on every rank, while FSDP can shard parameters,
gradients, and optimizer state. That reduces per-rank memory but introduces
all-gather and reduce-scatter work and more complex state-dict behavior. The
right choice depends on whether memory is actually the limiting resource.

## Variables

- Independent variables: distributed strategy and sharded state categories.
- Fixed variables: one AWS `g7e.12xlarge`, two visible GPUs, pinned inputs,
  restored checkpoint per variant, fixed precision candidate, fixed sequence
  length, and one immutable PyTorch image.
- Dependent metrics: model-state memory, peak memory, initialization time,
  all-gather/reduce-scatter traffic, throughput, checkpoint size, state-dict
  round-trip behavior, and loss agreement.
- Baseline: `EXP-08-A2V2` DDP replicated baseline.

## Correctness gates

- AWS-A2 qualification and input manifest verification pass.
- DDP baseline completes first with finite loss on both ranks.
- FSDP API choice and any FSDP1/FSDP2 fallback reason are recorded.
- Each FSDP variant completes forward, backward, optimizer step, and checkpoint
  round trip.
- Compared variants use the same checkpoint, input batches, seeds, and optimizer
  settings.
- No model-size change is used to manufacture an OOM result.

## Decision rule

Confirm the hypothesis when sharding reduces the expected memory categories and
the report can attribute any throughput loss to added collectives,
materialization, initialization, or checkpoint overhead.

Reject it when FSDP neither reduces the expected memory categories nor provides
a coherent communication or initialization explanation.

Leave it inconclusive when memory attribution, distributed logs, checkpoint
round-trip evidence, input identity, or image identity is incomplete.

## Invalidating conditions

- The run is not on one `AWS-A2` host with two visible GPUs.
- FSDP fallback behavior is not recorded.
- A model or batch-size change is introduced only to create an OOM result.
- Compared variants use different checkpoints, inputs, optimizer settings, or
  seeds.
- Artifacts are not durably staged out.
