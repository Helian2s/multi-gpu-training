# EXP-11: Tensor plus sequence parallelism

Design status: Accepted

Catalog entry: [EXPERIMENT_CATALOG.md](../../EXPERIMENT_CATALOG.md)

## Scenario

A team enables TP=2 on a transformer block that already fits on one GPU and
expects speedup from using a second GPU.

## Question

When does tensor parallelism reduce memory or enable larger layers, and what
communication cost appears when sequence parallelism is added?

## Hypothesis

TP=2 will reduce per-rank MLP shard state but add an all-reduce path. Sequence
parallelism will reduce local sequence activation pressure while changing the
per-rank token geometry.

## Rationale

Tensor parallelism shards large layer dimensions across ranks. Sequence
parallelism avoids duplicating some sequence activations across the TP group,
which is useful when activation memory rather than parameter memory is binding.

## Variables

- Independent variables: TP degree and sequence-parallel setting.
- Fixed variables: A100 SXM Pod, image digest, synthetic model shape, precision,
  optimizer, measured step count, and input shape.
- Dependent metrics: step time, tokens/second, peak memory, and estimated
  collective bytes.
- Baseline: TP=1 on one visible GPU.

## Correctness gates

Each selected variant must launch with the declared world size, finish all
measured steps on every rank, and write rank metrics plus the run manifest.

## Decision rule

The hypothesis is confirmed when TP=2 reduces per-rank shard memory but reports
non-zero collective traffic, and sequence parallelism reduces local sequence
activation footprint relative to TP=2 without sequence parallelism.

## Invalidating conditions

Changing GPU type, image digest, visible-device mask, shape, precision, or
measured step window invalidates direct comparison.
