# EXP-13: Context parallelism for long sequences

Design status: Accepted

Catalog entry: [EXPERIMENT_CATALOG.md](../../EXPERIMENT_CATALOG.md)

## Scenario

A team increases context length until attention activations, not parameters,
become the memory bottleneck.

## Question

At what sequence lengths does context parallelism's activation-memory reduction
justify its attention communication?

## Hypothesis

CP=2 will reduce each rank's local sequence length and peak activation pressure
for longer contexts, while adding K/V exchange traffic that can dominate shorter
contexts.

## Rationale

Context parallelism partitions the sequence dimension across ranks. It lowers
per-rank attention activation size, but each rank must still access remote K/V
state to attend over the full context.

## Variables

- Independent variables: context-parallel degree, sequence length, and
  activation checkpointing.
- Fixed variables: A100 SXM Pod, image digest, hidden size, precision, measured
  step count, and optimizer.
- Dependent metrics: peak memory, step time, tokens/second, and estimated K/V
  exchange bytes.
- Baseline: CP=1 at the same sequence length where available.

## Correctness gates

Each variant must use the declared world size, complete all ranks, and write
rank metrics plus the run manifest.

## Decision rule

The hypothesis is confirmed when CP=2 records half-length local sequence shards
and lower or bounded memory at long context while exposing non-zero K/V exchange
traffic. The crossover is inferred from the memory and throughput tables.

## Invalidating conditions

Changing GPU type, image digest, visible-device mask, shape, precision, or
measured step window invalidates direct comparison.
