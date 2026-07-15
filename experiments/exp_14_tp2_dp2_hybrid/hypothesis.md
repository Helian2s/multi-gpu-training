# EXP-14: TP=2 x DP=2 for model width and throughput

Design status: Accepted

Catalog entry: [EXPERIMENT_CATALOG.md](../../EXPERIMENT_CATALOG.md)

## Scenario

A team has four peer-accessible A100 SXM GPUs and must choose between
replicating the model, sharding the model, or combining the two.

## Question

When is a TP=2 x DP=2 hybrid preferable to DP=4 or TP=4 on the same four GPUs?

## Hypothesis

The hybrid layout will expose both TP and DP communication paths and should sit
between DP=4 and TP=4 in the memory/throughput trade-off, rather than being
universally best.

## Rationale

DP=4 maximizes replicas but duplicates model state. TP=4 shards model
dimensions but uses a larger TP group. TP=2 x DP=2 is the smallest layout where
both process-group types are non-trivial.

## Variables

- Independent variables: process-group layout.
- Fixed variables: four A100 SXM GPUs, one image digest, synthetic model shape,
  precision, optimizer, and measured step count.
- Dependent metrics: step time, tokens/second, peak memory, rank maps, and
  estimated collective bytes.
- Baseline: DP=4.

## Correctness gates

Each variant must run with world size 4, record the expected TP and DP groups,
complete all ranks, and write rank metrics plus the run manifest.

## Decision rule

The hypothesis is confirmed when the hybrid records TP groups `[0,1]` and
`[2,3]`, DP groups `[0,2]` and `[1,3]`, and produces memory/throughput behavior
between pure DP and pure TP for the chosen shape.

## Invalidating conditions

Changing GPU type, image digest, visible-device set, shape, precision, or
measured step window invalidates direct comparison.
