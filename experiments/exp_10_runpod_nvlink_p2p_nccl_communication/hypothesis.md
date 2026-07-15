# EXP-10: Runpod NVLink P2P and NCCL communication

Design status: Accepted

Catalog entry: [EXPERIMENT_CATALOG.md](../../EXPERIMENT_CATALOG.md)

## Scenario

A team rents a two-GPU A100 SXM Pod for later model-parallel training. The
provider product name is not enough evidence that the selected pair is actually
connected by the expected fabric or that NCCL uses it.

## Question

Does NCCL behavior agree with the observed A100 SXM topology and CUDA P2P
measurements?

## Hypothesis

If the Pod exposes two A100 SXM GPUs linked by NVLink, then CUDA P2P and NCCL
collective bandwidth will show the high-bandwidth regime expected from that
topology rather than a CPU or PCIe-only fallback.

## Rationale

Tensor, context, and hybrid parallel experiments depend on fast peer traffic.
This experiment is the Runpod equivalent of EXP-01 and establishes the
communication baseline for EXP-11, EXP-13, and EXP-14.

## Variables

- Independent variables: collective type and message size.
- Fixed variables: two visible A100 SXM GPUs, one Pod, one immutable image,
  NCCL environment, and one measured run ID.
- Dependent metrics: P2P bandwidth/latency, NCCL algorithm, bus bandwidth, and
  latency.
- Baseline: the observed Runpod default NCCL environment on the qualified Pod.

## Correctness gates

The run is valid only if two A100 SXM devices are visible, topology output shows
the selected pair's fabric, `p2pBandwidthLatencyTest` exits successfully, every
required NCCL test exits successfully, and the immutable GHCR image digest plus
Pod metadata are recorded.

## Decision rule

The hypothesis is confirmed when topology, P2P, and NCCL evidence agree on a
GPU-direct high-bandwidth path. It is rejected when the selected pair lacks
NVLink or NCCL falls back to an unexpected path. Missing counters make only the
counter-specific conclusion inconclusive.

## Invalidating conditions

Changing GPU type, datacenter, image digest, visible device pair, or NCCL
environment invalidates direct comparison with this run.
