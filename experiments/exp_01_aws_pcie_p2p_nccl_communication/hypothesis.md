# EXP-01: AWS PCIe P2P and NCCL communication

Design status: Draft

Catalog entry: [EXPERIMENT_CATALOG.md](../../EXPERIMENT_CATALOG.md)

## Scenario

A two-GPU EC2 training job scales poorly. Before changing model code, the team
needs to determine whether the two GPUs have a usable GPUDirect P2P path, what
latency and bandwidth that path actually provides, and how NCCL collective
behavior changes from small to large messages on the same server.

## Question

Does NCCL behavior agree with the measured G7e PCIe topology and
peer-to-peer transfer characteristics?

## Hypothesis

The two RTX PRO 6000 Blackwell Server Edition GPUs on one `g7e.12xlarge`
should expose a PCIe peer path whose CUDA P2P latency/bandwidth measurements
explain the shape of the NCCL collective curves. Small collectives should be
latency-bound, while large collectives should move toward a bandwidth-bound
regime. No exact bandwidth threshold is predeclared because the actual topology,
driver, and provider host must be measured.

## Rationale

G7e is a PCIe-connected GPU server rather than an NVLink/NVSwitch domain. CUDA
peer access and NCCL transport selection therefore need to be measured on the
allocated host rather than inferred from GPU marketing names. `nvidia-smi topo
-m` gives the topology inventory, CUDA Samples measures direct peer transfer,
and `nccl-tests` shows whether collective latency and bus bandwidth are
consistent with that peer path.

## Variables

- Independent variables: message size and collective type.
- Fixed variables: one AWS `g7e.12xlarge`, two visible GPUs, one immutable
  PyTorch/NVIDIA-tools image, default NCCL settings for the primary result.
- Dependent metrics: peer-access capability, P2P bandwidth, P2P latency, NCCL
  latency, NCCL algorithm, and NCCL bus bandwidth.
- Baseline: recorded host topology and CUDA P2P measurements before NCCL
  collective tests.

## Correctness gates

- Exactly two NVIDIA GPUs are visible in the container.
- `nvidia-smi -L`, `nvidia-smi topo -m`, driver, CUDA, NCCL, and image digest
  are recorded.
- `p2pBandwidthLatencyTest` exits successfully.
- Required `nccl-tests` collectives exit successfully without timeout or
  data-check failure.
- The artifact directory is staged out before the EC2 instance is terminated.

## Decision rule

Confirm the hypothesis when the peer-access matrix, P2P measurements, NCCL
debug output, and collective curves form a coherent explanation of the observed
small-message latency and large-message bandwidth regimes.

Reject it when P2P and NCCL behavior disagree materially and the discrepancy
cannot be explained by topology, NCCL algorithm selection, driver/runtime
version, or host bottlenecks.

Leave it inconclusive when qualification, GPU visibility, P2P, NCCL logs, or
artifact stage-out are incomplete.

## Invalidating conditions

- Instance type is not `g7e.12xlarge`.
- The run does not use exactly two visible GPUs on one EC2 host.
- The image digest, NVIDIA driver, CUDA, or NCCL version is missing.
- CUDA P2P or any required NCCL collective fails.
- The instance is not terminated or artifacts are not durably staged out.
