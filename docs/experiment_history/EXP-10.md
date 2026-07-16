# EXP-10: Runpod NVLink P2P and NCCL communication

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-10`, `nemo-megatron`, `runpod`, `RUNPOD-A100-SXM2`

## Goal And Design

Educational goal: Learn how to qualify an A100 SXM/NVLink host and explain how NVLink topology changes P2P and collective behavior compared with the AWS PCIe baseline.

### Scenario

A team rents a two-GPU A100 SXM Pod for later model-parallel training. The
provider product name is not enough evidence that the selected pair is actually
connected by the expected fabric or that NCCL uses it.

### Question

Does NCCL behavior agree with the observed A100 SXM topology and CUDA P2P
measurements?

### Hypothesis

If the Pod exposes two A100 SXM GPUs linked by NVLink, then CUDA P2P and NCCL
collective bandwidth will show the high-bandwidth regime expected from that
topology rather than a CPU or PCIe-only fallback.

### Decision rule

The hypothesis is confirmed when topology, P2P, and NCCL evidence agree on a
GPU-direct high-bandwidth path. It is rejected when the selected pair lacks
NVLink or NCCL falls back to an unexpected path. Missing counters make only the
counter-specific conclusion inconclusive.

### Configuration Snapshot

| Field | Value |
| --- | --- |
| Framework/image family | nemo-megatron |
| Image | `ghcr.io/helian2s/multi-gpu-training-nemo@sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b` |
| Provider | runpod |
| Compute profile | RUNPOD-A100-SXM2 |
| Resource type | secure-cloud-pod |
| GPU type | NVIDIA A100-SXM4-80GB |
| Physical GPUs | 2 |
| Normal visible GPUs | 2 |
| Workload profile | communication |
| Baseline | default_nccl_environment |
| Declared variants | 7 |

## Current Conclusion

Runpod A2 exposed two A100-SXM4-80GB GPUs with NV12 topology. P2P writes reached about 269-274 GB/s per direction and about 518-525 GB/s bidirectional. NCCL collective bus bandwidth was much higher than AWS PCIe for comparable two-GPU collectives.

## Interpretation

The A100 SXM host provided NVLink/NVS topology with 12 links per GPU, so it is the right environment for the topology-sensitive Megatron TP/SP/CP experiments.

## Run Inventory

### runpod-a2-megatron-20260715T203057Z

- Run ID: `runpod-a2-megatron-20260715T203057Z`
- Artifact directory: `artifacts/runs/runpod-volume-mirror/EXP-10/runpod-a2-megatron-20260715T203057Z`
- Raw artifact file count below `raw/`: `13`

## Communication And Topology From `runpod-a2-megatron-20260715T203057Z`

- CUDA peer access is enabled in both directions.
- Unidirectional P2P write bandwidth: GPU0->GPU1 268.96 GB/s, GPU1->GPU0 274.12 GB/s.
- Bidirectional P2P bandwidth: GPU0<->GPU1 518.10 and 525.26 GB/s.
- Enabled GPU P2P latency: GPU0->GPU1 2.88 us, GPU1->GPU0 3.01 us.

NCCL average bus bandwidth:

| Collective | Avg bus bandwidth GB/s | Raw log |
| --- | --- | --- |
| all_reduce | 54.0993 | artifacts/runs/runpod-volume-mirror/EXP-10/runpod-a2-megatron-20260715T203057Z/raw/nccl_all_reduce.txt |
| all_gather | 42.2329 | artifacts/runs/runpod-volume-mirror/EXP-10/runpod-a2-megatron-20260715T203057Z/raw/nccl_all_gather.txt |
| broadcast | 74.4334 | artifacts/runs/runpod-volume-mirror/EXP-10/runpod-a2-megatron-20260715T203057Z/raw/nccl_broadcast.txt |
| reduce_scatter | 43.7539 | artifacts/runs/runpod-volume-mirror/EXP-10/runpod-a2-megatron-20260715T203057Z/raw/nccl_reduce_scatter.txt |
| all_to_all | 46.0213 | artifacts/runs/runpod-volume-mirror/EXP-10/runpod-a2-megatron-20260715T203057Z/raw/nccl_all_to_all.txt |

Topology excerpt:

```text
GPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	NV12	0-95	0		N/A
GPU1	NV12	 X 	0-95	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
```

## Formal Report Status

- Report file: `experiments/exp_10_runpod_nvlink_p2p_nccl_communication/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

NVLink qualification should record both `nvidia-smi topo -m` and measured P2P/NCCL behavior. Do not infer NVLink just from GPU name.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.
