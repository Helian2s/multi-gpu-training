# EXP-01: AWS PCIe P2P and NCCL communication

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-01`, `pytorch`, `aws`, `AWS-A2`

## Goal And Design

Educational goal: Learn how to turn raw GPU topology, peer-access, P2P bandwidth/latency, and NCCL collective measurements into a concrete explanation of two-GPU communication behavior on one PCIe server.

### Scenario

A two-GPU EC2 training job scales poorly. Before changing model code, the team
needs to determine whether the two GPUs have a usable GPUDirect P2P path, what
latency and bandwidth that path actually provides, and how NCCL collective
behavior changes from small to large messages on the same server.

### Question

Does NCCL behavior agree with the measured G7e PCIe topology and
peer-to-peer transfer characteristics?

### Hypothesis

The two RTX PRO 6000 Blackwell Server Edition GPUs on one `g7e.12xlarge`
should expose a PCIe peer path whose CUDA P2P latency/bandwidth measurements
explain the shape of the NCCL collective curves. Small collectives should be
latency-bound, while large collectives should move toward a bandwidth-bound
regime. No exact bandwidth threshold is predeclared because the actual topology,
driver, and provider host must be measured.

### Decision rule

Confirm the hypothesis when the peer-access matrix, P2P measurements, NCCL
debug output, and collective curves form a coherent explanation of the observed
small-message latency and large-message bandwidth regimes.

Reject it when P2P and NCCL behavior disagree materially and the discrepancy
cannot be explained by topology, NCCL algorithm selection, driver/runtime
version, or host bottlenecks.

Leave it inconclusive when qualification, GPU visibility, P2P, NCCL logs, or
artifact stage-out are incomplete.

### Configuration Snapshot

| Field | Value |
| --- | --- |
| Framework/image family | pytorch |
| Image | `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486` |
| Provider | aws |
| Compute profile | AWS-A2 |
| Resource type | g7e.12xlarge |
| GPU type | RTX PRO 6000 Blackwell Server Edition |
| Physical GPUs | 2 |
| Normal visible GPUs | 2 |
| Workload profile | communication |
| Baseline | default_nccl_environment |
| Declared variants | 7 |

## Current Conclusion

AWS-A2 G7e exposed two RTX PRO 6000 Blackwell GPUs over PIX/PCIe. CUDA peer access worked in both directions. P2P writes reached about 55 GB/s per direction and about 105 GB/s bidirectional, while the NCCL collectives showed materially lower bus bandwidth than the Runpod NVLink baseline.

## Interpretation

PCIe P2P is usable and much faster than host-staged fallback for latency, but it is not equivalent to NVLink. The AWS logs show NCCL building local rings over PIX and using direct CUDA P2P for the two local GPUs.

## Run Inventory

### aws-a2-full-fp16fix-20260715T023252Z

- Run ID: `aws-a2-full-fp16fix-20260715T023252Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-01/runs/aws-a2-full-fp16fix-20260715T023252Z`
- Exit statuses: `exit_status-EXP-01-A2.txt`=0
- Finished UTC markers: `finished-EXP-01-A2-utc.txt`=2026-07-15T02:34:06Z
- Raw artifact file count below `raw/`: `13`

## Communication And Topology From `aws-a2-full-fp16fix-20260715T023252Z`

- CUDA peer access is enabled in both directions.
- Unidirectional P2P write bandwidth: GPU0->GPU1 54.73 GB/s, GPU1->GPU0 55.80 GB/s.
- Bidirectional P2P bandwidth: GPU0<->GPU1 104.78 and 104.81 GB/s.
- Enabled GPU P2P latency: GPU0->GPU1 0.47 us, GPU1->GPU0 0.53 us.

NCCL average bus bandwidth:

| Collective | Avg bus bandwidth GB/s | Raw log |
| --- | --- | --- |
| all_reduce | 12.7264 | artifacts/runs/aws-s3-mirror/EXP-01/runs/aws-a2-full-fp16fix-20260715T023252Z/raw/nccl_all_reduce.txt |
| all_gather | 10.8577 | artifacts/runs/aws-s3-mirror/EXP-01/runs/aws-a2-full-fp16fix-20260715T023252Z/raw/nccl_all_gather.txt |
| broadcast | 15.062 | artifacts/runs/aws-s3-mirror/EXP-01/runs/aws-a2-full-fp16fix-20260715T023252Z/raw/nccl_broadcast.txt |
| reduce_scatter | 10.3584 | artifacts/runs/aws-s3-mirror/EXP-01/runs/aws-a2-full-fp16fix-20260715T023252Z/raw/nccl_reduce_scatter.txt |
| all_to_all | 12.1193 | artifacts/runs/aws-s3-mirror/EXP-01/runs/aws-a2-full-fp16fix-20260715T023252Z/raw/nccl_all_to_all.txt |

Topology excerpt:

```text
GPU0	GPU1	CPU Affinity	NUMA Affinity	GPU NUMA ID
GPU0	 X 	PIX	0-47	0		N/A
GPU1	PIX	 X 	0-47	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
```

## Formal Report Status

- Report file: `experiments/exp_01_aws_pcie_p2p_nccl_communication/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

Always qualify topology before interpreting distributed-training speed. Two GPUs on one host can have peer access while still being limited by PCIe-class bandwidth.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.
