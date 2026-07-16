# EXP-01: AWS PCIe P2P and NCCL communication - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-01.md](../../docs/experiment_history/EXP-01.md).

## Executive conclusion

AWS-A2 G7e exposed two RTX PRO 6000 Blackwell GPUs over PIX/PCIe. CUDA peer access worked in both directions. P2P writes reached about 55 GB/s per direction and about 105 GB/s bidirectional, while the NCCL collectives showed materially lower bus bandwidth than the Runpod NVLink baseline.

## Run inventory

### aws-a2-full-fp16fix-20260715T023252Z

- Run ID: `aws-a2-full-fp16fix-20260715T023252Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-01/runs/aws-a2-full-fp16fix-20260715T023252Z`
- Exit statuses: `exit_status-EXP-01-A2.txt`=0
- Finished UTC markers: `finished-EXP-01-A2-utc.txt`=2026-07-15T02:34:06Z
- Raw artifact file count below `raw/`: `13`

## Measured results

### `aws-a2-full-fp16fix-20260715T023252Z` communication/topology

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

## Interpretation

PCIe P2P is usable and much faster than host-staged fallback for latency, but it is not equivalent to NVLink. The AWS logs show NCCL building local rings over PIX and using direct CUDA P2P for the two local GPUs.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

Always qualify topology before interpreting distributed-training speed. Two GPUs on one host can have peer access while still being limited by PCIe-class bandwidth.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.
