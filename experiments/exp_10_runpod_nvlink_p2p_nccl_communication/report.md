# EXP-10: Runpod NVLink P2P and NCCL communication - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-10.md](../../docs/experiment_history/EXP-10.md).

## Executive conclusion

Runpod A2 exposed two A100-SXM4-80GB GPUs with NV12 topology. P2P writes reached about 269-274 GB/s per direction and about 518-525 GB/s bidirectional. NCCL collective bus bandwidth was much higher than AWS PCIe for comparable two-GPU collectives.

## Run inventory

### runpod-a2-megatron-20260715T203057Z

- Run ID: `runpod-a2-megatron-20260715T203057Z`
- Artifact directory: `artifacts/runs/runpod-volume-mirror/EXP-10/runpod-a2-megatron-20260715T203057Z`
- Raw artifact file count below `raw/`: `13`

## Measured results

### `runpod-a2-megatron-20260715T203057Z` communication/topology

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

## Interpretation

The A100 SXM host provided NVLink/NVS topology with 12 links per GPU, so it is the right environment for the topology-sensitive Megatron TP/SP/CP experiments.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

NVLink qualification should record both `nvidia-smi topo -m` and measured P2P/NCCL behavior. Do not infer NVLink just from GPU name.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.
