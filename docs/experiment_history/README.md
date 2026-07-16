# Experiment History For RAG

This directory is a tracked, text-first summary of the project experiment history. It is intentionally redundant with `EXPERIMENT_CATALOG.md`, `PROJECT_DECISIONS.md`, experiment specs, reports, and artifact mirrors so another workstation can answer questions about experiment goals, methods, results, and known anomalies without relying on Codex chat history.

The files here are derived from tracked experiment specifications plus the local ignored artifact mirrors under `artifacts/runs/`. Raw logs, profiler traces, checkpoints, credentials, and model/data caches remain outside Git.

## How To Use

- Start with this README for the table of all experiments.
- Use `EXP-NN.md` for RAG chunks about a single experiment.
- Use `operational_timeline.md` for launch, image, provider, and bug-fix history.
- Restore the compressed artifact archive when exact raw logs or profiler traces are needed.
- Prefer these history files over stale placeholder report text for experiments whose formal reports are still pending.

## Experiment Index

| ID | Experiment | Framework | Provider | Compute profile | Run IDs | State | Current conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [EXP-01](EXP-01.md) | AWS PCIe P2P and NCCL communication | pytorch | aws | AWS-A2 | aws-a2-full-fp16fix-20260715T023252Z | artifact mirror present | AWS-A2 G7e exposed two RTX PRO 6000 Blackwell GPUs over PIX/PCIe. CUDA peer access worked in both directions. P2P writes reached about 55... |
| [EXP-02](EXP-02.md) | Mixed precision and Tensor Cores in distributed training | pytorch | aws | AWS-A2 | aws-a2-full-fp16fix-20260715T023252Z | artifact mirror present | Reduced precision accelerated eligible matrix and training paths on AWS-A2. BF16 was the strongest measured training mode: one-rank BF16... |
| [EXP-03](EXP-03.md) | Microbatch, global batch, and gradient accumulation | pytorch | aws | AWS-A1 | aws-a1-pytorch-20260715T003637Z | artifact mirror present | With fixed effective global batch 8 on one AWS GPU, larger microbatches reduced optimizer-step time until microbatch 4, then memory press... |
| [EXP-04](EXP-04.md) | Activation checkpointing/recomputation | pytorch | aws | AWS-A1 | aws-a1-pytorch-20260715T003637Z | artifact mirror present | Activation checkpointing reduced peak memory at both sequence lengths and increased step time. At sequence length 4096, peak memory fell... |
| [EXP-05](EXP-05.md) | PyTorch SDPA/FlashAttention and operator fusion | pytorch | aws | AWS-A1 | aws-a1-pytorch-20260715T003637Z | artifact mirror present | Automatic/FlashAttention SDPA was dramatically faster and smaller than the math backend for the Qwen-like GQA shape. Automatic SDPA measu... |
| [EXP-06](EXP-06.md) | Profiler triangulation | pytorch | aws | AWS-A1 | aws-a1-pytorch-20260715T003637Z | artifact mirror present | Profiler triangulation produced PyTorch Profiler traces/key averages for math and automatic attention backends, and confirmed Nsight Syst... |
| [EXP-07](EXP-07.md) | DDP scaling and communication overlap | pytorch | aws | AWS-A2 | aws-a2-full-fp16fix-20260715T023252Z | artifact mirror present | The two-rank DDP variants completed on AWS-A2. One rank measured about 113 ms/step, while two-rank DDP with batch 2 measured about 240 ms... |
| [EXP-08](EXP-08.md) | FSDP sharding and ZeRO-style memory trade-offs | pytorch | aws | AWS-A2 | aws-a2-full-fp16fix-20260715T023252Z | artifact mirror present | FSDP completed on two AWS GPUs and reduced peak memory versus DDP from about 19.3 GiB to 14.5 GiB per rank while also reducing step time... |
| [EXP-09](EXP-09.md) | Controlled troubleshooting and failure diagnosis | pytorch | aws | AWS-A2 | aws-a2-full-fp16fix-20260715T023252Z | artifact mirror present | Controlled failure cases produced the intended signatures: healthy smoke passed, OOM and FP16 overflow were expected failures, input star... |
| [EXP-10](EXP-10.md) | Runpod NVLink P2P and NCCL communication | nemo-megatron | runpod | RUNPOD-A100-SXM2 | runpod-a2-megatron-20260715T203057Z | artifact mirror present | Runpod A2 exposed two A100-SXM4-80GB GPUs with NV12 topology. P2P writes reached about 269-274 GB/s per direction and about 518-525 GB/s... |
| [EXP-11](EXP-11.md) | Tensor plus sequence parallelism | nemo-megatron | runpod | RUNPOD-A100-SXM2 | runpod-a2-megatron-20260715T203057Z | artifact mirror present | Tensor parallelism reduced per-rank memory from about 190 MiB to 106 MiB and introduced non-zero collective traffic. Sequence parallelism... |
| [EXP-12](EXP-12.md) | Pipeline schedules and bubble size | nemo-megatron | aws | AWS-A2 | aws-a2-megatron-exp12-20260715T0340Z | completed report present | Pipeline schedule mechanics were validated on AWS-A2. More microbatches reduced estimated bubble fraction, PP=2 introduced point-to-point... |
| [EXP-13](EXP-13.md) | Context parallelism for long sequences | nemo-megatron | runpod | RUNPOD-A100-SXM2 | runpod-a2-megatron-20260715T203057Z | artifact mirror present | Context parallel variants completed on Runpod A2. CP=2 halved local sequence length for the same global sequence and lowered memory versu... |
| [EXP-14](EXP-14.md) | TP=2 x DP=2 for model width and throughput | nemo-megatron | runpod | RUNPOD-A100-SXM4 | runpod-a4-megatron-20260715T210918Z, runpod-a4-megatron-20260715T212017Z | completed report present | The four-GPU Runpod A4 hybrid run completed. DP=4 gave the highest aggregate tokens/s in the bounded workload, TP=4 used the least memory... |

## Artifact Mirrors Used

| Mirror | Exists | Files |
| --- | --- | --- |
| artifacts/runs/aws-s3-mirror | yes | 226 |
| artifacts/runs/runpod-volume-mirror | yes | 121 |

## Regeneration

After restoring `artifacts/runs/` on another workstation, regenerate this directory with:

```bash
python3 scripts/build_experiment_history.py
```
