# EXP-14: TP=2 x DP=2 for model width and throughput - report

Report status: Completed from Runpod A4 run
`runpod-a4-megatron-20260715T212017Z`

## Executive conclusion

EXP-14 completed on a four-GPU Runpod A100 SXM Pod. The run validated the
rank-map mechanics for DP=4, TP=4, and TP=2 x DP=2 on one NVLink-connected
host. In this bounded synthetic MLP workload, DP=4 gave the highest aggregate
tokens/s, TP=4 used the least memory and least estimated communication, and the
hybrid TP=2 x DP=2 layout landed between them on throughput, memory, and
collective volume.

## Run inventory

- Successful run ID: `runpod-a4-megatron-20260715T212017Z`
- Queue: `RUNPOD-A4-Megatron`
- Run units: `QUAL-RUNPOD-A4-Megatron`, `EXP-14-RUNPOD-A4-Megatron`
- Exit statuses: both run units exited `0`
- Pod: `iijfdo0p8nvbpx`, Runpod Secure Cloud, 4 x A100-SXM4-80GB
- Local ignored mirror:
  `artifacts/runs/runpod-volume-mirror/{QUAL-RUNPOD-A4-Megatron,EXP-14,RUNPOD-A4-Megatron}/runpod-a4-megatron-20260715T212017Z/`
- Excluded diagnostic run: `runpod-a4-megatron-20260715T210918Z`; it wrote raw
  rank metrics but was terminated with queue status `143` after two ranks hung
  during explicit NCCL process-group cleanup.

## Environment

Qualification observed four `NVIDIA A100-SXM4-80GB` GPUs with 81,920 MiB each,
driver `580.126.16`, CUDA available through PyTorch `12.8`, PyTorch
`2.7.0a0+7c8ec84dab.nv25.03`, Megatron Core `0.12.3`, Transformer Engine
`2.2.0.dev0+bee4649c`, and Python `3.12.3`. `nvidia-smi topo -m` reported
`NV12` links between every GPU pair used by the experiment.

Image:
`ghcr.io/helian2s/multi-gpu-training-nemo@sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b`

The completed rerun used the same image with
`common/megatron_executor.py` hotpatched in the running container to skip
explicit NCCL process-group destruction on subprocess exit. The patch affects
worker cleanup after metrics are written, not the measured forward/backward
step. Publish a replacement immutable image containing this fix before future
reproducibility runs.

## Method

The experiment ran a synthetic Megatron-aware MLP training step on four ranks
with identical precision, sequence length, hidden size, microbatch size, warmup
count, and measured-step count. Variants:

- `dp4`: four data-parallel replicas, no tensor sharding.
- `tp4`: one four-rank tensor-parallel group, no data-parallel replicas.
- `tp2_dp2`: tensor groups `[0,1]` and `[2,3]` with data groups `[0,2]` and
  `[1,3]`.

Each rank recorded max-rank step time, aggregate tokens/s, peak allocated GPU
memory, process-group membership, and estimated collective bytes.

## Results

| Variant | Layout | TP | DP | Ranks | Max step ms | Tokens/s | Peak MiB | Collective MiB/rank |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `EXP-14-RUNPOD-A4-dp4` | `dp4` | 1 | 4 | `[0, 1, 2, 3]` | 3.832 | 534,415 | 183.3 | 896.0 |
| `EXP-14-RUNPOD-A4-tp4` | `tp4` | 4 | 1 | `[0, 1, 2, 3]` | 2.345 | 218,329 | 60.3 | 28.0 |
| `EXP-14-RUNPOD-A4-tp2-dp2` | `tp2_dp2` | 2 | 2 | `[0, 1, 2, 3]` | 3.316 | 308,836 | 101.3 | 238.0 |

## Interpretation

- DP=4 was fastest in aggregate tokens/s for this small synthetic workload
  because each rank kept a full local shard-free model and contributed
  independent data-parallel samples.
- TP=4 reduced peak memory from `183.3 MiB` to `60.3 MiB`, but aggregate
  tokens/s was lower because the per-step work was model-sharded rather than
  replicated across four samples.
- TP=2 x DP=2 provided the intended middle point: less memory and less
  estimated collective traffic than DP=4, with more aggregate throughput than
  TP=4.
- The rank maps match the planned hybrid interpretation: TP groups shard the
  model within each replica, and DP groups synchronize corresponding TP shards
  across replicas.

## Limitations and anomalies

This is a bounded synthetic workload that imports and records Megatron Core
availability, but it is not a full Qwen3 Megatron Bridge recipe. It is valid
for process-group mechanics and directional throughput/memory trade-offs, not
for model-quality claims.

The first A4 attempt exposed a cleanup-path issue: metrics were written, but
two ranks stayed in uninterruptible cleanup after `dist.destroy_process_group()`
with custom TP/DP NCCL groups. The executor now relies on short-lived subprocess
exit for NCCL cleanup unless `MGT_DESTROY_PROCESS_GROUP=1` is explicitly set.

## Cost

The successful queue itself ran for about 68 seconds. Including the diagnostic
run, restart, and manual SSH/debug time, the Pod was active from approximately
20:44:19 UTC to the successful queue completion at 21:21:26 UTC, about
0.62 billed Pod hours. At `$5.96/hr`, that is about `$3.69` before any
post-run idle inspection time. The Pod remained running after artifact mirror
so the user could verify state manually.

## Exam takeaway

Hybrid TP x DP is useful when neither pure replication nor pure model sharding
matches the memory and throughput target. On four GPUs, TP=2 x DP=2 is the
smallest layout that exercises both non-trivial tensor-parallel and
data-parallel groups; choose it when the model needs some sharding but the
system still benefits from replicated data-parallel throughput.

## Reproduction

Artifacts are mirrored locally under `artifacts/runs/runpod-volume-mirror/`.
Re-run the summary table with:

```bash
python experiments/exp_14_tp2_dp2_hybrid/analyze.py \
  artifacts/runs/runpod-volume-mirror/EXP-14/runpod-a4-megatron-20260715T212017Z
```
