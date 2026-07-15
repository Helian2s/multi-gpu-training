# EXP-12: Pipeline schedules and bubble size - report

Report status: Completed from AWS-A2 run `aws-a2-megatron-exp12-20260715T0340Z`

## Executive conclusion

EXP-12 completed on AWS-A2 with the NeMo/Megatron image. The run validated the
expected pipeline-schedule mechanics: increasing microbatches reduced the
estimated two-stage bubble fraction, PP=2 introduced point-to-point activation
and gradient traffic, and an imbalanced stage split reduced throughput.

## Run inventory

- Run ID: `aws-a2-megatron-exp12-20260715T0340Z`
- Queue: `AWS-A2-Megatron`
- Run units: `QUAL-A2`, `EXP-12-A2V1`, `EXP-12-A2V2`
- Exit statuses: all three run units exited `0`
- S3 prefixes:
  `artifacts/QUAL-A2/runs/aws-a2-megatron-exp12-20260715T0340Z/`,
  `artifacts/EXP-12/runs/aws-a2-megatron-exp12-20260715T0340Z/`, and
  `artifacts/AWS-A2-Megatron/runs/aws-a2-megatron-exp12-20260715T0340Z/`
- Local ignored mirror:
  `artifacts/runs/aws-s3-mirror/{QUAL-A2,EXP-12,AWS-A2-Megatron}/runs/aws-a2-megatron-exp12-20260715T0340Z/`

## Environment

AWS `g7e.12xlarge` through profile `AWS-A2`, with one- and two-visible-GPU
phases on the same physical host profile. Qualification observed two
`NVIDIA RTX PRO 6000 Blackwell Server Edition` GPUs with 97,887 MiB each,
driver `595.71.05`, CUDA `13.2`, PyTorch `2.12.0a0+0291f960b6.nv26.04.48445190`,
Megatron Core `0.18.0`, Megatron Bridge `0.5.0`, and Transformer Engine
`2.16.0+4220403e`. `nvidia-smi topo -m` reported `PIX` between GPU0 and GPU1.

Image:
`037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-nemo@sha256:ea7616a570d7e271eff25b4f3c0655a9910024e119171e2ead7569f6714b35fb`

## Method

The AWS adaptation uses a bounded synthetic pipeline-stage workload inside the
NeMo/Megatron image. It compares PP=1, PP=2 flush-style scheduling, PP=2 1F1B
style scheduling, microbatch-count sensitivity, and one imbalanced stage split.

## Results

| Variant | Schedule | Microbatches | Bubble estimate | Max step ms | Tokens/s | Peak MiB | Stages |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `EXP-12-A2V1-pp1-mb4-balanced` | no pipeline | 4 | 0.000 | 3.074 | 666,128 | 155.0 | `[4]` |
| `EXP-12-A2V2-pp2-flush-mb2-balanced` | flush | 2 | 0.333 | 1.738 | 589,223 | 117.0 | `[2, 2]` |
| `EXP-12-A2V2-pp2-flush-mb8-balanced` | flush | 8 | 0.111 | 5.312 | 771,084 | 177.0 | `[2, 2]` |
| `EXP-12-A2V2-pp2-1f1b-mb2-balanced` | 1F1B style | 2 | 0.333 | 1.777 | 576,400 | 116.0 | `[2, 2]` |
| `EXP-12-A2V2-pp2-1f1b-mb8-balanced` | 1F1B style | 8 | 0.111 | 5.586 | 733,266 | 122.0 | `[2, 2]` |
| `EXP-12-A2V2-pp2-1f1b-mb8-imbalanced` | 1F1B style | 8 | 0.111 | 6.905 | 593,198 | 136.0 | `[1, 3]` |

## Interpretation

- The two-stage bubble estimate fell from `0.333` at two microbatches to
  `0.111` at eight microbatches, matching the expected formula
  `(pipeline_stages - 1) / (microbatches + pipeline_stages - 1)`.
- The higher-microbatch balanced PP=2 variants processed more tokens per second
  than the PP=1 baseline in this synthetic workload, despite longer per-step
  wall time because each step contained more microbatches.
- PP=2 variants recorded point-to-point traffic: about 28 MiB per rank at two
  microbatches and 112 MiB per rank at eight microbatches. PP=1 recorded none.
- The imbalanced `[1, 3]` stage split was slower than the balanced `[2, 2]`
  1F1B-style eight-microbatch run: `6.905 ms` vs `5.586 ms` max-rank step time,
  and `593k` vs `733k` tokens/s.

## Limitations and anomalies

The AWS run uses PCIe peer access on G7e, not A100 SXM/NVLink. It is valid for
pipeline-schedule mechanics but must not be interpreted as a Runpod NVLink
Megatron result.

The workload is a bounded synthetic stage benchmark that imports and records
Megatron Core pipeline schedule availability, but it does not train a full
Qwen3 Megatron recipe. The 1F1B-style path is a local point-to-point schedule
used for controlled evidence, not a full Megatron interleaved pipeline engine.

## Cost

One `g7e.12xlarge` instance was launched and stopped by the queue success path
after the 15-minute hold. Additional retained 300 GiB gp3 cache volumes were
created in `us-west-2a`, `us-west-2c`, and `us-west-2d` so future AWS-selected
placement can attach a matching cache volume.

## Exam takeaway

For pipeline parallelism, do not assume splitting layers across GPUs is enough.
Choose enough microbatches to reduce the bubble, keep stage work balanced, and
account for the point-to-point activation/gradient traffic introduced by PP.

## Reproduction

Artifacts are mirrored locally under `artifacts/runs/aws-s3-mirror/`. Re-run
the summary table with:

```bash
python experiments/exp_12_pipeline_schedules_bubble_size/analyze.py \
  artifacts/runs/aws-s3-mirror/EXP-12/runs/aws-a2-megatron-exp12-20260715T0340Z
```
