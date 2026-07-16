# EXP-09: Controlled troubleshooting and failure diagnosis

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-09`, `pytorch`, `aws`, `AWS-A2`

## Goal And Design

Educational goal: Learn to identify common training and distributed failure classes from logs, utilization, memory, timeout, numerical, and timeline evidence, then verify that the corrective action actually fixes the root cause.

### Scenario

A training job fails or slows down in several different ways: OOM during
backward, NaNs after enabling FP16, idle GPUs while the input pipeline works,
and distributed hangs around collectives. The team needs a bounded diagnostic
playbook rather than a single generic failure label.

### Question

Can common single-rank and two-rank training failures be recognized from their
logs, utilization, memory, timeout, numerical, and timeline evidence, and can
the corrective action be proven with a follow-up healthy case?

### Hypothesis

Each injected failure will produce a distinct evidence pattern: OOM cases should
show memory-pressure signatures, FP16 instability should show non-finite loss or
gradients, input starvation should show GPU idle gaps with CPU/data waiting, and
distributed faults should show rank-specific NCCL or PyTorch distributed logs.
Corrective variants should remove the symptom while preserving the same healthy
training workload.

### Decision rule

Confirm the hypothesis when each injected fault produces its expected evidence
signature and the paired corrective action removes the symptom in a bounded
follow-up run.

Reject it when fault evidence is ambiguous enough that the root cause cannot be
classified or when the corrective action does not resolve the symptom.

Leave it inconclusive when logs, utilization, memory telemetry, timeout data,
rank cleanup evidence, input identity, or image identity is incomplete.

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
| Normal visible GPUs | variant-dependent |
| Workload profile | smoke |
| Baseline | EXP-09-A2V1-healthy-smoke |
| Declared variants | 7 |

## Current Conclusion

Controlled failure cases produced the intended signatures: healthy smoke passed, OOM and FP16 overflow were expected failures, input starvation produced a timed wait, mismatched collectives surfaced an explicit collective fingerprint mismatch, and a timeout case recorded bounded timeout evidence.

## Interpretation

The experiment created known-bad cases so future troubleshooting can distinguish memory pressure, numerical overflow, input stalls, rank stragglers, collective mismatches, and NCCL timeout symptoms.

## Run Inventory

### aws-a2-full-fp16fix-20260715T023252Z

- Run ID: `aws-a2-full-fp16fix-20260715T023252Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-09/runs/aws-a2-full-fp16fix-20260715T023252Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486`
- Manifest status: `completed`
- Run units: `EXP-09-A2V2`
- Planned variant count in manifest/plan: `3`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-09/runs/aws-a2-full-fp16fix-20260715T023252Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status-EXP-09-A2V1.txt`=0, `exit_status-EXP-09-A2V2.txt`=0
- Finished UTC markers: `finished-EXP-09-A2V1-utc.txt`=2026-07-15T02:39:31Z, `finished-EXP-09-A2V2-utc.txt`=2026-07-15T02:41:54Z
- Raw artifact file count below `raw/`: `18`

## Measured Results From `aws-a2-full-fp16fix-20260715T023252Z`

| Variant | Case | World | Status | Elapsed s | Detail |
| --- | --- | --- | --- | --- | --- |
| EXP-09-A2V1-healthy-smoke | healthy_training_smoke | 1 | pass | 0.165 | healthy CUDA smoke completed |
| EXP-09-A2V1-oom-fragmentation | bounded_oom_or_fragmentation | 1 | expected_failure | 0 | controlled CUDA out of memory diagnostic |
| EXP-09-A2V1-numerical-overflow | fp16_nonfinite_gradient | 1 | expected_failure | 0.558 | controlled fp16 non-finite diagnostic |
| EXP-09-A2V1-input-starvation | input_pipeline_starvation | 1 | pass | 5 | controlled input wait of 5.0 seconds |
| EXP-09-A2V2-mismatched-collective | mismatched_collective | 2 | expected_failure | 0.965 | Detected mismatch between collectives on ranks. Rank 0 is running collective: CollectiveFingerPrint(SequenceNumber=0, OpType=ALLREDUCE, TensorShape=[1], TensorDtypes=Float, Tens... |
| EXP-09-A2V2-rank-straggler | artificial_rank_straggler | 2 | pass | 10.955 | straggler collective completed |
| EXP-09-A2V2-nccl-timeout-evidence | nccl_timeout_signature | 2 | expected_failure | 120 | controlled timeout after 120s; see /workspace/multi-gpu-training/artifacts/runs/EXP-09/aws-a2-full-fp16fix-20260715T023252Z/raw/EXP-09-A2V2-nccl-timeout-evidence/subprocess.log |

## Formal Report Status

- Report file: `experiments/exp_09_controlled_troubleshooting_failure_diagnosis/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

Good distributed debugging starts from the failure signature: OOM, non-finite gradients, input wait, rank mismatch, and timeout each point to different first checks.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.
