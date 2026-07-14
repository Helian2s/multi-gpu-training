# EXP-09: Controlled troubleshooting and failure diagnosis

Design status: Draft

Catalog entry: [EXPERIMENT_CATALOG.md](../../EXPERIMENT_CATALOG.md)

## Scenario

A training job fails or slows down in several different ways: OOM during
backward, NaNs after enabling FP16, idle GPUs while the input pipeline works,
and distributed hangs around collectives. The team needs a bounded diagnostic
playbook rather than a single generic failure label.

## Question

Can common single-rank and two-rank training failures be recognized from their
logs, utilization, memory, timeout, numerical, and timeline evidence, and can
the corrective action be proven with a follow-up healthy case?

## Hypothesis

Each injected failure will produce a distinct evidence pattern: OOM cases should
show memory-pressure signatures, FP16 instability should show non-finite loss or
gradients, input starvation should show GPU idle gaps with CPU/data waiting, and
distributed faults should show rank-specific NCCL or PyTorch distributed logs.
Corrective variants should remove the symptom while preserving the same healthy
training workload.

## Rationale

Troubleshooting distributed training is mostly evidence classification. The
same high-level symptom can come from memory, numerical instability, data input,
rank disagreement, or communication timeout. Bounded injected faults create a
known ground truth for learning which signal identifies which root cause.

## Variables

- Independent variables: injected fault case and corrective action.
- Fixed variables: one AWS `g7e.12xlarge`, pinned inputs, restored checkpoint
  between cases, bounded timeouts, fixed seeds, and one immutable PyTorch image.
- Dependent metrics: exit status, log signature, timeout behavior, GPU
  utilization, memory evidence, CPU/input wait evidence, root-cause
  classification, and recovery proof.
- Baseline: `EXP-09-A2V1` healthy one-rank smoke case.

## Correctness gates

- AWS-A2 qualification and input manifest verification pass.
- Healthy smoke case passes before fault cases run.
- Every fault case exits within its hard timeout.
- Controlled failures are classified as expected failures, not infrastructure
  failures.
- Recovery cases complete successfully and demonstrate the corrective action.
- Distributed ranks are cleaned up after every fault case.
- Artifacts are staged out even when the case intentionally fails.

## Decision rule

Confirm the hypothesis when each injected fault produces its expected evidence
signature and the paired corrective action removes the symptom in a bounded
follow-up run.

Reject it when fault evidence is ambiguous enough that the root cause cannot be
classified or when the corrective action does not resolve the symptom.

Leave it inconclusive when logs, utilization, memory telemetry, timeout data,
rank cleanup evidence, input identity, or image identity is incomplete.

## Invalidating conditions

- An injected fault can hang beyond the configured timeout.
- A controlled failure is indistinguishable from infrastructure failure.
- Distributed ranks remain alive after a case finishes.
- Recovery is asserted without a successful corrected run.
- Artifacts are not durably staged out.
