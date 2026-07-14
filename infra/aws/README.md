# AWS adapter plan

AWS is the first provider to qualify. The approved quota is:

- **Region:** `us-west-2` (US West, Oregon)
- **Quota:** Running On-Demand G and VT instances
- **Limit:** 96 vCPUs

This quota does not cover P-family A100/H100 instances or Spot G instances.
VT1 shares the quota bucket but contains video-transcoding accelerators rather
than CUDA GPUs, so it is not an experiment resource.

## Accepted primary compute profiles

| Profile | Instance type | GPUs | vCPUs | Host memory | Memory per GPU | Use |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `AWS-G7E-1` | `g7e.2xlarge` | 1 | 8 | 64 GiB | 96 GB | One-GPU baselines and one-GPU experiments |
| `AWS-G7E-2` | `g7e.12xlarge` | 2 | 48 | 512 GiB | 96 GB | Two-rank PyTorch and NVIDIA-tool experiments |
| `AWS-G7E-4` | `g7e.24xlarge` | 4 | 96 | 1,024 GiB | 96 GB | EXP-07 four-rank DDP sub-run only |

All three profiles use the RTX PRO 6000 Blackwell Server Edition. The two- and
four-GPU profiles support GPUDirect P2P over PCIe; the one-GPU profile has no
inter-GPU path. `AWS-G7E-4` consumes the complete approved quota, so the adapter
must verify that no other G or VT instance is running before launch. The
profiles remain subject to current On-Demand price, Availability Zone offering,
capacity, permission, container, and profiler qualification.

G6e/L40S is a contingency, not a silent runtime fallback. Using it would change
GPU memory, CPU allocation, topology, precision support, and potentially the
one-GPU workload boundary, so it requires a project-decision and catalog update.

Official sources:

- [G7e product details](https://aws.amazon.com/ec2/instance-types/g7e/)
- [G7e availability in `us-west-2`](https://aws.amazon.com/about-aws/whats-new/2026/02/amazon-ec2-g7e-instances-oregon-region/)
- [G6e product details](https://aws.amazon.com/ec2/instance-types/g6e/)
- [Accelerated-computing specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [EC2 instance-type quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)
- [NVIDIA CUDA GPU compute capabilities](https://developer.nvidia.com/cuda/gpus)
- [Transformer Engine FP8 Delayed Scaling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)
- [Transformer Engine MXFP8 support](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)
- [Transformer Engine NVFP4 support](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)

## Hardware consequences

- G7e and G6e use PCIe-connected GPUs rather than an A100/H100 NVLink/NVSwitch
  domain. Qualification must measure peer access, latency, and bandwidth before
  setting collective, DDP, or FSDP expectations.
- G7e's RTX PRO 6000 is SM 12.0. Standard FP8 is planned only after the pinned
  Transformer Engine stack proves a native recipe and kernels.
- Current Transformer Engine documentation lists MXFP8/NVFP4 training support
  for SM 10.0/10.3 rather than SM 12.0. Do not schedule those formats on G7e
  merely because the GPU advertises Blackwell FP4 operations.
- G6e L40S supports FP8 through a compatible Ada/Transformer Engine stack but
  does not support NVLink.
- Exact GPU model, memory, clocks, topology, P2P capability, and selected driver
  must come from the launched instance rather than this planning table.

## Planned implementation

- **Compute:** one G-family EC2 instance with a compatible NVIDIA host driver
  and container runtime; workloads pull the project image from private ECR.
- **Framework placement:** AWS measured training uses the PyTorch image; all
  NeMo/Megatron experiments are assigned to Runpod A100 SXM profiles.
- **Identity:** an instance IAM role with only the S3 and management permissions
  required by the implemented lifecycle plus ECR pull-only access.
- **Durable storage:** versioned S3 prefixes for pinned inputs and complete run
  artifacts.
- **Staging:** local instance-store NVMe for performance-sensitive temporary
  data; EBS where persistence across stop/start is worth its cost.
- **Access:** SSH or AWS Systems Manager will be chosen during qualification;
  experiment code must not depend on that choice.
- **Visibility:** use the exact `AWS-G7E-1`, `AWS-G7E-2`, or `AWS-G7E-4`
  profile; no masking is expected in normal AWS runs.
- **Cost safety:** require project/run tags, maximum instance lifetime, verified
  stage-out, and stop/termination in success and failure paths.

## EXP-01 preflight wrapper

`exp01_preflight.py` is a read-only draft wrapper for the first AWS experiment.
It verifies the selected account, Region, G/VT quota, `g7e.12xlarge` shape and
Availability Zone offerings, current On-Demand price, ECR repository/image
state, EC2 instance-role pull permissions, S3 artifact bucket, and budget
envelope.

EXP-01 writes run artifacts under:

```text
s3://finetuning-lab-1-037678282394-us-west-2-an/artifacts/EXP-01/
```

The EC2 instance role is limited to this project artifact prefix for EXP-01
stage-out rather than receiving broad bucket write access.

Run it locally with:

```bash
make aws-exp01-preflight
```

The current recorded EXP-01 image is:

```text
037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:c36c871dcd7e1894f6666c81280e8416c556b44d50e9b4ff5247756472dff59c
```

It was published as tag `exp01-20260714-98ed22f`; ECR reported the image as
`ACTIVE`, with scan-on-push still `IN_PROGRESS` immediately after publication.

The wrapper intentionally does not call `ec2 run-instances`, publish images, or
change AWS resources. A later launch wrapper must keep the same gates and add
explicit launch confirmation, maximum lifetime enforcement, artifact stage-out,
and termination-on-failure behavior before it is allowed to start a billable
host.

## Admission checklist

Before the first measured experiment, record and verify:

- AWS account/Region context without storing credentials in Git.
- The 96-vCPU On-Demand G/VT quota and zero reliance on P or Spot quotas.
- All planned G7e type offerings by Availability Zone and, before each
  profile's first use, a successful launch.
- Current On-Demand price and projected cost for each planned session.
- Actual GPU identity/memory and full `nvidia-smi topo -m` output.
- Driver, Docker/NVIDIA Container Toolkit, ECR pull through the instance role,
  and container smoke test.
- Peer access, NCCL, and exact one-/two-/four-GPU visibility behavior.
- DCGM, Nsight Systems, Nsight Compute, Transformer Engine, and precision paths.
- S3 stage-in/out, local-disk throughput, checksums, and recovery after failure.
- Maximum instance lifetime and verified stop/termination guard.
