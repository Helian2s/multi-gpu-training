# AWS adapter plan

AWS is the first provider to qualify. The approved quota is:

- **Region:** `us-west-2` (US West, Oregon)
- **Quota:** Running On-Demand G and VT instances
- **Limit:** 96 vCPUs

This quota does not cover P-family A100/H100 instances or Spot G instances.
VT1 shares the quota bucket but contains video-transcoding accelerators rather
than CUDA GPUs, so it is not an experiment resource.

## Accepted primary compute profiles

`AWS-A1`, `AWS-A2`, and `AWS-A4` are project profile names for AWS G7e EC2
instances. The `A` prefix means AWS, and the number is the billed physical GPU
count. These names are aliases for exact instance types; they are not AWS API
instance-family names.

| Profile | Instance type | GPUs | vCPUs | Host memory | Memory per GPU | Use |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `AWS-A1` | `g7e.2xlarge` | 1 | 8 | 64 GiB | 96 GB | One-GPU baselines and one-GPU experiments |
| `AWS-A2` | `g7e.12xlarge` | 2 | 48 | 512 GiB | 96 GB | Current AWS distributed queue; may run one- or two-visible-GPU phases |
| `AWS-A4` | `g7e.24xlarge` | 4 | 96 | 1,024 GiB | 96 GB | Not in the current queue; requires a new decision before use |

All three profiles use the RTX PRO 6000 Blackwell Server Edition. The two- and
four-GPU profiles support GPUDirect P2P over PCIe; the one-GPU profile has no
inter-GPU path. `AWS-A2V1` and `AWS-A2V2` are run-unit phase suffixes, not
separate EC2 profiles: they mean one or two visible GPUs on the same billed
`AWS-A2` host. `AWS-A4` consumes the complete approved quota, so any future
AWS-A4 launch must verify that no other G or VT instance is running before
launch. The profiles remain subject to current On-Demand price, Availability
Zone offering, capacity, permission, container, and profiler qualification.

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
- **Persistent cache:** one manually retained EBS cache volume per Availability
  Zone where we repeatedly launch AWS runs.
- **Staging:** local instance-store NVMe for performance-sensitive temporary
  data; retained EBS where persistence across terminated runs is worth its
  cost.
- **Access:** SSH or AWS Systems Manager will be chosen during qualification;
  experiment code must not depend on that choice.
- **Visibility:** use the exact `AWS-A1`, `AWS-A2`, or `AWS-A4` physical
  profile; `AWS-A2` may deliberately mask to one visible GPU for `A2V1`
  phases.
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

The current recorded EXP-01 host and image are:

```text
AMI: ami-04b4c34375925db5f
AMI name: Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) 20260710
Instance type: g7e.12xlarge
Compute profile: AWS-A2
Security group: sg-0797f3b8520d4efa9
Instance profile: FinetuningGpuInstanceRole
Root EBS: 120 GiB gp3, encrypted, delete-on-termination
Active subnet/AZ: AWS-selected default subnet/AZ
Persistent cache EBS: selected by AZ from the retained cache-volume map
Shutdown behavior: terminate
IMDS: IMDSv2 required
Image:
037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:e17de82324539ff25707ebe267dede8e70c558005c9e9f0f0c6e3dbd7f9f9d8f
```

### Persistent AWS cache volume

The AWS phase uses terminated EC2 instances for cost control, so root EBS and
instance-store NVMe do not preserve Docker layers between runs. The reusable
cache is a separate retained EBS volume:

| Storage | Size | Lifetime | Use |
| --- | ---: | --- | --- |
| Root EBS | 120 GiB gp3 | delete on termination | OS, Docker config, bootstrap only |
| Persistent cache EBS | 300 GiB gp3 | keep across AWS runs | Docker layer cache and later model/data cache for AWS PyTorch experiments |
| Instance-store NVMe | 3.8 TiB on `g7e.12xlarge` | lost on terminate | Fast temporary run scratch and profiler data |
| S3 | existing bucket | durable | Authoritative artifacts/results |

Retained cache volumes currently exist in:

| Availability Zone | Volume ID | Status |
| --- | --- | --- |
| `us-west-2a` | `vol-052b8f4246bd0d909` | Retained fallback cache volume |
| `us-west-2b` | `vol-055b18a2e1e5fdf79` | Retained fallback cache volume |
| `us-west-2c` | `vol-0189cec8b1c5bb224` | Retained fallback cache volume |
| `us-west-2d` | `vol-0746f5d3a6d2cd859` | Retained fallback cache volume |

EBS volumes are Availability-Zone scoped. EXP-01 now lets AWS select a default
subnet/AZ and then attaches the retained cache volume that matches the
instance's actual AZ. The duplicate volumes were created after transient
`g7e.12xlarge` capacity blocked fixed-AZ attempts.

After the volume exists, record its ID in `cache_volume.volume_id` in
`exp01_qualification.yaml`. The launch wrapper checks that the volume is
`available`, in the selected subnet's AZ, encrypted, `gp3`, and 300 GiB before
creating an instance. During launch it attaches the volume as `/dev/sdf`; on
Nitro hosts the guest sees it as an NVMe device, mounts it at `/mnt/aws-cache`,
and configures Docker's `data-root` as `/mnt/aws-cache/docker`.

The first EC2 launch used tag `exp01-20260714-98ed22f` at digest
`sha256:c36c871dcd7e1894f6666c81280e8416c556b44d50e9b4ff5247756472dff59c`.
It validated G7e host access, SSM, ECR pull, and S3 stage-out, but failed
before measurement because the image omitted the accepted EXP-01 directory and
could not find `collect_exp01.sh`. Do not reuse that digest for measurement.

The replacement image was published as tag `exp01-20260714-f08a362` at digest
`sha256:e17de82324539ff25707ebe267dede8e70c558005c9e9f0f0c6e3dbd7f9f9d8f`.
Local smoke checks confirmed `collect_exp01.sh`, `p2pBandwidthLatencyTest`, and
the required `nccl-tests` binaries are present. ECR reported the image as
`ACTIVE`; scan-on-push was still `IN_PROGRESS` when recorded.

The launch wrapper defaults to an AWS `RunInstances` dry run:

```bash
make aws-exp01-launch-dry-run
```

The dry run validates the EC2 request shape and permissions without creating an
instance. A real launch is refused while `safety.launch_enabled` is `false`; if
enabled in a later approved step, it also requires the exact confirmation
phrase printed by the dry run. The generated user-data script includes the
maximum lifetime guard, ECR pull, EXP-01 container execution, S3 artifact
stage-out, and shutdown. The default launch request sets
`InstanceInitiatedShutdownBehavior=terminate`, so successful or failed shutdown
terminates the instance.

Manual inspection runs use `HOLD_OPEN_ON_EXIT=1`. In that mode, the wrapper
sets `InstanceInitiatedShutdownBehavior=stop`, keeps a hard 90-minute safety
cap from each boot, stages artifacts to S3 after the container exits, keeps the
host available for 15 minutes after success or failure, and then stops the
instance. The stopped instance remains available for manual restart/debug, but
the hard cap is installed as a systemd timer so it is rearmed on every later
start.

For manual inspection of an already-running EXP-01 host, use the SSM operator
helpers:

```bash
make aws-exp01-status
make aws-exp01-monitor
make aws-exp01-logs
make aws-exp01-host-command CMD='nvidia-smi'
make aws-exp01-container-command CMD='python -c "import torch; print(torch.cuda.device_count())"'
make aws-exp01-artifacts
```

If more than one EXP-01 host is running, pass `INSTANCE_ID=i-...`. The
container helpers target the predictable container name `exp01-${RUN_ID}`; pass
`RUN_ID=...` only if the run ID is not available from the EC2 tag. Interactive
shells use Session Manager and require the local `session-manager-plugin`:

```bash
make aws-ssm-plugin-check
make aws-exp01-host-shell
make aws-exp01-container-shell
```

For planned manual debugging, run the launch dry run with
`HOLD_OPEN_ON_EXIT=1` first. If the later real launch is approved with the same
option, the host stages artifacts to S3 and then stays available for the
15-minute post-run inspection window before stopping.

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
