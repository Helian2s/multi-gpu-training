# Tooling readiness

Last checked: 2026-07-14

This is a per-workstation operational status record, not a project-decision
log. The approved toolset is maintained in `PROJECT_DECISIONS.md`; this file
answers whether each workstation can actually use it. A successful check on one
machine does not establish readiness on another machine. Never record a secret
value here.

## macOS ARM64 workstation

Role: documentation, source changes, input preparation, analysis, and
lightweight cross-platform container work. Large `linux/amd64` NVIDIA image
builds should normally move to the Ubuntu workstation.

| Capability | Tool/status | Readiness |
| --- | --- | --- |
| Runpod API and Pod lifecycle | `runpodctl` 2.6.1; `doctor` passed API connectivity and SSH-key synchronization | Connectivity ready; paid use is blocked until the exposed key is rotated and `doctor` passes again |
| AWS API and ECR | AWS CLI is not installed | Blocked; AWS identity, Region, quota, EC2, S3, IAM, and ECR access cannot yet be verified |
| Local OCI image build | Docker Desktop engine 29.6.1, Buildx 0.35.0, and BuildKit 0.31.1; `linux/amd64` Alpine emulation returned `x86_64` | Ready for local cross-platform image builds; project GPU validation still requires a provider host |
| Git repository work | Git 2.47.0 | Ready |
| GitHub repository automation | GitHub CLI 2.96.0 is authenticated as `Helian2s` through the macOS keyring | Ready for repository operations |
| Local GHCR push/pull | GitHub authentication is valid, but package read/write access has not been tested | Pending a non-production package permission check or GitHub Actions with explicit `GITHUB_TOKEN` package permissions |
| JSON, transfers, and remote shell | `jq` 1.7.1, OpenSSH, and `rsync` | Ready |
| Local entry points and analysis | GNU Make 3.81 and Python 3.14.3; isolated input preparation uses Python 3.12.5 | Ready for the current scaffold; container Python remains separately pinned |
| Local model and dataset inputs | Pinned Qwen3 model/tokenizer and WikiText-103 snapshots plus canonical token streams and checksums | Ready locally; durable S3 and Runpod copies are still pending |

## Ubuntu x86_64 workstation

Role: preferred local machine for native `linux/amd64` container pulls and
builds and other CPU/RAM/disk-intensive preparation.

Status: **audited with pending provider and registry checks**. Local repository
checks and input verification pass, and Docker can run native `linux/amd64`
containers with Buildx.

| Capability | Tool/status | Readiness |
| --- | --- | --- |
| OS, CPU, RAM, and storage | Ubuntu 24.04.4 LTS on x86_64; Intel Core i7-12700K with 20 logical CPUs; 31 GiB RAM and 8 GiB swap; project and Docker storage share `/dev/nvme0n1p2` with 749 GiB free of 915 GiB after local image builds | Ready for source work, input verification, and large local image pulls/builds |
| Git repository work | Git 2.43.0; checkout on `main` tracking `origin/main`; origin is `git@github.com:Helian2s/multi-gpu-training.git`; `git ls-remote origin` succeeded and GitHub SSH authentication identified `Helian2s` | Ready for local and remote Git work |
| JSON, transfers, and remote shell | `jq` 1.7, OpenSSH 9.6p1, and `rsync` 3.2.7 | Ready |
| Local entry points and analysis | GNU Make 4.3 and Python 3.12.3; `make check` passed; ignored `.venv` prepared with `requirements-preparation.txt` | Ready for current scaffold and input verification; container Python remains separately pinned |
| Native OCI image build | Docker Engine 29.1.3 daemon is running; Buildx 0.30.1 with BuildKit v0.26.2 is installed; Docker Hub manifest lookup succeeded; `docker run --rm --platform linux/amd64 alpine:3.20 uname -m` returned `x86_64`; local Docker images include pulled NGC bases, `multi-gpu-training-pytorch:local`, `multi-gpu-training-nemo:local`, and A2-prep PyTorch image `multi-gpu-training-pytorch:a2-prep-b328fa3bed00` (`sha256:cc0ab368ca60de1725a483cb48f5e13241c8fc3aaa5da4d62cf64b9d57d38056`); non-GPU container smoke passed for Python imports and `EXP-08` dry-run planning; Docker reports 65.28 GiB of images | Ready for local native `linux/amd64` pulls and image builds; project GPU validation still requires a provider host |
| NVIDIA GPU runtime | `lspci` shows Intel integrated graphics only; `nvidia-smi` is not installed; no `/dev/nvidia*` devices are visible; Docker runtimes are `io.containerd.runc.v2` and `runc` only | No local CUDA validation; cloud GPU validation remains mandatory |
| NVIDIA NGC access | SSM SecureString `/finetuning/ngc/api-key` exists for NGC authentication; Docker login to `nvcr.io` using that value succeeded with a temporary Docker config; manifest access succeeded for candidate bases `nvcr.io/nvidia/pytorch:26.06-py3` (`sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`) and `nvcr.io/nvidia/nemo:26.06` (`sha256:64fcec59b0eeee2853761d16767c603e03e0aa4ba03becc9a7793bb0c46545e7`); local CPU-only inspection is recorded in `containers/base-image-compatibility.md` | Ready for NGC-derived Dockerfile design; candidate digests are verified but not yet accepted project image pins |
| AWS API and ECR | AWS CLI 2.33.27 is authenticated with profile `finetuning-local` as account `037678282394` in `us-west-2`; private ECR repositories `multi-gpu-training-pytorch` and `multi-gpu-training-nemo` exist at `037678282394.dkr.ecr.us-west-2.amazonaws.com` with immutable tags, AES256 encryption, scan-on-push, and untagged-image cleanup after 7 days; Docker ECR credential helper 0.6.4 is installed and `~/.docker/config.json` maps the project ECR registry to `ecr-login`, but this helper failed with the current AWS login credential source during manual push and a temporary Docker config was used instead; candidate images were pushed with immutable tag `publish-test-20260713-36621dd` and ECR digests recorded in `containers/base-image-compatibility.md`; PyTorch and NeMo ECR scans completed with findings pending review; failed EXP-01 image `exp01-20260714-98ed22f` remains in ECR at digest `sha256:c36c871dcd7e1894f6666c81280e8416c556b44d50e9b4ff5247756472dff59c` and must not be reused for measurement; replacement EXP-01 image `exp01-20260714-f08a362` was pushed to ECR with digest `sha256:e17de82324539ff25707ebe267dede8e70c558005c9e9f0f0c6e3dbd7f9f9d8f`, status `ACTIVE`, and scan-on-push `IN_PROGRESS` when recorded; existing EC2 instance role `FinetuningGpuInstanceRole` has inline policy `FinetuningGpuEcrPullOnly` allowing pull-only access to the two project repositories; IAM simulation allows ECR authorization-token and pull actions and denies `ecr:PutImage`; `FinetuningGpuS3Access` default version `v3` permits `FinetuningGpuInstanceRole` to list/read/write only the `artifacts/EXP-01/` project artifact prefix in the existing S3 bucket; EXP-01 preflight passes with pinned AMI `ami-04b4c34375925db5f`, default public subnets, no-ingress security group `sg-0797f3b8520d4efa9`, IMDSv2 required, shutdown behavior set to terminate, and the replacement image digest; persistent AWS cache EBS volumes exist in all four `us-west-2` AZs as 300 GiB encrypted `gp3` with default 3000 IOPS and 125 MiB/s throughput, tagged `DeletePolicy=manual`: `us-west-2a` `vol-052b8f4246bd0d909`, `us-west-2b` `vol-055b18a2e1e5fdf79`, `us-west-2c` `vol-0189cec8b1c5bb224`, and `us-west-2d` `vol-0746f5d3a6d2cd859`; EXP-01 launch now lets AWS select the default subnet/AZ and attaches the cache volume matching the instance placement; the first EXP-01 EC2 launch used `g7e.12xlarge` in `us-west-2b`, validated SSM, two visible RTX PRO 6000 GPUs, ECR login/pull through the instance role, and S3 stage-out, then terminated itself; that run failed before measurement because the recorded image digest omitted the accepted EXP-01 directory and could not find `collect_exp01.sh`; EXP-01 SSM operator helpers are configured for status, one-off host/container commands, logs, monitoring, artifact listing, and interactive host/container shells; local `session-manager-plugin` 1.2.835.0 is installed; old `FT-EXP-00` `g6e.2xlarge` instance `i-0c769a18f50fd1fe6` was terminated and its 100 GiB and 250 GiB EBS volumes no longer exist | Host/IAM/ECR/S3/cache-volume qualification partially validated; an explicitly approved launch is required to verify replacement image execution and collect EXP-01 measurement |
| GitHub and GHCR | GitHub CLI 2.45.0 is authenticated as `Helian2s` through the local keyring; `gh repo view` reports `ADMIN` permission on `Helian2s/multi-gpu-training`; Docker login to `ghcr.io` with the `gh` token succeeded and was then removed; `gh api /user/packages?package_type=container` failed because the token lacks `read:packages` | Ready for repository automation and basic GHCR connectivity; package read/write requires package scopes or GitHub Actions package permissions |
| Runpod | `runpodctl` is not installed | Optional until a Runpod lifecycle task is assigned here; paid use remains blocked until the exposed key is rotated |
| Local model and dataset inputs | `data/raw` is 3.6 GiB and `data/processed` is 550 MiB; `make verify-inputs` passed for 10 model files, 6 dataset files, and 9 processed files | Ready locally; durable S3 and Runpod copies are still pending |

## Required next checks

1. Add or use GitHub package permissions before GHCR package checks: either
   refresh the local `gh` token with package scopes or use GitHub Actions with
   explicit package permissions.
2. Create a GitHub Actions OIDC publishing role before automated ECR image
   pushes.
3. Configure and verify GHCR package permissions for the mirror. Runpod receives
   pull-only credentials; AWS credentials are never stored in Runpod.
4. Upload the pinned input snapshots and generated manifest to versioned S3 and
   Runpod network-volume paths, then verify their checksums.
5. Verify the replacement image pull and container entry command during the
   next AWS host qualification.
6. After explicit approval, push `multi-gpu-training-pytorch:a2-prep-b328fa3bed00`
   to ECR, record its immutable digest, and update accepted experiment manifests
   before any EXP-02/07/08/09 measured run.
7. Record the EXP-01 ECR scan finding review disposition before treating the
   image as accepted beyond qualification.
8. On the first host from each compute profile, run qualification for the
   NVIDIA driver, container runtime, CUDA, NCCL, DCGM, Nsight Systems, Nsight
   Compute, storage, image pull, topology, and termination guard. These tools
   cannot be validated on the non-NVIDIA local workstation.
9. During the next explicitly approved EXP-01 launch, verify that the active
   AZ-matched cache volume attaches to the host, mounts at `/mnt/aws-cache`,
   and becomes Docker's data root at `/mnt/aws-cache/docker`.
10. For manual EXP-01 launches, verify the new lifecycle policy: capacity
   failures before instance creation leave no instance, launched instances use
   `InstanceInitiatedShutdownBehavior=stop`, the hard 90-minute systemd safety
   timer is active from each boot, and success or failure gets a 15-minute
   post-run inspection window before the instance stops.

## Current AWS capacity note

On 2026-07-14, an EXP-01 retry using the fixed image and the `us-west-2b`
cache-volume placement was blocked by transient EC2 capacity:
`RunInstances` returned `InsufficientInstanceCapacity` for `g7e.12xlarge` in
`us-west-2b` and did not create a second instance. After the manual-run policy
was changed to stop rather than terminate, a second retry returned the same
capacity error and also created no instance. A prior launch attempt in the same
session created `i-08563b4ece80877cf` but failed before cache attach because
the wrapper attached while the instance was still `pending`; the wrapper
requested termination, and the instance is now `terminated`. The persistent
cache volume `vol-055b18a2e1e5fdf79` remains `available`.

The active EXP-01 placement now lets AWS select the default subnet/AZ and then
attaches the matching retained cache volume. Fixed-AZ retries in `us-west-2a`,
`us-west-2b`, `us-west-2c`, and `us-west-2d` returned
`InsufficientInstanceCapacity`; the AWS-selected placement retry also returned
`Insufficient capacity`. A later AWS-selected AWS-A2 retry returned the same
capacity error. None of those capacity failures created an instance.

## Security action

A Runpod API key was pasted into chat during initial setup. Rotate that key in
Runpod before any paid experiment, update the local Runpod configuration, and
repeat `runpodctl doctor`. Never store the replacement key in Git, YAML, shell
history, logs, or experiment artifacts.

## Readiness gate

No paid AWS or Runpod resource should be launched until the provider adapter can
perform a non-mutating identity/status check and the launch configuration has a
maximum lifetime, maximum cost, durable stage-out target, and explicit
termination path.
