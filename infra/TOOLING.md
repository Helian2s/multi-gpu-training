# Tooling readiness

Last checked: 2026-07-13

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

Status: **not yet audited**. The first session on this workstation must replace
the unknown entries with observed facts; it must not copy readiness from macOS.

| Capability | Tool/status | Readiness |
| --- | --- | --- |
| OS, CPU, RAM, and storage | Exact Ubuntu release, Intel CPU, RAM, and free disk are not recorded | Pending local audit |
| Git repository work | Version and authentication are not recorded | Pending local audit |
| Native OCI image build | Docker Engine, Buildx, BuildKit, and `linux/amd64` behavior are not recorded | Pending local audit |
| NVIDIA GPU runtime | Local GPU and NVIDIA Container Toolkit availability are not known | Pending local audit; cloud GPU validation remains mandatory |
| NVIDIA NGC access | Registry authentication and base-image pulls are not tested | Pending only when image work is approved |
| AWS API and ECR | AWS CLI, identity, Region, IAM, S3, and ECR access are not recorded | Pending local audit |
| GitHub and GHCR | GitHub CLI and package read/write access are not recorded | Pending local audit |
| Runpod | CLI and API connectivity are not recorded | Optional until a Runpod lifecycle task is assigned here |
| Local model and dataset inputs | Local snapshots and canonical token streams are not recorded | Reproduce or transfer with manifest verification when needed |

## Required next checks

1. Audit the Ubuntu x86_64 workstation according to `WORKFLOW.md` and fill its
   readiness table without recording credentials.
2. Install AWS CLI v2 and select a short-lived authentication method. Verify
   the caller identity and default Region without writing credentials to this
   repository.
3. Build the first pinned project image for `linux/amd64` and verify its content
   locally before publishing it.
4. Create the two private ECR repositories in `us-west-2`, an EC2 pull-only
   role, and a GitHub Actions OIDC publishing role before the first image push.
5. Configure and verify GHCR package permissions for the mirror. Runpod receives
   pull-only credentials; AWS credentials are never stored in Runpod.
6. Verify access to the selected NGC PyTorch and NeMo base images during the
   container compatibility build.
7. Upload the pinned input snapshots and generated manifest to versioned S3 and
   Runpod network-volume paths, then verify their checksums.
8. On the first host from each compute profile, run qualification for the
   NVIDIA driver, container runtime, CUDA, NCCL, DCGM, Nsight Systems, Nsight
   Compute, storage, image pull, topology, and termination guard. These tools
   cannot be validated on the non-NVIDIA local workstation.

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
