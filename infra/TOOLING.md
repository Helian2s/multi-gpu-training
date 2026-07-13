# Tooling readiness

Last checked: 2026-07-13

This is an operational status record, not a project-decision log. The approved
toolset is maintained in `PROJECT_DECISIONS.md`; this file answers whether the
current workstation can actually use it.

## Current workstation

| Capability | Tool/status | Readiness |
| --- | --- | --- |
| Runpod API and Pod lifecycle | `runpodctl` 2.6.1; `doctor` passed API connectivity and SSH-key synchronization | Connectivity ready; paid use is blocked until the exposed key is rotated and `doctor` passes again |
| AWS API and ECR | AWS CLI is not installed | Blocked; AWS identity, Region, quota, EC2, S3, IAM, and ECR access cannot yet be verified |
| Local OCI image build | Docker CLI/engine and Buildx are not installed | Not ready locally; project images can instead be built by GitHub Actions after GitHub authentication and workflow permissions are ready |
| Git repository work | Git 2.47.0 | Ready |
| GitHub repository automation | GitHub CLI 2.96.0 is installed, but `gh auth status` reports that the active token for `Helian2s` is invalid | Blocked until `gh auth login -h github.com` succeeds |
| Local GHCR push/pull | GitHub CLI authentication is invalid; package scopes cannot be meaningfully verified yet | Blocked locally; after reauthentication, verify package read/write access or use GitHub Actions with explicit `GITHUB_TOKEN` package permissions |
| JSON, transfers, and remote shell | `jq` 1.7.1, OpenSSH, and `rsync` | Ready |
| Local entry points and analysis | GNU Make 3.81 and Python 3.14.3 | Ready for the current scaffold; container Python remains separately pinned |

## Required next checks

1. Install AWS CLI v2 and select a short-lived authentication method. Verify
   the caller identity and default Region without writing credentials to this
   repository.
2. Choose the first image-build route: install a Docker-compatible local engine
   with Buildx, or prepare the GitHub Actions build. In either case, verify a
   `linux/amd64` build because the workstation and GPU hosts may have different
   architectures.
3. Create the two private ECR repositories in `us-west-2`, an EC2 pull-only
   role, and a GitHub Actions OIDC publishing role before the first image push.
4. Reauthenticate GitHub CLI, then configure and verify GHCR package permissions
   for the mirror. Runpod receives pull-only credentials; AWS credentials are
   never stored in Runpod.
5. Verify access to the selected NGC PyTorch and NeMo base images during the
   container compatibility build.
6. On the first host from each compute profile, run qualification for the
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
