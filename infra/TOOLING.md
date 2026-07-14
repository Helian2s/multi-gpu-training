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

Status: **audited with pending provider and registry checks**. Local repository
checks and input verification pass, and Docker can run native `linux/amd64`
containers with Buildx.

| Capability | Tool/status | Readiness |
| --- | --- | --- |
| OS, CPU, RAM, and storage | Ubuntu 24.04.4 LTS on x86_64; Intel Core i7-12700K with 20 logical CPUs; 31 GiB RAM and 8 GiB swap; project and Docker storage share `/dev/nvme0n1p2` with 750 GiB free of 915 GiB after local image builds | Ready for source work, input verification, and large local image pulls/builds |
| Git repository work | Git 2.43.0; checkout on `main` tracking `origin/main`; origin is `git@github.com:Helian2s/multi-gpu-training.git`; `git ls-remote origin` succeeded and GitHub SSH authentication identified `Helian2s` | Ready for local and remote Git work |
| JSON, transfers, and remote shell | `jq` 1.7, OpenSSH 9.6p1, and `rsync` 3.2.7 | Ready |
| Local entry points and analysis | GNU Make 4.3 and Python 3.12.3; `make check` passed; ignored `.venv` prepared with `requirements-preparation.txt` | Ready for current scaffold and input verification; container Python remains separately pinned |
| Native OCI image build | Docker Engine 29.1.3 daemon is running; Buildx 0.30.1 with BuildKit v0.26.2 is installed; Docker Hub manifest lookup succeeded; `docker run --rm --platform linux/amd64 alpine:3.20 uname -m` returned `x86_64`; local Docker images include pulled NGC bases and candidate project images `multi-gpu-training-pytorch:local` and `multi-gpu-training-nemo:local`, using about 64.8 GiB total | Ready for local native `linux/amd64` pulls and image builds; project GPU validation still requires a provider host |
| NVIDIA GPU runtime | `lspci` shows Intel integrated graphics only; `nvidia-smi` is not installed; no `/dev/nvidia*` devices are visible; Docker runtimes are `io.containerd.runc.v2` and `runc` only | No local CUDA validation; cloud GPU validation remains mandatory |
| NVIDIA NGC access | SSM SecureString `/finetuning/ngc/api-key` exists for NGC authentication; Docker login to `nvcr.io` using that value succeeded with a temporary Docker config; manifest access succeeded for candidate bases `nvcr.io/nvidia/pytorch:26.06-py3` (`sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`) and `nvcr.io/nvidia/nemo:26.06` (`sha256:64fcec59b0eeee2853761d16767c603e03e0aa4ba03becc9a7793bb0c46545e7`); local CPU-only inspection is recorded in `containers/base-image-compatibility.md` | Ready for NGC-derived Dockerfile design; candidate digests are verified but not yet accepted project image pins |
| AWS API and ECR | AWS CLI 2.33.27 is authenticated with profile `finetuning-local` as account `037678282394` in `us-west-2`; private ECR repositories `multi-gpu-training-pytorch` and `multi-gpu-training-nemo` exist at `037678282394.dkr.ecr.us-west-2.amazonaws.com` with immutable tags, AES256 encryption, scan-on-push, and untagged-image cleanup after 7 days; Docker ECR credential helper 0.6.4 is installed and `~/.docker/config.json` maps the project ECR registry to `ecr-login`; helper profile `finetuning-ecr-helper` uses `credential_process` to export short-lived credentials from `finetuning-local`; candidate images were pushed with immutable tag `publish-test-20260713-36621dd` and ECR digests recorded in `containers/base-image-compatibility.md`; PyTorch ECR scan completed with findings pending review, while NeMo scan was still `IN_PROGRESS`; existing EC2 instance role `FinetuningGpuInstanceRole` has inline policy `FinetuningGpuEcrPullOnly` allowing pull-only access to the two project repositories; IAM simulation allows pull actions and denies `ecr:PutImage`; old `FT-EXP-00` `g6e.2xlarge` instance `i-0c769a18f50fd1fe6` was terminated and its 100 GiB and 250 GiB EBS volumes no longer exist | Ready for local ECR push/pull mechanics; EC2 pull validation, GHCR mirror publication, scan review, and provider-side GPU qualification remain pending |
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
5. Verify ECR image pull through `FinetuningGpuInstanceRole` during AWS host
   qualification.
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
