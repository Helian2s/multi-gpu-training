# Tooling readiness

Last checked: 2026-07-15

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
| AWS API and ECR | AWS CLI 2.33.27 is authenticated with profile `finetuning-local` as account `037678282394` in `us-west-2`; private ECR repositories `multi-gpu-training-pytorch` and `multi-gpu-training-nemo` exist at `037678282394.dkr.ecr.us-west-2.amazonaws.com` with immutable tags, AES256 encryption, scan-on-push, and untagged-image cleanup after 7 days; Docker ECR credential helper 0.6.4 is installed, but temporary Docker configs were used for known-good pushes; ECR pull-only instance-role simulation passes and denies `ecr:PutImage`; `FinetuningGpuS3Access` default version `v7` permits the instance role to read pinned inputs, write approved EXP/queue artifact prefixes, and write AWS-A1/A2 queue logs; the active shared AWS PyTorch queue image is `multi-gpu-training-pytorch@sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486`; it was pushed as `aws-a2-fp16fix-20260715-0230-4aa4672-dirty` after fixing FP16 AMP parameter precision; ECR scan completed with 62 critical, 178 high, 250 medium, 16 low, and 4 undefined findings, and measured-run disposition remains pending review; current A2 launch config is fixed to `us-west-2b`, subnet `subnet-0d50d4374d2149a57`, AMI `ami-04b4c34375925db5f`, no-ingress security group `sg-0797f3b8520d4efa9`, IMDSv2 required, and retained cache volume `vol-055b18a2e1e5fdf79`; local `session-manager-plugin` 1.2.835.0 is installed; old `FT-EXP-00` `g6e.2xlarge` instance `i-0c769a18f50fd1fe6` was terminated and its 100 GiB and 250 GiB EBS volumes no longer exist | AWS EXP-01 through EXP-09 raw measurements have been staged to S3; reports and completed status remain pending analysis/validation |
| GitHub and GHCR | GitHub CLI 2.45.0 is authenticated as `Helian2s` through the local keyring; `gh repo view` reports `ADMIN` permission on `Helian2s/multi-gpu-training`; local `gh` auth was refreshed with package permission and now reports `write:packages`; `gh api /user/packages?package_type=container` succeeds and returns no container packages; Docker login to `ghcr.io` with the `gh` token succeeded using a temporary Docker config and was then logged out; no `multi-gpu-training-pytorch` or `multi-gpu-training-nemo` GHCR package exists yet | Ready for local GHCR publication checks; GHCR mirror images still need to be pushed and recorded by digest; Runpod still needs a separate pull-only GHCR credential rather than the broad local `gh` token |
| Runpod | `runpodctl` 2.7.1-06a0a26 is installed at `/usr/bin/runpodctl`; local config exists at `~/.runpod/config.toml` with mode `0600` and `~/.runpod` mode `0700`; `runpodctl doctor -o json` passed API-key, API-connectivity, and SSH-key sync checks; `runpodctl registry list -o json` returned no registry auth entries; the active Runpod queue model is `RUNPOD-A1-PyTorch` for one-visible-GPU PyTorch smoke work, `RUNPOD-A2-PyTorch` for two-GPU readiness/communication/EXP-10, `RUNPOD-A2-Megatron` for one- and two-visible-GPU EXP-11 through EXP-13 work, and `RUNPOD-A4-Megatron` for EXP-14 | Ready for non-mutating Runpod identity/status checks; paid use remains blocked until the old exposed key revocation is confirmed, a pull-only GHCR registry auth is added, GHCR mirror images exist, and durable storage/stage-out is planned |
| Local model and dataset inputs | `data/raw` is 3.6 GiB and `data/processed` is 550 MiB; `make verify-inputs` passed for 10 model files, 6 dataset files, and 9 processed files; the same pinned inputs are staged in S3 under `inputs/qwen3-wikitext-v1/` for AWS queues | Ready locally and staged for AWS; Runpod copies are still pending |

## Required next checks

1. Publish the required GHCR mirror images and record immutable GHCR digests.
2. Create a GitHub Actions OIDC publishing role before automated ECR image
   pushes.
3. Confirm in the Runpod console that the old exposed API key has been revoked;
   local `runpodctl` is installed and `doctor` passes with a configured key.
4. Create a pull-only GHCR credential for Runpod and add it with
   `runpodctl registry create`; do not store the broad local `gh` token or any
   AWS credentials in Runpod.
5. Upload the pinned input snapshots and generated manifest to Runpod
   network-volume paths, then verify their checksums; the AWS S3 copy is
   staged under `inputs/qwen3-wikitext-v1/`.
6. Review the replacement shared AWS PyTorch image scan for
   `sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486`
   and record the measured-run scan disposition. The completed scan reported 62
   critical, 178 high, 250 medium, 16 low, and 4 undefined findings.
7. Analyze and validate the AWS artifacts for EXP-01 through EXP-09, write the
   experiment reports, and only then change catalog rows to `completed`.
8. On the first host from each future compute profile, run qualification for the
   NVIDIA driver, container runtime, CUDA, NCCL, DCGM, Nsight Systems, Nsight
   Compute, storage, image pull, topology, and termination guard. These tools
   cannot be validated on the non-NVIDIA local workstation.
9. Review the ECR scan disposition for the AWS-A1 queue image
   `sha256:e12af417e7e905f30182122a95d73610e3acc9cb41829093d0265dfd6cca4225`.
   The completed scan reported 62 critical, 178 high, 248 medium, 16 low, and
   4 undefined findings.

## Current AWS-A1 note

On 2026-07-14, local AWS-A1 preparation added
`infra/aws/a1_qualification.yaml` for `g7e.2xlarge` with one visible RTX PRO
6000 GPU, the shared AWS PyTorch image digest
`sha256:ffde9efc9d69ea98fb4da0bb22736a7c7efdee9f72a21e825b6aa51377892bb8`,
the retained cache-volume map, a one-GPU CUDA/NCCL smoke command, a 60-minute
hard lifetime, and a 15-minute hold-open inspection mode. Read-only preflight
confirmed the AWS identity, 96-vCPU G/VT quota with 0 active G/VT vCPUs,
`g7e.2xlarge` shape of 8 vCPUs and one RTX PRO Server 6000 GPU, offerings in
all four `us-west-2` AZs, current On-Demand price of 3.36312 USD/hr, the pinned
AMI, no-ingress security group, retained cache volume state, ECR image and
pull-only role permissions. The shared PyTorch image ECR scan disposition is
recorded for short-lived qualification smoke only. EC2 `RunInstances` dry-run
is authorized with confirmation phrase `launch QUAL-A1 AWS-A1 stop-after-60m`.
After the approved S3 policy update to `FinetuningGpuS3Access` default version
`v5`, A1 preflight passes: IAM simulation allows listing and object
read/write/multipart actions under `artifacts/QUAL-A1/` and still denies
EXP-03. A1 SSM operator helpers are configured for status, one-off
host/container commands, logs, monitoring, artifact listing, and interactive
host/container shells using the `qual-a1-${RUN_ID}` container name. An approved
real sequential capacity probe on 2026-07-14 pinned `g7e.2xlarge` launch
requests to `us-west-2a`, `us-west-2b`, `us-west-2c`, and `us-west-2d`; every
request failed at `RunInstances` with `InsufficientInstanceCapacity`. No
`QUAL-A1` instance was created, and all four retained cache volumes remained
`available` and unattached afterward.

On 2026-07-14/15, a later sequential AWS-A1 probe found `g7e.2xlarge` capacity
in `us-west-2b` and launched `i-0e1acce0415f88196` with run ID
`capacity-20260714T235106Z-aws-a1-us-west-2b`; the instance reached SSM online
with one visible RTX PRO 6000 Blackwell GPU and `InstanceInitiatedShutdownBehavior=stop`.
The launch bootstrap failed before cache mount because generated user-data had
indented heredoc delimiters; a manual safety stop was scheduled for
2026-07-15 00:45:12 UTC, and the launcher/user-data tests were fixed to catch
that class of error.

EXP-03 through EXP-06 are accepted for the AWS-A1 PyTorch queue. The queue image
was built from the dirty worktree as
`multi-gpu-training-pytorch:aws-a1-prep-local`, pushed to ECR as immutable tag
`aws-a1-queue-20260715-0018-4aa4672-dirty`, and recorded in
`infra/aws/a1_experiment_queue.yaml` at digest
`sha256:e12af417e7e905f30182122a95d73610e3acc9cb41829093d0265dfd6cca4225`.
ECR scan completed with 62 critical, 178 high, 248 medium, 16 low, and
4 undefined findings; measured-run disposition remains pending review.
`FinetuningGpuS3Access` default version `v6` permits the instance role to read
`inputs/qwen3-wikitext-v1/`, write `artifacts/EXP-03/` through
`artifacts/EXP-06/`, and write queue logs under `artifacts/AWS-A1-PyTorch/`;
IAM simulation allowed the new list/read/write actions and still denies input
writes. The pinned local model, raw dataset, and processed token stream were
uploaded to S3 under `inputs/qwen3-wikitext-v1/data/`, and the processed
manifest is present. The AWS-A1 queue run
`aws-a1-pytorch-20260715T003637Z` completed EXP-03, EXP-04, EXP-05, and EXP-06
with exit status 0 and staged artifacts under the corresponding S3
`artifacts/EXP-NN/runs/` prefixes.

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
capacity error. An explicitly approved hold-open AWS-A2 smoke launch attempt
with confirmation phrase `launch EXP-01 AWS-A2 stop-after-90m` also returned
`Insufficient capacity` before instance creation. None of those capacity
failures created an instance; the latest post-check found no active
Codex-managed instance and all four retained cache volumes remained
`available`. A later explicit fixed-AZ retry attempted `g7e.12xlarge` in
`us-west-2d`, `us-west-2c`, `us-west-2a`, and `us-west-2b`; each attempt
returned `InsufficientInstanceCapacity` before instance creation. The
post-check again found no active Codex-managed instance, all retained cache
volumes remained `available`, and the G/VT quota was still 96 vCPUs.

## Current AWS-A2 queue note

On 2026-07-15, the AWS-A2 PyTorch completion queue was prepared for
`EXP-01-A2`, `EXP-02-A2V1`, `EXP-02-A2V2`, `EXP-07-A2V1`, `EXP-07-A2V2`,
`EXP-08-A2V2`, `EXP-09-A2V1`, and `EXP-09-A2V2`. The queue uses ECR image digest
`sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486`,
S3 input prefix `inputs/qwen3-wikitext-v1/`, queue log prefix
`artifacts/AWS-A2-PyTorch/`, and the retained `us-west-2b` cache volume
`vol-055b18a2e1e5fdf79`.

An approved fixed-`us-west-2b` launch with run ID
`aws-a2-pytorch-20260715T013926Z` created `i-0d72d0cbf38dbe44a`
(`g7e.12xlarge`) and attached `vol-055b18a2e1e5fdf79`. The host reached SSM
online and pulled the ECR image, but the queue failed before measurement at
`EXP-01-A2` because generated user-data wrote run units through a TSV file and
lost the shell command JSON fields. The queue log was uploaded to
`s3://finetuning-lab-1-037678282394-us-west-2-an/artifacts/AWS-A2-PyTorch/runs/aws-a2-pytorch-20260715T013926Z/queue.log`.
The local queue generator now emits shell-quoted `run_queue_unit` calls instead
of the TSV loop, and `make check` covers the regression. The instance stopped
through the configured guard; its 120 GiB root volume and the retained 300 GiB
cache volume remained attached. The next retry included `EXP-08-A2V2` and had
`stop_on_failure=false`, so a queue failure would upload logs and leave the
instance running for manual inspection while the hard 300-minute safety
shutdown remained scheduled.

The approved retry with run ID `aws-a2-full-fp16fix-20260715T023252Z` completed
all A2 run units with exit status 0: `EXP-01-A2`, `EXP-02-A2V1`,
`EXP-02-A2V2`, `EXP-07-A2V1`, `EXP-07-A2V2`, `EXP-08-A2V2`, `EXP-09-A2V1`,
and `EXP-09-A2V2`. S3 artifacts are present under each corresponding
`artifacts/EXP-NN/runs/aws-a2-full-fp16fix-20260715T023252Z/` prefix and the
queue log is under
`artifacts/AWS-A2-PyTorch/runs/aws-a2-full-fp16fix-20260715T023252Z/`. The run
used the replacement image because the first retry failed at `EXP-02-A2V1`
when FP16 training loaded model parameters as FP16 while also enabling
`GradScaler`; the executor now keeps FP16+GradScaler model parameters in FP32
and uses FP16 autocast. After the configured 15-minute success hold, instance
`i-0d72d0cbf38dbe44a` stopped successfully in `us-west-2b`; its public IP was
released. A local ignored mirror of the successful AWS artifacts exists at
`artifacts/runs/aws-s3-mirror/`.

## Current Runpod preparation note

On 2026-07-14, Runpod preparation became the next active path while preserving
the technical ability to resume AWS when G7e capacity appears. The active
Runpod queues are `RUNPOD-A1-PyTorch` for one-visible-GPU PyTorch smoke work,
`RUNPOD-A2-PyTorch` for credential/tooling/storage/registry readiness plus the
two-GPU NVLink/NCCL baseline, `RUNPOD-A2-Megatron` for one- and
two-visible-GPU EXP-11 through EXP-13 NeMo/Megatron phases, and
`RUNPOD-A4-Megatron` for the EXP-14 four-GPU hybrid. The local workstation has
`runpodctl` 2.7.1-06a0a26 installed; `runpodctl doctor -o json` passed API-key,
API-connectivity, and SSH-key sync checks after the local Runpod config
directory permissions were tightened. No paid Runpod resource has been launched
from this workstation in the current session.

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
