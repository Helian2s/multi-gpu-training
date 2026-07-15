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
| Native OCI image build | Docker Engine 29.1.3 daemon is running; Buildx 0.30.1 with BuildKit v0.26.2 is installed; Docker Hub manifest lookup succeeded; `docker run --rm --platform linux/amd64 alpine:3.20 uname -m` returned `x86_64`; local Docker images include pulled NGC bases, `multi-gpu-training-pytorch:local`, `multi-gpu-training-nemo:local`, A2-prep PyTorch image `multi-gpu-training-pytorch:a2-prep-b328fa3bed00` (`sha256:cc0ab368ca60de1725a483cb48f5e13241c8fc3aaa5da4d62cf64b9d57d38056`), and CUDA 12.8 Runpod NeMo image `multi-gpu-training-nemo:runpod-cuda128-20260715-0900943-dirty` (`sha256:acb063d182d4cf0d852630baa9daf2c15750533ed88dd740c41b40a05ae3db49`) with CUDA Samples revision `b7c5481c556c3fe98db060207ecaa41a4b9a9abc` constrained to `sm_80`; non-GPU container smoke passed for Megatron imports, Runpod dry-runs, Runpod queue plans, SSH daemon startup, NCCL test binaries, and `p2pBandwidthLatencyTest` presence | Ready for local native `linux/amd64` pulls and image builds; the CUDA 12.8 Runpod NeMo image is published to GHCR and passed provider-side GPU validation on Runpod A2 and A4 Pods |
| NVIDIA GPU runtime | `lspci` shows Intel integrated graphics only; `nvidia-smi` is not installed; no `/dev/nvidia*` devices are visible; Docker runtimes are `io.containerd.runc.v2` and `runc` only | No local CUDA validation; cloud GPU validation remains mandatory |
| NVIDIA NGC access | SSM SecureString `/finetuning/ngc/api-key` exists for NGC authentication; Docker login to `nvcr.io` using that value succeeded with a temporary Docker config; manifest access succeeded for candidate bases `nvcr.io/nvidia/pytorch:26.06-py3` (`sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`) and `nvcr.io/nvidia/nemo:26.06` (`sha256:64fcec59b0eeee2853761d16767c603e03e0aa4ba03becc9a7793bb0c46545e7`); local CPU-only inspection is recorded in `containers/base-image-compatibility.md` | Ready for NGC-derived Dockerfile design; candidate digests are verified but not yet accepted project image pins |
| AWS API and ECR | AWS CLI 2.33.27 is authenticated with profile `finetuning-local` as account `037678282394` in `us-west-2`; private ECR repositories `multi-gpu-training-pytorch` and `multi-gpu-training-nemo` exist with immutable tags, AES256 encryption, scan-on-push, and untagged-image cleanup after 7 days; temporary Docker configs were used for known-good ECR pushes; ECR pull-only instance-role simulation passes and denies `ecr:PutImage`; `FinetuningGpuS3Access` default version `v8` permits pinned input reads plus approved AWS artifact prefixes through EXP-12 and `QUAL-A2`; AMI is `ami-04b4c34375925db5f`, security group is `sg-0797f3b8520d4efa9`, IMDSv2 is required, and `session-manager-plugin` 1.2.835.0 is installed; after approved cleanup on 2026-07-15, read-only checks found no active/stopped project EC2 instances, no project-tagged EBS volumes, and no unattached available EBS volumes in `us-west-2` | AWS EXP-01 through EXP-09 raw measurements are staged to S3; EXP-12 completed and is reported; EXP-01 through EXP-09 reports remain pending analysis/validation; future AWS retries must recreate cache volumes |
| GitHub and GHCR | GitHub CLI 2.45.0 is authenticated as `Helian2s` through the local keyring; `gh repo view` reports `ADMIN` permission on `Helian2s/multi-gpu-training`; local `gh` auth was refreshed with package permission and now reports `write:packages`; Docker login to `ghcr.io` with the `gh` token succeeded using a temporary Docker config and was then logged out; private GHCR package `multi-gpu-training-nemo` exists with CUDA 12.8 Runpod tag `runpod-cuda128-20260715-0900943-dirty` and digest `sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b`; previous CUDA 13.2 SSH-start tag `runpod-sshd-start-20260715-1839-0900943-dirty` has digest `sha256:8a03f34f9cb8e101ba59a92f98d50ff8e2e0dfa8094dffc2fb977af26176f484`; previous wrapper-keepalive digest `sha256:bd0a9910c44ba58393eb072ae6c4b4571442812e5e65f71abed76b9123a58754` pulled but exited immediately on Runpod | GHCR publication is ready for Runpod by immutable digest |
| Runpod | `runpodctl` 2.7.1-06a0a26 is installed at `/usr/bin/runpodctl`; local config exists at `~/.runpod/config.toml` with mode `0600` and `~/.runpod` mode `0700`; `runpodctl doctor -o json` passed API-key, API-connectivity, and SSH-key sync checks on 2026-07-15 after API-key rotation; working pull-only GHCR registry auth is `readonly2` / `cmrmgu74b005rha4eqqn8bsoo`; failed A2 Pods using the non-SSH wrapper image were deleted; `RUNPOD-A2-Megatron` run `runpod-a2-megatron-20260715T203057Z` completed with status 0 on Pod `8gweq12m656oic` and was mirrored locally; `RUNPOD-A4-Megatron` run `runpod-a4-megatron-20260715T212017Z` completed with status 0 on Pod `iijfdo0p8nvbpx` and was mirrored locally; read-only account checks report no active Pods, no network volumes, `currentSpendPerHr=0`, `clientBalance=17.3833530899`, and `spendLimit=80` | Ready for read-only Runpod accounting and future approved launches; future Runpod launches should use CUDA 12.8 digest with `--min-cuda-version 12.8`, Pod volume disk mounted at `/runpod-volume`, and local artifact copy-out before Pod deletion |
| Local model and dataset inputs | `data/raw` is 3.6 GiB and `data/processed` is 550 MiB; `make verify-inputs` passed for 10 model files, 6 dataset files, and 9 processed files; the same pinned inputs are staged in S3 under `inputs/qwen3-wikitext-v1/` for AWS queues | Ready locally and staged for AWS; Runpod copies are still pending |

## Required next checks

1. Create a GitHub Actions OIDC publishing role before automated ECR image
   pushes.
2. Review the replacement shared AWS PyTorch image scan for
   `sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486`
   and record the measured-run scan disposition. The completed scan reported 62
   critical, 178 high, 250 medium, 16 low, and 4 undefined findings.
3. Analyze and validate the AWS artifacts for EXP-01 through EXP-09, write the
   experiment reports, and only then change catalog rows to `completed`.
4. On the first host from each future compute profile, run qualification for the
   NVIDIA driver, container runtime, CUDA, NCCL, DCGM, Nsight Systems, Nsight
   Compute, storage, image pull, topology, and termination guard. These tools
   cannot be validated on the non-NVIDIA local workstation.
5. Review the ECR scan disposition for the AWS-A1 queue image
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

## Current AWS-A2-Megatron queue note

On 2026-07-15, EXP-12 was prepared and run as the `AWS-A2-Megatron` exception.
Fixed `us-west-2b` and `us-west-2a` launch attempts failed before instance
creation with `InsufficientInstanceCapacity`, so retained 300 GiB gp3 cache
volumes were created in all four default AZs and the launch config now lets AWS
select default placement. The successful run launched `i-0c12083e968c55477` in
`us-west-2b` with cache volume `vol-055b18a2e1e5fdf79`. The queue ran `QUAL-A2`
inside the NeMo/Megatron image first, then `EXP-12-A2V1` and `EXP-12-A2V2`;
all three exited `0`. It uses a hard 300-minute lifetime, a 15-minute success
hold, and `stop_on_failure=false` so a launched queue failure leaves the host
available for inspection while the hard shutdown remains scheduled.

The EXP-12-capable NeMo image was pushed to ECR as tag
`aws-exp12-20260715-0338-0900943-dirty` with immutable digest
`sha256:ea7616a570d7e271eff25b4f3c0655a9910024e119171e2ead7569f6714b35fb`.
The image was built from local dirty source at Git commit `0900943`; commit the
repo changes before treating that image as a durable cross-workstation state.
ECR scan completed with 69 critical, 286 high, 479 medium, 29 low, and 133
undefined findings from the NVIDIA/Ubuntu base stack. This is acceptable only
for the short-lived lab run; refresh or patch the upstream base before any
long-lived or production use.

`FinetuningGpuS3Access` default version `v8` permits the EC2 instance role to
list and read/write `artifacts/QUAL-A2/`, `artifacts/EXP-12/`, and
`artifacts/AWS-A2-Megatron/`. IAM simulation returned `allowed` for object
read/write and scalar `s3:prefix` list checks on those prefixes.

S3 artifacts are present under `artifacts/QUAL-A2/`, `artifacts/EXP-12/`, and
`artifacts/AWS-A2-Megatron/` for run
`aws-a2-megatron-exp12-20260715T0340Z`. The local ignored mirror is under
`artifacts/runs/aws-s3-mirror/`.

After the configured success hold, instance `i-0c12083e968c55477` stopped in
`us-west-2b` and its public IP was released.

On 2026-07-15, an explicitly approved AWS cleanup terminated the remaining
project stopped instances `i-0e1acce0415f88196`, `i-0d72d0cbf38dbe44a`, and
`i-0c12083e968c55477`; their root volumes were deleted automatically. The four
300 GiB project cache volumes in `us-west-2a/b/c/d` were deleted afterward.
Read-only verification found no active/stopped project EC2 instances, no
project-tagged EBS volumes, and no unattached available EBS volumes in
`us-west-2`.

## Current Runpod preparation note

On 2026-07-14, Runpod preparation became the next active path while preserving
the technical ability to resume AWS when G7e capacity appears. On 2026-07-15,
the Runpod plan was consolidated around the NeMo/Megatron image: the active
Runpod queues are `RUNPOD-A2-Megatron` for `QUAL-RUNPOD-A2-Megatron`, EXP-10,
EXP-11, and EXP-13 on one two-GPU A100 SXM Pod, and `RUNPOD-A4-Megatron` for
EXP-14 on a separate four-GPU A100 SXM Pod. The local workstation has
`runpodctl` 2.7.1-06a0a26 installed; `runpodctl doctor -o json` passed API-key,
API-connectivity, and SSH-key sync checks after the local Runpod config
directory permissions were tightened. `runpodctl registry list -o json`
returned no registry auth entries, and `runpodctl gpu list -o json` reported
A100 SXM / `NVIDIA A100-SXM4-80GB` available in Secure Cloud with medium stock.
The NeMo/Megatron wrapper-keepalive image was published to private GHCR package
`multi-gpu-training-nemo` as tag
`runpod-wrapper-keepalive-20260715-1830-0900943-dirty` with immutable digest
`sha256:bd0a9910c44ba58393eb072ae6c4b4571442812e5e65f71abed76b9123a58754`.
Runpod pulled the digest successfully but exited the Pod immediately, which is
consistent with the image lacking a Runpod-compatible SSH startup layer. The
replacement image `runpod-sshd-start-20260715-1839-0900943-dirty` preserves
the NVIDIA base entrypoint, sets a default command that configures Runpod's
injected SSH public key, starts `sshd`, and then sleeps. Local non-GPU smoke
verified SSH daemon startup, repository presence, Megatron imports,
`all_reduce_perf`, and `p2pBandwidthLatencyTest`. The SSH-start image is
published to GHCR as immutable digest
`sha256:8a03f34f9cb8e101ba59a92f98d50ff8e2e0dfa8094dffc2fb977af26176f484`.

On 2026-07-15, UI Pod `qxin1w7sinlr7p` resumed successfully after the
replacement GHCR registry credential `readonly2` was selected. SSH worked and
the pod exposed two A100-SXM4-80GB GPUs, but `RUNPOD-A2-Megatron` qualification
failed before measurement because the host driver was 570.172.08 / CUDA 12.8
while the NeMo 26.06 image uses CUDA 13.2. The queue helper was patched to
prepend the repository root to `PYTHONPATH`, fixing the first failed attempt's
`ModuleNotFoundError: common`; the remaining blocker is host CUDA compatibility.
Failure artifacts were copied to `artifacts/runs/runpod-volume-mirror/`.

For that CUDA compatibility blocker, a local candidate image was built from
`nvcr.io/nvidia/nemo:25.04@sha256:0b981b39cb822feec53d22f789c48be8967bc48567b7b051f4d0a3ffaeb44703`.
The local tag
`multi-gpu-training-nemo:runpod-cuda128-20260715-0900943-dirty`
has image ID
`sha256:acb063d182d4cf0d852630baa9daf2c15750533ed88dd740c41b40a05ae3db49`,
reports `torch.version.cuda == 12.8`, imports NeMo and `megatron.core`, starts
the Runpod SSH keepalive command, and contains `all_reduce_perf` plus
`p2pBandwidthLatencyTest`. The CUDA Samples build was constrained to `sm_80`
for the A100 SXM target. It was published to GHCR as tag
`runpod-cuda128-20260715-0900943-dirty` with immutable digest
`sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b`;
Runpod queue YAML now references that digest and requests CUDA 12.8 hosts.

On 2026-07-15, user-created Pod `8gweq12m656oic` (`runpod-a2-migration`)
started from the CUDA 12.8 digest, exposed two A100-SXM4-80GB GPUs on driver
570.172.08, and passed the `RUNPOD-A2-Megatron` queue as run
`runpod-a2-megatron-20260715T203057Z`. Qualification, EXP-10, EXP-11 V1/V2,
and EXP-13 V1/V2 all exited with status 0. Artifacts were copied locally under
`artifacts/runs/runpod-volume-mirror/`. EXP-10 and queue artifacts were written
to `/runpod-volume`; EXP-11 and EXP-13 artifacts were recovered from the
container workspace because the queue script did not export `ARTIFACT_ROOT` and
`IMAGE_REF` to nested worker shells for this run. That bug was fixed after the
run in `infra/runpod/runpod_queue.py`; future runner manifests should record the
current image digest and write directly to `/runpod-volume`.

On 2026-07-15, user-created Pod `iijfdo0p8nvbpx` (`a4`) started from the CUDA
12.8 digest, exposed four A100-SXM4-80GB GPUs on driver `580.126.16`, and
passed the `RUNPOD-A4-Megatron` queue as run
`runpod-a4-megatron-20260715T212017Z`. Qualification and EXP-14 exited with
status 0, and artifacts were copied locally under
`artifacts/runs/runpod-volume-mirror/`. A prior diagnostic run
`runpod-a4-megatron-20260715T210918Z` wrote rank metrics but was terminated
with queue status 143 after two ranks hung during explicit NCCL cleanup. The
executor now skips `dist.destroy_process_group()` by default for short-lived
torchrun workers and relies on subprocess exit for NCCL cleanup; publish a new
immutable Runpod image with this fix before future reproducibility runs. After
artifact mirror, `runpodctl pod list -o json` initially showed Pod
`iijfdo0p8nvbpx` still running at about `$5.96/hr`; the latest read-only check
showed no active Pods or network volumes and `currentSpendPerHr=0`.

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
