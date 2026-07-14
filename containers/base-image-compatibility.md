# Base Image Compatibility Record

Checked: 2026-07-14

This record captures local, CPU-only compatibility checks on the Ubuntu
x86_64 workstation before adding project Dockerfiles. It does not replace the
mandatory provider-side GPU qualification for CUDA, NCCL, profilers, topology,
or driver compatibility.

## Candidate Images

| Image family | Candidate tag | Manifest digest | linux/amd64 digest | Local tag |
| --- | --- | --- | --- | --- |
| PyTorch | `nvcr.io/nvidia/pytorch:26.06-py3` | `sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1` | `sha256:abd110b23600e877173dafc3078385b7c13ddacd7e0c6a6acb0a864586d59622` | `ncp-genl-base/pytorch:26.06-py3-amd64` |
| NeMo/Megatron | `nvcr.io/nvidia/nemo:26.06` | `sha256:64fcec59b0eeee2853761d16767c603e03e0aa4ba03becc9a7793bb0c46545e7` | `sha256:bb1dbe94646d5a6490570823cafa0d6f753e1cb60df8f5e89e3b32f3f87893fc` | `ncp-genl-base/nemo:26.06-amd64` |

Both images were pulled by the `linux/amd64` digest and tagged locally for
inspection. The local tags are convenience aliases only; project Dockerfiles
must use the immutable source digest.

## Local Inventory

| Check | PyTorch image | NeMo/Megatron image |
| --- | --- | --- |
| OS | Ubuntu 24.04.4 LTS | Ubuntu 24.04.4 LTS |
| Python | 3.12.3 | 3.12.3 |
| PyTorch | `2.13.0a0+8145d630e8.nv26.06` | `2.12.0a0+0291f960b6.nv26.04.48445190` |
| `torch.version.cuda` | 13.3 | 13.2 |
| `nvcc` | CUDA 13.3, `V13.3.33` | CUDA 13.2, `V13.2.78` |
| NCCL env | `2.30.4` | `2.29.7` |
| Transformer Engine import | `2.16.0+4220403e` | `2.16.0+4220403e` |
| Nsight Systems | `2026.3.1.117` | `2026.2.1.210` |
| Nsight Compute | `2026.2.0.0` | `2026.1.1.0` |
| CPU torch forward/backward | Passed | Passed |

`torch.cuda.is_available()` returned `False` in both images, as expected on the
non-NVIDIA local workstation.

## Python Package Findings

### PyTorch Image

The PyTorch image includes the NVIDIA compute stack, Apex, DALI, TensorRT,
ModelOpt, tokenizers, datasets, safetensors, pandas, and NumPy. It does not
ship the full accepted workload package set by default; notably,
`transformers` is absent. A dry-run install of `requirements-preparation.txt`
completed without resolver failure and would add or update the missing workload
packages.

### NeMo/Megatron Image

The NeMo image includes:

- `NeMo-FW 26.4`
- `megatron-core 0.18.0`
- `megatron-bridge 0.5.0`
- `megatron-energon 7.3.2`
- `transformers 5.12.0`
- `tokenizers 0.22.2`
- `datasets 4.8.4`

Plain `import nemo` fails because `/opt/NeMo` is not on the default Python
path. With `PYTHONPATH=/opt/NeMo:/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM`,
imports for `nemo`, `megatron.core`, and `megatron.bridge` succeed. The project
NeMo Dockerfile or launcher should make this path requirement explicit before
any NeMo experiment is accepted.

The NeMo image emits warnings on the local non-GPU workstation during import,
including no active CUDA driver, Triton/vLLM CUDA extension warnings, and a
`pynvml` deprecation warning from `torch.cuda`. These are expected for local
CPU-only inspection and must be re-evaluated on the provider GPU host.

A dry-run install of `requirements-preparation.txt` completed without resolver
failure, but it would change framework-adjacent packages such as
`transformers`, `datasets`, `numpy`, and `pandas`. The NeMo runtime dependency
lock should therefore be narrower than the local preparation environment and
should avoid overriding NVIDIA-pinned framework packages unless a compatibility
test requires it.

## Verdict

Both candidate bases are locally usable and suitable for minimal project image
builds. They are not yet final experiment image pins.

## Local Project Image Builds

Minimal project Dockerfiles were added after the base-image inspection:

| Image family | Local project tag | Dockerfile | Local smoke result |
| --- | --- | --- | --- |
| PyTorch | `multi-gpu-training-pytorch:local` | `containers/pytorch/Dockerfile` | Project scripts compile; `torch`, `transformers`, `tokenizers`, `datasets`, `safetensors`, and `huggingface_hub` import; CPU forward/backward passes |
| NeMo/Megatron | `multi-gpu-training-nemo:local` | `containers/nemo/Dockerfile` | Project scripts compile; explicit `PYTHONPATH` exposes `nemo`, `megatron.core`, and `megatron.bridge`; CPU forward/backward passes |

The PyTorch Dockerfile installs only the minimal Hugging Face runtime packages
needed on top of the NGC base and builds CUDA Samples
`p2pBandwidthLatencyTest` from NVIDIA CUDA Samples commit
`b7c5481c556c3fe98db060207ecaa41a4b9a9abc` for EXP-01. The NeMo Dockerfile
does not install additional Python packages; it preserves the NVIDIA-pinned
NeMo/Megatron stack and makes the required source paths explicit.

An EXP-01-capable local PyTorch image was built as
`multi-gpu-training-pytorch:exp01-local` with local image ID
`sha256:b419251cdd5b503de26e242dab681364489329253421e77814dde4e75df8cfec`.
It contains `/usr/local/bin/p2pBandwidthLatencyTest`, `nccl-tests`, Nsight
tools, and the existing PyTorch runtime packages. Running the CUDA sample on
the non-NVIDIA local workstation still fails with a missing/insufficient CUDA
driver, which is expected; GPU execution remains provider-side qualification.

After the EXP-01 source preparation commit, the PyTorch image was rebuilt from
Git commit `98ed22f8e54a` and published to ECR for AWS-side qualification.

| Image family | Local tag | ECR tag | ECR digest | Pushed | Scan state |
| --- | --- | --- | --- | --- | --- |
| PyTorch EXP-01 | `multi-gpu-training-pytorch:exp01-20260714-98ed22f` | `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch:exp01-20260714-98ed22f` | `sha256:c36c871dcd7e1894f6666c81280e8416c556b44d50e9b4ff5247756472dff59c` | `2026-07-14T01:14:55Z` | `COMPLETE`: 60 critical, 178 high, 236 medium, 14 low, 4 undefined |
| PyTorch EXP-01 replacement | `multi-gpu-training-pytorch:exp01-scriptfix-f08a362` | `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch:exp01-20260714-f08a362` | `sha256:e17de82324539ff25707ebe267dede8e70c558005c9e9f0f0c6e3dbd7f9f9d8f` | `2026-07-14T16:20:36Z` | `COMPLETE`: 62 critical, 178 high, 248 medium, 16 low, 4 undefined |
| Shared AWS PyTorch | `multi-gpu-training-pytorch:a2-prep-b328fa3bed00` | `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch:a2-prep-b328fa3bed00` | `sha256:ffde9efc9d69ea98fb4da0bb22736a7c7efdee9f72a21e825b6aa51377892bb8` | `2026-07-14T20:28:01Z` | `COMPLETE`: 62 critical, 178 high, 248 medium, 16 low, 4 undefined |

The image status was `ACTIVE` after push. ECR scan-on-push findings for the
replacement and shared AWS PyTorch images are package-level findings in the
NVIDIA-derived Ubuntu/runtime stack, with sampled Critical/High packages
including `linux-libc-dev`, `curl`/`libcurl`, `gnutls`, `openssl`,
`python3.12`, `vim`, `python-pip`, `libxml2`, and `rapidjson`. The scan does
not report findings in this repository's Python source code, but it is also not
a source-code security audit.

Disposition for AWS qualification smoke on 2026-07-14: use of the shared AWS
PyTorch digest
`sha256:ffde9efc9d69ea98fb4da0bb22736a7c7efdee9f72a21e825b6aa51377892bb8` is
accepted for short-lived `QUAL-A1`/`QUAL-A2` smoke runs only, under the current
controls: private ECR, no-ingress security group, SSM-only operator access,
least-privilege ECR pull and scoped S3 artifact permissions, no production
traffic, no long-running service, and automatic stop/termination guards.
Measured experiment runs still require either a rebuilt/refreshed image with
reviewed scan results or an explicit measured-run exception recorded before
launch.

A local smoke check confirmed `p2pBandwidthLatencyTest` and `all_reduce_perf`
are on `PATH`; executing the CUDA sample still fails locally with the expected
missing/insufficient CUDA driver error.

The first EC2 launch using the `98ed22f` digest pulled successfully through
`FinetuningGpuInstanceRole` and validated the G7e host driver/GPU visibility,
but the container exited before running EXP-01 because this image omitted the
accepted experiment directory and therefore did not contain
`experiments/exp_01_aws_pcie_p2p_nccl_communication/collect_exp01.sh`. Do not
reuse that digest for EXP-01 measurement.

The replacement `f08a362` image copies accepted experiment implementations into
the image. A local container smoke check confirmed `collect_exp01.sh`,
`p2pBandwidthLatencyTest`, `all_reduce_perf`, `reduce_scatter_perf`,
`all_gather_perf`, `broadcast_perf`, and `alltoall_perf` are present on the
expected paths or `PATH`.

## ECR Publication Smoke Test

The local candidate images were pushed once to private ECR to verify
authentication, repository immutability, upload behavior, and digest recording.
This is not an experiment-ready release tag.

| Image family | ECR tag | ECR digest | Pushed |
| --- | --- | --- | --- |
| PyTorch | `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch:publish-test-20260713-36621dd` | `sha256:d7ae8d63b704df53479ff1de539af134dc08e580f421ca5e9a9cd6bc4092a1b0` | `2026-07-14T00:06:46Z` |
| NeMo/Megatron | `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-nemo:publish-test-20260713-36621dd` | `sha256:f20d987b4b8e6ad413cea4abef7d6018672e8dd85ab2d31e065fe3e5747963a0` | `2026-07-14T00:14:00Z` |

Both ECR image statuses were `ACTIVE` after push. ECR scan-on-push was enabled.
The PyTorch scan completed and reported 60 critical, 178 high, 236 medium, 14
low, and 4 undefined findings. The NeMo scan completed and reported 67
critical, 286 high, 465 medium, 27 low, and 133 undefined findings. These
findings have not yet been adjudicated. The tags are immutable and should
remain publication test references only.

Before accepting either image for recorded experiments:

1. Preserve NVIDIA-pinned framework packages unless a tested override is
   required.
2. Run provider-side GPU qualification for driver, CUDA, NCCL, Nsight, DCGM,
   topology, and image execution.
3. Record final project image digests from ECR and GHCR after the same build is
   pushed to both registries.
