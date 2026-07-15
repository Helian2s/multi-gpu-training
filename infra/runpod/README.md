# Runpod adapter plan

Runpod supplies the A100 SXM NVLink environment that the AWS G7e profiles do not
provide. It hosts the Runpod communication baseline and the remaining
NeMo/Megatron experiments in the current catalog. It is not a silent general
fallback for AWS; broader Runpod use requires updating the planned compute
mapping.

The active Runpod execution queues are:

| Queue | Resource profile | Purpose |
| --- | --- | --- |
| `RUNPOD-A2-Megatron` | `RUNPOD-A100-SXM2` with one or two visible GPUs | Provider readiness, GHCR/storage validation, two-GPU qualification, EXP-10 NVLink/NCCL baseline, and EXP-11/EXP-13 one-/two-visible-GPU NeMo/Megatron phases |
| `RUNPOD-A4-Megatron` | `RUNPOD-A100-SXM4` | EXP-14 four-GPU TP=2 x DP=2 hybrid |

The `RUNPOD-A*` labels are execution queue labels. The suffix names the image
and framework family, while the resource profile records the exact rented Pod.
Every run still records the exact Runpod resource profile, GPU type, billed GPU
count, visible GPU count, datacenter, Pod ID, topology, visible mask, image
family, image digest, and billed resource.

## Accepted profiles

`RUNPOD-A100-SXM2` means:

- Runpod Secure Cloud Pod, not an Instant Cluster or multi-node deployment.
- One physical host with two rented GPUs.
- Exact Runpod GPU type ID `NVIDIA A100-SXM4-80GB`.
- One or two visible GPUs selected by a container visibility mask.
- Qualification evidence showing NVLink between every GPU pair used by a
  measured process group. NVSwitch is recorded only when the observed topology
  proves it.

It is used by `RUNPOD-A2-Megatron` for qualification, EXP-10, EXP-11, and
EXP-13. A one-visible-GPU phase still pays for both GPUs when it runs on this
two-GPU Pod, so the full resource cost is recorded. The current EXP-10 path
uses CUDA/NCCL tools inside the NeMo/Megatron image so the two-GPU Pod does not
need nested Docker or an image switch.

`RUNPOD-A100-SXM4` has the same requirements with four rented and visible GPUs.
It is used only by `RUNPOD-A4-Megatron` / EXP-14 because TP=2 x DP=2 requires
four ranks.

## Immediate preparation gates

Before a paid Runpod Pod is launched:

1. Rotate the previously exposed Runpod API key outside chat and Git.
2. Install and configure `runpodctl` locally, then run `runpodctl doctor`.
3. Publish the NeMo/Megatron image to GHCR and record the immutable digest.
4. Configure GHCR pull-only access for Runpod; do not store AWS credentials in
   Runpod.
5. Use Pod volume disk for working artifacts and copy the completed artifacts
   back to `artifacts/runs/runpod-volume-mirror/` before deleting the Pod.
6. Define launch guards for maximum lifetime/cost, durable stage-out, and
   stop/delete behavior on success and failure.

## Planned implementation

- **Compute:** one Runpod Pod on one physical host, provisioned and inspected
  through `runpodctl` or the Runpod API.
- **Run storage:** Pod volume disk mounted at `/runpod-volume`; it is not
  durable after Pod deletion, so copy artifacts to the ignored local mirror
  before deleting the Pod.
- **Staging:** container storage for performance-sensitive temporary files.
- **Images:** the immutable GHCR mirror of the NeMo/Megatron image content; use
  read-only GHCR registry credentials.
- **Visibility:** use the exact two- or four-GPU profile. Only the two-GPU
  profile masks one device for controlled one-GPU TP/PP/CP baselines. Record
  both physical and visible device sets for every run.
- **Cost safety:** maximum Pod lifetime, verified artifact collection, and stop
  or delete handling in success and failure paths.

Network-volume placement can constrain Pod selection to one datacenter. The
current Runpod queues intentionally avoid network volumes so the Pod can launch
wherever the required A100 SXM topology is available. If network volumes are
reintroduced, the adapter must report that placement constraint rather than
silently changing GPU type or topology. See the [Runpod network-volume documentation](https://docs.runpod.io/storage/network-volumes).

Runpod has no EC2-style instance-type identifier. Preserve the cloud class,
datacenter, Pod ID, exact GPU type ID, GPU count, host/topology output, CPU/RAM,
storage, and price as the resource identity. The official GPU table names the
required type `NVIDIA A100-SXM4-80GB`; Runpod's A100 SXM product page documents
80 GB memory and NVLink capability.

## Official sources

- [Runpod GPU types](https://docs.runpod.io/references/gpu-types)
- [Runpod A100 SXM](https://www.runpod.io/gpu-models/a100-sxm)
