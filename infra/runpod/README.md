# Runpod adapter plan

Runpod supplies the A100 SXM NVLink environment that the AWS G7e profiles do not
provide and hosts every NeMo/Megatron experiment in the current catalog. It is
not a silent general fallback for AWS; broader Runpod use requires updating the
planned compute mapping.

## Accepted profiles

`RUNPOD-A100-SXM2` means:

- Runpod Secure Cloud Pod, not an Instant Cluster or multi-node deployment.
- One physical host with two rented GPUs.
- Exact Runpod GPU type ID `NVIDIA A100-SXM4-80GB`.
- One or two visible GPUs selected by a container visibility mask.
- Qualification evidence showing NVLink between every GPU pair used by a
  measured process group. NVSwitch is recorded only when the observed topology
  proves it.

It is used by EXP-10 and EXP-11 through EXP-13. A one-visible-GPU baseline
still pays for both GPUs, so the full resource cost is recorded. Once the shared
workload and NeMo image are accepted, stage the pinned image, converted model,
and dataset once and reuse them across the grouped R2 session.

`RUNPOD-A100-SXM4` has the same requirements with four rented and visible GPUs.
It is used only by EXP-14 because TP=2 x DP=2 requires four ranks.

## Planned implementation

- **Compute:** one Runpod Pod on one physical host, provisioned and inspected
  through `runpodctl` or the Runpod API.
- **Durable storage:** a network volume for prepared model/data snapshots and
  complete run artifacts when its datacenter placement is compatible with GPU
  availability.
- **Staging:** Pod volume/container storage for performance-sensitive working
  files.
- **Images:** the immutable GHCR mirror of the image content published to AWS
  ECR; use read-only GHCR registry credentials.
- **Visibility:** use the exact two- or four-GPU profile. Only the two-GPU
  profile masks one device for controlled one-GPU TP/PP/CP baselines. Record
  both physical and visible device sets for every run.
- **Cost safety:** maximum Pod lifetime, verified artifact collection, and stop
  or delete handling in success and failure paths.

Network-volume placement can constrain Pod selection to one datacenter, so the
adapter must report that constraint rather than silently changing GPU type or
topology. See the [Runpod network-volume documentation](https://docs.runpod.io/storage/network-volumes).

Runpod has no EC2-style instance-type identifier. Preserve the cloud class,
datacenter, Pod ID, exact GPU type ID, GPU count, host/topology output, CPU/RAM,
storage, and price as the resource identity. The official GPU table names the
required type `NVIDIA A100-SXM4-80GB`; Runpod's A100 SXM product page documents
80 GB memory and NVLink capability.

## Official sources

- [Runpod GPU types](https://docs.runpod.io/references/gpu-types)
- [Runpod A100 SXM](https://www.runpod.io/gpu-models/a100-sxm)
