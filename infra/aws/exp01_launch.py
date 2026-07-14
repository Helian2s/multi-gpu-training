#!/usr/bin/env python3
"""Guarded AWS launcher for EXP-01.

The default mode is an EC2 dry run. It validates the RunInstances request shape
and permissions but does not create an instance. A real launch requires both a
configuration flag and an exact confirmation phrase.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "infra" / "aws" / "exp01_qualification.yaml"


class LaunchError(RuntimeError):
    """Raised when the launch request is unsafe or invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Create an EC2 instance. Omit for the default AWS dry run.",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help="Exact confirmation phrase required with --launch.",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument(
        "--hold-open-on-exit",
        action="store_true",
        help="After stage-out, keep the host alive for the configured post-run inspection window, then stop it.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise LaunchError(f"configuration is not a mapping: {path}")
    return data


def utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise LaunchError(f"run ID contains unsupported characters: {run_id!r}")
    return run_id


def container_name(run_id: str) -> str:
    return f"exp01-{validate_run_id(run_id)}"


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def image_reference(config: dict[str, Any]) -> str:
    image = config["image"]
    digest = image.get("digest")
    if not digest:
        raise LaunchError("image.digest must be recorded before launch")
    return f"{image['registry']}/{image['repository']}@{digest}"


def cache_volume_config(config: dict[str, Any]) -> dict[str, Any] | None:
    cache = config.get("cache_volume") or {}
    if not cache.get("enabled"):
        return None
    return cache


def cache_volume_summary(config: dict[str, Any]) -> str:
    cache = cache_volume_config(config)
    if cache is None:
        return "disabled"
    volume_id = cache.get("volume_id") or "pending-create"
    return (
        f"{volume_id} {cache['size_gib']} GiB {cache['volume_type']} "
        f"{cache['availability_zone']} mounted at {cache['mount_point']}; "
        f"Docker data root {cache['docker_data_root']}"
    )


def cache_volume_id_for_az(config: dict[str, Any], availability_zone: str) -> str:
    cache = cache_volume_config(config)
    if cache is None:
        raise LaunchError("cache volume is disabled")
    volume_ids_by_az = cache.get("volume_ids_by_az") or {}
    volume_id = volume_ids_by_az.get(availability_zone)
    if volume_id:
        return volume_id
    if availability_zone == cache.get("availability_zone") and cache.get("volume_id"):
        return cache["volume_id"]
    raise LaunchError(f"no cache volume is configured for {availability_zone}")


def auto_select_subnet(config: dict[str, Any]) -> bool:
    return bool(config["network"].get("auto_select_subnet"))


def shutdown_behavior(config: dict[str, Any], hold_open_on_exit: bool = False) -> str:
    safety = config["safety"]
    if hold_open_on_exit:
        return safety.get("manual_instance_initiated_shutdown_behavior", "stop")
    return safety["instance_initiated_shutdown_behavior"]


def post_run_inspection_seconds(config: dict[str, Any], hold_open_on_exit: bool = False) -> int:
    if not hold_open_on_exit:
        return 0
    return int(config["safety"].get("post_run_inspection_minutes", 15)) * 60


def confirmation_phrase(config: dict[str, Any], hold_open_on_exit: bool = False) -> str:
    lifetime = config["safety"]["maximum_lifetime_minutes"]
    action = shutdown_behavior(config, hold_open_on_exit)
    return f"launch {config['experiment_id']} {config['compute']['profile']} {action}-after-{lifetime}m"


def required_tags(config: dict[str, Any], run_id: str) -> list[dict[str, str]]:
    tags = {
        **config["safety"]["required_tags"],
        "RunId": run_id,
        "ComputeProfile": config["compute"]["profile"],
        "InstanceType": config["compute"]["instance_type"],
        "ImageDigest": config["image"]["digest"],
        "GitCommit": git_commit(),
        "MaxLifetimeMinutes": str(config["safety"]["maximum_lifetime_minutes"]),
    }
    return [{"Key": key, "Value": str(value)} for key, value in sorted(tags.items())]


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise LaunchError(f"invalid S3 artifact URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/")


def cache_volume_user_data(config: dict[str, Any]) -> str:
    cache = cache_volume_config(config)
    if cache is None:
        return 'echo "Persistent AWS cache volume disabled"'

    init_value = "1" if cache.get("initialize_if_empty") else "0"
    volume_id = cache.get("volume_id") or ""
    volume_cases = []
    for az, az_volume_id in sorted((cache.get("volume_ids_by_az") or {}).items()):
        volume_cases.append(f'    {az}) CACHE_VOLUME_ID="{az_volume_id}" ;;')
    volume_case_block = "\n".join(volume_cases)
    return textwrap.dedent(
        f"""\
        CACHE_VOLUME_ID="{volume_id}"
        CACHE_DEVICE_NAME="{cache['device_name']}"
        CACHE_MOUNT_POINT="{cache['mount_point']}"
        CACHE_FILESYSTEM="{cache['filesystem']}"
        CACHE_DOCKER_DATA_ROOT="{cache['docker_data_root']}"
        CACHE_INIT_IF_EMPTY="{init_value}"
        export CACHE_DOCKER_DATA_ROOT

        configure_cache_volume_id() {{
          local token instance_az
          token="$(curl -fsS -X PUT "http://169.254.169.254/latest/api/token" \\
            -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" || true)"
          if [[ -n "${{token}}" ]]; then
            instance_az="$(curl -fsS -H "X-aws-ec2-metadata-token: ${{token}}" \\
              "http://169.254.169.254/latest/meta-data/placement/availability-zone" || true)"
          else
            instance_az=""
          fi
          case "${{instance_az}}" in
        {volume_case_block}
          esac
          echo "INSTANCE_AZ=${{instance_az}}"
          echo "CACHE_VOLUME_ID=${{CACHE_VOLUME_ID}}"
        }}

        find_cache_device() {{
          local normalized_volume_id="${{CACHE_VOLUME_ID//-/}}"
          local by_id_plain="/dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_${{normalized_volume_id}}"
          local by_id_hyphen="/dev/disk/by-id/nvme-Amazon_Elastic_Block_Store_${{CACHE_VOLUME_ID}}"
          if [[ -e "${{by_id_plain}}" ]]; then
            readlink -f "${{by_id_plain}}"
            return 0
          fi
          if [[ -e "${{by_id_hyphen}}" ]]; then
            readlink -f "${{by_id_hyphen}}"
            return 0
          fi
          local dev serial
          for dev in /dev/nvme*n1; do
            [[ -e "${{dev}}" ]] || continue
            serial="$(cat "/sys/block/${{dev##*/}}/serial" 2>/dev/null || true)"
            if [[ "${{serial}}" == "${{normalized_volume_id}}" || "${{serial}}" == "${{CACHE_VOLUME_ID}}" ]]; then
              echo "${{dev}}"
              return 0
            fi
          done
          return 1
        }}

        wait_for_cache_device() {{
          local device
          for _ in $(seq 1 180); do
            if device="$(find_cache_device)"; then
              echo "${{device}}"
              return 0
            fi
            sleep 1
          done
          echo "Timed out waiting for EBS cache volume ${{CACHE_VOLUME_ID}}" >&2
          ls -l /dev/disk/by-id/ >&2 || true
          lsblk >&2 || true
          return 1
        }}

        configure_cache_volume() {{
          if [[ -z "${{CACHE_VOLUME_ID}}" ]]; then
            echo "cache_volume.enabled=true but cache_volume.volume_id is empty" >&2
            exit 20
          fi
          systemctl stop docker.socket docker >/dev/null 2>&1 || true
          local device
          device="$(wait_for_cache_device)"
          mkdir -p "${{CACHE_MOUNT_POINT}}"
          if ! blkid "${{device}}" >/dev/null 2>&1; then
            if [[ "${{CACHE_INIT_IF_EMPTY}}" == "1" ]]; then
              mkfs -t "${{CACHE_FILESYSTEM}}" -F "${{device}}"
            else
              echo "Cache volume ${{CACHE_VOLUME_ID}} has no filesystem and initialization is disabled" >&2
              exit 21
            fi
          fi
          if ! mountpoint -q "${{CACHE_MOUNT_POINT}}"; then
            mount "${{device}}" "${{CACHE_MOUNT_POINT}}"
          fi
          mkdir -p "${{CACHE_DOCKER_DATA_ROOT}}"
          python3 - <<'PY'
        import json
        import os
        from pathlib import Path

        path = Path("/etc/docker/daemon.json")
        data = {{}}
        if path.exists() and path.read_text(encoding="utf-8").strip():
            data = json.loads(path.read_text(encoding="utf-8"))
        data["data-root"] = os.environ["CACHE_DOCKER_DATA_ROOT"]
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
        PY
          echo "Docker data root configured at ${{CACHE_DOCKER_DATA_ROOT}}"
          df -h "${{CACHE_MOUNT_POINT}}"
        }}

        configure_cache_volume_id
        configure_cache_volume
        """
    ).rstrip()


def user_data_script(config: dict[str, Any], run_id: str, hold_open_on_exit: bool = False) -> str:
    bucket, prefix = parse_s3_uri(config["artifacts"]["durable_uri"])
    run_id = validate_run_id(run_id)
    image_ref = image_reference(config)
    region = config["aws"]["region"]
    registry = config["image"]["registry"]
    lifetime_seconds = int(config["safety"]["maximum_lifetime_minutes"]) * 60
    host_run_dir = f"/opt/multi-gpu-training/artifacts/runs/{config['experiment_id']}/{run_id}"
    container_run_dir = f"/workspace/artifacts/runs/{config['experiment_id']}/{run_id}"
    artifact_target = f"s3://{bucket}/{prefix}/runs/{run_id}/"
    hold_value = "1" if hold_open_on_exit else "0"
    inspection_seconds = post_run_inspection_seconds(config, hold_open_on_exit)
    inspection_minutes = inspection_seconds // 60

    return textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        exec > >(tee -a /var/log/exp01-user-data.log) 2>&1

        EXPERIMENT_ID="{config['experiment_id']}"
        RUN_ID="{run_id}"
        AWS_REGION="{region}"
        ECR_REGISTRY="{registry}"
        IMAGE_REF="{image_ref}"
        HOST_RUN_DIR="{host_run_dir}"
        CONTAINER_RUN_DIR="{container_run_dir}"
        CONTAINER_NAME="{container_name(run_id)}"
        ARTIFACT_TARGET="{artifact_target}"
        MAX_LIFETIME_SECONDS="{lifetime_seconds}"
        POST_RUN_INSPECTION_SECONDS="{inspection_seconds}"
        POST_RUN_INSPECTION_MINUTES="{inspection_minutes}"
        HOLD_OPEN_ON_EXIT="{hold_value}"

        echo "EXP-01 host bootstrap started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "RUN_ID=${{RUN_ID}}"
        echo "IMAGE_REF=${{IMAGE_REF}}"
        echo "ARTIFACT_TARGET=${{ARTIFACT_TARGET}}"

        mkdir -p "${{HOST_RUN_DIR}}/raw" "${{HOST_RUN_DIR}}/metrics"

        install_safety_timer() {{
          cat >/usr/local/sbin/exp01-safety-shutdown.sh <<'SCRIPT'
        #!/usr/bin/env bash
        set -euo pipefail
        echo "EXP-01 maximum lifetime reached at $(date -u +%Y-%m-%dT%H:%M:%SZ); shutting down" | tee -a /var/log/exp01-user-data.log
        shutdown -h now
        SCRIPT
          chmod 0755 /usr/local/sbin/exp01-safety-shutdown.sh

          cat >/etc/systemd/system/exp01-safety-shutdown.service <<'UNIT'
        [Unit]
        Description=EXP-01 maximum lifetime shutdown guard

        [Service]
        Type=oneshot
        ExecStart=/usr/local/sbin/exp01-safety-shutdown.sh
        UNIT

          cat >/etc/systemd/system/exp01-safety-shutdown.timer <<UNIT
        [Unit]
        Description=EXP-01 maximum lifetime timer

        [Timer]
        OnBootSec=${{MAX_LIFETIME_SECONDS}}s
        Unit=exp01-safety-shutdown.service
        AccuracySec=30s
        Persistent=false

        [Install]
        WantedBy=timers.target
        UNIT

          systemctl daemon-reload
          systemctl enable --now exp01-safety-shutdown.timer
          systemctl list-timers exp01-safety-shutdown.timer --no-pager || true
        }}

        install_safety_timer

        finish() {{
          status="$?"
          set +e
          cp /var/log/exp01-user-data.log "${{HOST_RUN_DIR}}/raw/" 2>/dev/null || true
          cp /var/log/cloud-init-output.log "${{HOST_RUN_DIR}}/raw/" 2>/dev/null || true
          echo "${{status}}" > "${{HOST_RUN_DIR}}/exit_status.txt"
          date -u +%Y-%m-%dT%H:%M:%SZ > "${{HOST_RUN_DIR}}/finished_utc.txt"
          aws s3 sync "${{HOST_RUN_DIR}}/" "${{ARTIFACT_TARGET}}" --region "${{AWS_REGION}}" --only-show-errors
          sync
          if [[ "${{HOLD_OPEN_ON_EXIT}}" == "1" ]]; then
            echo "Holding host open for ${{POST_RUN_INSPECTION_MINUTES}} minutes after container exit"
            inspection_deadline="$((SECONDS + POST_RUN_INSPECTION_SECONDS))"
            while (( SECONDS < inspection_deadline )); do
              remaining="$((inspection_deadline - SECONDS))"
              if (( remaining > 60 )); then
                sleep 60
              else
                sleep "${{remaining}}"
              fi
              aws s3 sync "${{HOST_RUN_DIR}}/" "${{ARTIFACT_TARGET}}" --region "${{AWS_REGION}}" --only-show-errors
            done
            echo "Post-run inspection window finished; stopping instance"
            shutdown -h now
            exit "${{status}}"
          fi
          echo "Shutting down; InstanceInitiatedShutdownBehavior controls stop or terminate"
          shutdown -h now
        }}
        trap finish EXIT

        __CACHE_VOLUME_SETUP__

        systemctl start docker
        aws ecr get-login-password --region "${{AWS_REGION}}" \\
          | docker login --username AWS --password-stdin "${{ECR_REGISTRY}}"
        docker pull "${{IMAGE_REF}}"

        set +e
        docker rm -f "${{CONTAINER_NAME}}" >/dev/null 2>&1 || true
        docker run --rm --gpus all --ipc=host \\
          --name "${{CONTAINER_NAME}}" \\
          --ulimit memlock=-1 --ulimit stack=67108864 \\
          -e RUN_ID="${{RUN_ID}}" \\
          -e RUN_DIR="${{CONTAINER_RUN_DIR}}" \\
          -e CUDA_VISIBLE_DEVICES="0,1" \\
          -e NCCL_DEBUG="INFO" \\
          -e NCCL_DEBUG_SUBSYS="INIT,COLL,GRAPH" \\
          -v "${{HOST_RUN_DIR}}:${{CONTAINER_RUN_DIR}}" \\
          "${{IMAGE_REF}}" \\
          bash experiments/exp_01_aws_pcie_p2p_nccl_communication/collect_exp01.sh \\
          2>&1 | tee "${{HOST_RUN_DIR}}/raw/docker-run.log"
        docker_status="${{PIPESTATUS[0]}}"
        set -e
        exit "${{docker_status}}"
        """
    ).replace("__CACHE_VOLUME_SETUP__", cache_volume_user_data(config))


def selected_subnet(config: dict[str, Any]) -> str:
    subnet_ids = config["network"].get("subnet_ids") or []
    if not subnet_ids:
        raise LaunchError("network.subnet_ids must contain at least one subnet")
    return subnet_ids[0]


def build_run_instances_request(
    config: dict[str, Any],
    run_id: str,
    hold_open_on_exit: bool = False,
) -> dict[str, Any]:
    run_id = validate_run_id(run_id)
    host = config["host"]
    network = config["network"]
    user_data = user_data_script(config, run_id, hold_open_on_exit=hold_open_on_exit)
    request: dict[str, Any] = {
        "ImageId": host["ami_id"],
        "InstanceType": config["compute"]["instance_type"],
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": config["identity"]["instance_profile"]},
        "InstanceInitiatedShutdownBehavior": shutdown_behavior(config, hold_open_on_exit),
        "MetadataOptions": {
            "HttpEndpoint": config["safety"]["metadata_options"]["http_endpoint"],
            "HttpTokens": config["safety"]["metadata_options"]["http_tokens"],
            "InstanceMetadataTags": config["safety"]["metadata_options"]["instance_metadata_tags"],
        },
        "BlockDeviceMappings": [
            {
                "DeviceName": host["root_device_name"],
                "Ebs": {
                    "VolumeSize": int(host["root_volume_gib"]),
                    "VolumeType": host["root_volume_type"],
                    "Encrypted": bool(host["root_volume_encrypted"]),
                    "DeleteOnTermination": True,
                },
            }
        ],
        "TagSpecifications": [
            {"ResourceType": resource, "Tags": required_tags(config, run_id)}
            for resource in ("instance", "volume", "network-interface")
        ],
        "UserData": base64.b64encode(user_data.encode("utf-8")).decode("ascii"),
    }
    if auto_select_subnet(config):
        request["SecurityGroupIds"] = network["security_group_ids"]
    else:
        request["NetworkInterfaces"] = [
            {
                "DeviceIndex": 0,
                "SubnetId": selected_subnet(config),
                "Groups": network["security_group_ids"],
                "AssociatePublicIpAddress": bool(network["associate_public_ip_address"]),
            }
        ]
    key_name = network.get("ssh_key_name")
    if key_name:
        request["KeyName"] = key_name
    return request


def request_summary(
    config: dict[str, Any],
    request: dict[str, Any],
    run_id: str,
    hold_open_on_exit: bool = False,
) -> dict[str, Any]:
    user_data = base64.b64decode(request["UserData"]).decode("utf-8")
    return {
        "experiment_id": config["experiment_id"],
        "run_id": run_id,
        "aws_profile": config["aws"]["profile"],
        "region": config["aws"]["region"],
        "ami_id": request["ImageId"],
        "ami_name": config["host"]["ami_name"],
        "instance_type": request["InstanceType"],
        "compute_profile": config["compute"]["profile"],
        "subnet_id": request.get("NetworkInterfaces", [{"SubnetId": "auto-default-vpc"}])[0]["SubnetId"],
        "security_group_ids": request.get("SecurityGroupIds")
        or request["NetworkInterfaces"][0]["Groups"],
        "instance_profile": request["IamInstanceProfile"]["Name"],
        "image": image_reference(config),
        "container_name": container_name(run_id),
        "artifact_uri": config["artifacts"]["durable_uri"],
        "cache_volume": cache_volume_summary(config),
        "maximum_lifetime_minutes": config["safety"]["maximum_lifetime_minutes"],
        "post_run_inspection_minutes": int(config["safety"].get("post_run_inspection_minutes", 15))
        if hold_open_on_exit
        else 0,
        "shutdown_behavior": request["InstanceInitiatedShutdownBehavior"],
        "hold_open_on_exit": hold_open_on_exit,
        "launch_enabled": bool(config["safety"]["launch_enabled"]),
        "confirmation_phrase": confirmation_phrase(config, hold_open_on_exit),
        "user_data_sha256": hashlib.sha256(user_data.encode("utf-8")).hexdigest(),
    }


def run_aws(
    config: dict[str, Any],
    request: dict[str, Any],
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    command = [
        "aws",
        "ec2",
        "run-instances",
        "--profile",
        config["aws"]["profile"],
        "--region",
        config["aws"]["region"],
        "--output",
        "json",
        "--cli-input-json",
    ]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as handle:
        json.dump(request, handle)
        handle.flush()
        command.append(f"file://{handle.name}")
        if dry_run:
            command.append("--dry-run")
        return subprocess.run(command, check=False, capture_output=True, text=True)


def aws_command(config: dict[str, Any], *args: str) -> list[str]:
    return [
        "aws",
        *args,
        "--profile",
        config["aws"]["profile"],
        "--region",
        config["aws"]["region"],
        "--output",
        "json",
    ]


def run_aws_command(config: dict[str, Any], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(aws_command(config, *args), check=False, capture_output=True, text=True)


def aws_json(config: dict[str, Any], *args: str) -> Any:
    result = run_aws_command(config, *args)
    if result.returncode != 0:
        raise LaunchError(
            f"AWS command failed: {' '.join(aws_command(config, *args))}\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return json.loads(result.stdout or "{}")


def selected_subnet_az(config: dict[str, Any]) -> str:
    data = aws_json(
        config,
        "ec2",
        "describe-subnets",
        "--subnet-ids",
        selected_subnet(config),
    )
    subnets = data.get("Subnets", [])
    if len(subnets) != 1:
        raise LaunchError(f"expected one selected subnet, got {len(subnets)}")
    return subnets[0]["AvailabilityZone"]


def validate_cache_volume_for_launch(config: dict[str, Any]) -> dict[str, Any] | None:
    cache = cache_volume_config(config)
    if cache is None:
        return None
    volume_id = cache.get("volume_id")
    if not volume_id:
        raise LaunchError("cache_volume.volume_id must be recorded before launch")
    volume_ids = sorted(set((cache.get("volume_ids_by_az") or {}).values()) or {volume_id})
    if not auto_select_subnet(config):
        subnet_az = selected_subnet_az(config)
        volume_id = cache_volume_id_for_az(config, subnet_az)
        if cache["availability_zone"] != subnet_az and not (cache.get("volume_ids_by_az") or {}).get(subnet_az):
            raise LaunchError(
                "selected subnet and cache volume availability zones differ: "
                f"{selected_subnet(config)} is {subnet_az}, cache config is {cache['availability_zone']}"
            )
        volume_ids = [volume_id]
    data = aws_json(config, "ec2", "describe-volumes", "--volume-ids", *volume_ids)
    volumes = data.get("Volumes", [])
    if len(volumes) != len(volume_ids):
        raise LaunchError(f"expected cache volumes {volume_ids}, got {len(volumes)}")
    problems: list[str] = []
    expected_by_volume = {
        value: key for key, value in (cache.get("volume_ids_by_az") or {}).items()
    }
    if not expected_by_volume:
        expected_by_volume = {volume_id: cache["availability_zone"]}
    for volume in volumes:
        current_volume_id = volume["VolumeId"]
        expected_az = expected_by_volume.get(current_volume_id, cache["availability_zone"])
        if volume.get("State") != "available":
            problems.append(f"{current_volume_id} state={volume.get('State')}, expected available")
        if volume.get("AvailabilityZone") != expected_az:
            problems.append(f"{current_volume_id} az={volume.get('AvailabilityZone')}, expected {expected_az}")
        if int(volume.get("Size", 0)) != int(cache["size_gib"]):
            problems.append(f"{current_volume_id} size={volume.get('Size')} GiB, expected {cache['size_gib']} GiB")
        if volume.get("VolumeType") != cache["volume_type"]:
            problems.append(f"{current_volume_id} type={volume.get('VolumeType')}, expected {cache['volume_type']}")
        if bool(volume.get("Encrypted")) != bool(cache["encrypted"]):
            problems.append(f"{current_volume_id} encrypted={volume.get('Encrypted')}, expected {cache['encrypted']}")
    if problems:
        raise LaunchError(f"cache volume is not launch-ready: {'; '.join(problems)}")
    return volumes[0]


def cleanup_after_launch_failure(
    config: dict[str, Any],
    instance_id: str,
    hold_open_on_exit: bool = False,
) -> None:
    if hold_open_on_exit:
        run_aws_command(config, "ec2", "stop-instances", "--instance-ids", instance_id)
    else:
        run_aws_command(config, "ec2", "terminate-instances", "--instance-ids", instance_id)


def wait_instance_running(config: dict[str, Any], instance_id: str) -> None:
    result = run_aws_command(
        config,
        "ec2",
        "wait",
        "instance-running",
        "--instance-ids",
        instance_id,
    )
    if result.returncode != 0:
        raise LaunchError(
            "instance did not reach running state before cache attach\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )


def instance_availability_zone(config: dict[str, Any], instance_id: str) -> str:
    data = aws_json(config, "ec2", "describe-instances", "--instance-ids", instance_id)
    instances = [
        instance
        for reservation in data.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if len(instances) != 1:
        raise LaunchError(f"expected one instance for {instance_id}, got {len(instances)}")
    return instances[0]["Placement"]["AvailabilityZone"]


def attach_cache_volume(config: dict[str, Any], instance_id: str) -> dict[str, Any] | None:
    cache = cache_volume_config(config)
    if cache is None:
        return None
    availability_zone = instance_availability_zone(config, instance_id)
    volume_id = cache_volume_id_for_az(config, availability_zone)
    result = run_aws_command(
        config,
        "ec2",
        "attach-volume",
        "--volume-id",
        volume_id,
        "--instance-id",
        instance_id,
        "--device",
        cache["device_name"],
    )
    if result.returncode != 0:
        raise LaunchError(
            "cache volume attach failed\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    wait_result = run_aws_command(
        config,
        "ec2",
        "wait",
        "volume-in-use",
        "--volume-ids",
        volume_id,
    )
    if wait_result.returncode != 0:
        raise LaunchError(
            "cache volume did not reach in-use state\n"
            f"stdout: {wait_result.stdout.strip()}\n"
            f"stderr: {wait_result.stderr.strip()}"
        )
    return json.loads(result.stdout or "{}")


def dry_run(config: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    result = run_aws(config, request, dry_run=True)
    combined = f"{result.stdout}\n{result.stderr}"
    if "DryRunOperation" in combined:
        return {"status": "ok", "detail": "EC2 RunInstances dry run authorized"}
    raise LaunchError(
        "EC2 RunInstances dry run failed\n"
        f"stdout: {result.stdout.strip()}\n"
        f"stderr: {result.stderr.strip()}"
    )


def launch(
    config: dict[str, Any],
    request: dict[str, Any],
    confirm: str,
    hold_open_on_exit: bool = False,
) -> dict[str, Any]:
    if not config["safety"].get("launch_enabled"):
        raise LaunchError("safety.launch_enabled is false; refusing to create EC2 resources")
    expected = confirmation_phrase(config, hold_open_on_exit)
    if confirm != expected:
        raise LaunchError(f"confirmation phrase mismatch; expected: {expected!r}")
    validate_cache_volume_for_launch(config)
    result = run_aws(config, request, dry_run=False)
    if result.returncode != 0:
        raise LaunchError(
            "EC2 RunInstances failed\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    launch_response = json.loads(result.stdout)
    instances = launch_response.get("Instances", [])
    if len(instances) != 1:
        raise LaunchError(f"expected one launched instance, got {len(instances)}")
    instance_id = instances[0]["InstanceId"]
    try:
        wait_instance_running(config, instance_id)
        attach_response = attach_cache_volume(config, instance_id)
    except LaunchError:
        cleanup_after_launch_failure(config, instance_id, hold_open_on_exit=hold_open_on_exit)
        raise
    return {"run_instances": launch_response, "cache_volume_attachment": attach_response}


def render(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"Mode: {payload['mode']}")
    for key, value in payload["summary"].items():
        print(f"{key}: {value}")
    print(f"result: {payload['result']['status']} - {payload['result']['detail']}")


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        run_id = validate_run_id(args.run_id or utc_run_id())
        request = build_run_instances_request(
            config,
            run_id,
            hold_open_on_exit=args.hold_open_on_exit,
        )
        summary = request_summary(
            config,
            request,
            run_id,
            hold_open_on_exit=args.hold_open_on_exit,
        )
        if args.launch:
            response = launch(
                config,
                request,
                args.confirm,
                hold_open_on_exit=args.hold_open_on_exit,
            )
            result = {"status": "launched", "detail": response}
            mode = "launch"
        else:
            result = dry_run(config, request)
            mode = "dry-run"
        render({"mode": mode, "summary": summary, "result": result}, args.json)
    except LaunchError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
