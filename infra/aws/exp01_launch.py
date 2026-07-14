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
        help="After stage-out, keep the host alive until the lifetime watchdog or manual shutdown.",
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


def confirmation_phrase(config: dict[str, Any]) -> str:
    lifetime = config["safety"]["maximum_lifetime_minutes"]
    return f"launch {config['experiment_id']} {config['compute']['profile']} terminate-after-{lifetime}m"


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
        HOLD_OPEN_ON_EXIT="{hold_value}"

        echo "EXP-01 host bootstrap started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "RUN_ID=${{RUN_ID}}"
        echo "IMAGE_REF=${{IMAGE_REF}}"
        echo "ARTIFACT_TARGET=${{ARTIFACT_TARGET}}"

        mkdir -p "${{HOST_RUN_DIR}}/raw" "${{HOST_RUN_DIR}}/metrics"

        (
          sleep "${{MAX_LIFETIME_SECONDS}}"
          echo "Maximum lifetime reached; shutting down for EC2 termination"
          shutdown -h now
        ) &
        WATCHDOG_PID="$!"

        finish() {{
          status="$?"
          set +e
          kill "${{WATCHDOG_PID}}" >/dev/null 2>&1 || true
          cp /var/log/exp01-user-data.log "${{HOST_RUN_DIR}}/raw/" 2>/dev/null || true
          cp /var/log/cloud-init-output.log "${{HOST_RUN_DIR}}/raw/" 2>/dev/null || true
          echo "${{status}}" > "${{HOST_RUN_DIR}}/exit_status.txt"
          date -u +%Y-%m-%dT%H:%M:%SZ > "${{HOST_RUN_DIR}}/finished_utc.txt"
          aws s3 sync "${{HOST_RUN_DIR}}/" "${{ARTIFACT_TARGET}}" --region "${{AWS_REGION}}" --only-show-errors
          sync
          if [[ "${{HOLD_OPEN_ON_EXIT}}" == "1" ]]; then
            echo "Holding host open for manual inspection until watchdog or manual shutdown"
            while true; do
              sleep 60
              aws s3 sync "${{HOST_RUN_DIR}}/" "${{ARTIFACT_TARGET}}" --region "${{AWS_REGION}}" --only-show-errors
            done
          fi
          echo "Shutting down; InstanceInitiatedShutdownBehavior=terminate"
          shutdown -h now
        }}
        trap finish EXIT

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
    )


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
    safety = config["safety"]
    user_data = user_data_script(config, run_id, hold_open_on_exit=hold_open_on_exit)
    request: dict[str, Any] = {
        "ImageId": host["ami_id"],
        "InstanceType": config["compute"]["instance_type"],
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": config["identity"]["instance_profile"]},
        "InstanceInitiatedShutdownBehavior": safety["instance_initiated_shutdown_behavior"],
        "MetadataOptions": {
            "HttpEndpoint": safety["metadata_options"]["http_endpoint"],
            "HttpTokens": safety["metadata_options"]["http_tokens"],
            "InstanceMetadataTags": safety["metadata_options"]["instance_metadata_tags"],
        },
        "NetworkInterfaces": [
            {
                "DeviceIndex": 0,
                "SubnetId": selected_subnet(config),
                "Groups": network["security_group_ids"],
                "AssociatePublicIpAddress": bool(network["associate_public_ip_address"]),
            }
        ],
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
        "subnet_id": request["NetworkInterfaces"][0]["SubnetId"],
        "security_group_ids": request["NetworkInterfaces"][0]["Groups"],
        "instance_profile": request["IamInstanceProfile"]["Name"],
        "image": image_reference(config),
        "container_name": container_name(run_id),
        "artifact_uri": config["artifacts"]["durable_uri"],
        "maximum_lifetime_minutes": config["safety"]["maximum_lifetime_minutes"],
        "shutdown_behavior": request["InstanceInitiatedShutdownBehavior"],
        "hold_open_on_exit": hold_open_on_exit,
        "launch_enabled": bool(config["safety"]["launch_enabled"]),
        "confirmation_phrase": confirmation_phrase(config),
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


def launch(config: dict[str, Any], request: dict[str, Any], confirm: str) -> dict[str, Any]:
    if not config["safety"].get("launch_enabled"):
        raise LaunchError("safety.launch_enabled is false; refusing to create EC2 resources")
    expected = confirmation_phrase(config)
    if confirm != expected:
        raise LaunchError(f"confirmation phrase mismatch; expected: {expected!r}")
    result = run_aws(config, request, dry_run=False)
    if result.returncode != 0:
        raise LaunchError(
            "EC2 RunInstances failed\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return json.loads(result.stdout)


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
            response = launch(config, request, args.confirm)
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
