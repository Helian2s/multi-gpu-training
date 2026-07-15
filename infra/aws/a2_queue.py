#!/usr/bin/env python3
"""Operator tooling for the AWS-A2 PyTorch experiment queue.

The queue can render its host script, send that script to an already-running
AWS-A2 host through SSM, or launch a new guarded AWS-A2 host whose user-data
runs the queue. A real launch requires an exact confirmation phrase.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from infra.aws.exp01_launch import (
    attach_cache_volume,
    auto_select_subnet,
    cache_volume_summary,
    cache_volume_user_data,
    git_commit,
    load_config as load_launch_config,
    run_aws,
    run_aws_command,
    selected_subnet,
    validate_cache_volume_for_launch,
    wait_instance_running,
)
from infra.aws.exp01_ops import select_running_instance, send_ssm_command


DEFAULT_QUEUE_CONFIG = REPOSITORY_ROOT / "infra" / "aws" / "a2_experiment_queue.yaml"


class QueueError(RuntimeError):
    """Raised when the AWS-A2 queue is unsafe or incomplete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-config", type=Path, default=DEFAULT_QUEUE_CONFIG)
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--run-id", default="")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Print the queue plan without contacting EC2.")
    subparsers.add_parser("host-script", help="Print the generated host script.")
    run = subparsers.add_parser("run", help="Run the queue on an already-running AWS-A2 host via SSM.")
    run.add_argument("--timeout-seconds", type=int, default=21600)
    subparsers.add_parser("launch-dry-run", help="Validate the EC2 launch request without creating resources.")
    launch = subparsers.add_parser("launch", help="Launch one AWS-A2 host and run the queue from user-data.")
    launch.add_argument("--confirm", default="", help="Exact confirmation phrase required for real launch.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise QueueError(f"configuration is not a mapping: {path}")
    return data


def utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise QueueError(f"run ID contains unsupported characters: {run_id!r}")
    return run_id


def image_reference(config: dict[str, Any]) -> str:
    image = config["image"]
    digest = image.get("digest")
    if not digest:
        raise QueueError("queue image.digest is missing; build and publish the AWS-A2 image first")
    return f"{image['registry']}/{image['repository']}@{digest}"


def validate_queue(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise QueueError("queue schema_version must be 1")
    if config["queue"]["compute_profile"] != "AWS-A2":
        raise QueueError("this runner only supports AWS-A2")
    expected = [
        "EXP-01-A2",
        "EXP-02-A2V1",
        "EXP-02-A2V2",
        "EXP-07-A2V1",
        "EXP-07-A2V2",
        "EXP-08-A2V2",
        "EXP-09-A2V1",
        "EXP-09-A2V2",
    ]
    seen: set[str] = set()
    for item in config.get("run_units", []):
        run_unit = item.get("run_unit")
        experiment_id = item.get("experiment_id")
        if not isinstance(run_unit, str) or not run_unit.startswith("EXP-"):
            raise QueueError(f"invalid AWS-A2 run_unit: {run_unit!r}")
        if not isinstance(experiment_id, str) or not experiment_id.startswith("EXP-"):
            raise QueueError(f"invalid experiment_id: {experiment_id!r}")
        if run_unit in seen:
            raise QueueError(f"duplicate run unit: {run_unit}")
        seen.add(run_unit)
        kind = item.get("kind")
        if kind == "runner":
            runner = REPOSITORY_ROOT / item["runner"]
            if not runner.is_file():
                raise QueueError(f"runner does not exist: {runner}")
        elif kind == "shell":
            command = item.get("command")
            if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
                raise QueueError(f"shell run unit {run_unit} must define a command list")
        else:
            raise QueueError(f"run unit {run_unit} has unsupported kind: {kind!r}")
    actual = [item["run_unit"] for item in config.get("run_units", [])]
    if actual != expected:
        raise QueueError(f"AWS-A2 queue order must be {expected}; got {actual}")


def render_plan(config: dict[str, Any]) -> dict[str, Any]:
    image_ready = bool(config["image"].get("digest"))
    return {
        "queue": config["queue"]["id"],
        "compute_profile": config["queue"]["compute_profile"],
        "image_ready": image_ready,
        "image": image_reference(config) if image_ready else None,
        "inputs": config["inputs"]["durable_uri"],
        "maximum_lifetime_minutes": config["queue"]["maximum_lifetime_minutes"],
        "post_queue_inspection_minutes": config["queue"]["post_queue_inspection_minutes"],
        "stop_on_success": bool(config["queue"].get("stop_on_success", True)),
        "stop_on_failure": bool(config["queue"].get("stop_on_failure", True)),
        "run_units": [
            {
                "run_unit": item["run_unit"],
                "experiment_id": item["experiment_id"],
                "kind": item["kind"],
                "runner": item.get("runner"),
                "command": item.get("command"),
                "visible_devices": item["visible_devices"],
                "durable_uri": item["durable_uri"],
                "timeout_seconds": item["timeout_seconds"],
            }
            for item in config["run_units"]
        ],
    }


def run_unit_invocations(config: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in config["run_units"]:
        command_json = json.dumps(item.get("command", []), separators=(",", ":"))
        args = [
            item["run_unit"],
            item["experiment_id"],
            item["kind"],
            item["visible_devices"],
            item.get("runner", ""),
            command_json,
            item["durable_uri"],
            str(item["timeout_seconds"]),
        ]
        invocation = "run_queue_unit " + " ".join(shlex.quote(arg) for arg in args)
        lines.extend(
            [
                'if [[ "${queue_status}" == "0" ]]; then',
                f'  {invocation} || queue_status="$?"',
                "fi",
            ]
        )
    return "\n".join(lines)


def env_exports(config: dict[str, Any]) -> str:
    return "\n".join(
        f"export {key}={shlex.quote(str(value))}"
        for key, value in sorted(config["container"].get("environment", {}).items())
    )


def host_script(config: dict[str, Any], launch_config: dict[str, Any], run_id: str) -> str:
    image = image_reference(config)
    queue = config["queue"]
    inputs = config["inputs"]
    artifacts = config["artifacts"]
    container = config["container"]
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        exec > >(tee -a /var/log/aws-a2-queue.log) 2>&1

        QUEUE_ID="{queue['id']}"
        RUN_ID="{validate_run_id(run_id)}"
        AWS_REGION="{config['aws']['region']}"
        ECR_REGISTRY="{config['image']['registry']}"
        IMAGE_REF="{image}"
        INPUTS_S3_URI="{inputs['durable_uri'].rstrip('/')}/"
        HOST_DATA_DIR="{inputs['host_data_dir']}"
        CONTAINER_DATA_DIR="{inputs['container_data_dir']}"
        REQUIRED_INPUT_MANIFEST="{inputs['required_manifest']}"
        HOST_ARTIFACT_ROOT="{artifacts['host_root']}"
        CONTAINER_ARTIFACT_ROOT="{artifacts['container_root']}"
        QUEUE_LOG_URI="{artifacts['queue_log_uri'].rstrip('/')}/"
        CONTAINER_NAME_PREFIX="{container['name_prefix']}"
        SHM_SIZE="{container['shm_size']}"
        MAX_LIFETIME_MINUTES="{queue['maximum_lifetime_minutes']}"
        POST_QUEUE_INSPECTION_MINUTES="{queue['post_queue_inspection_minutes']}"
        STOP_ON_SUCCESS="{str(bool(queue.get('stop_on_success', True))).lower()}"
        STOP_ON_FAILURE="{str(bool(queue.get('stop_on_failure', True))).lower()}"

        FINISH_STARTED=0

        finish() {{
          local status="$?"
          if [[ "${{FINISH_STARTED}}" == "1" ]]; then
            exit "${{status}}"
          fi
          FINISH_STARTED=1
          set +e
          echo "AWS-A2 queue finishing with status ${{status}} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
          mkdir -p "${{HOST_ARTIFACT_ROOT}}"
          cp /var/log/aws-a2-queue.log "${{HOST_ARTIFACT_ROOT}}/AWS-A2-PyTorch-${{RUN_ID}}.log" 2>/dev/null || true
          aws s3 cp "${{HOST_ARTIFACT_ROOT}}/AWS-A2-PyTorch-${{RUN_ID}}.log" \\
            "${{QUEUE_LOG_URI}}runs/${{RUN_ID}}/queue.log" \\
            --region "${{AWS_REGION}}" --only-show-errors || true
          if [[ "${{status}}" != "0" && "${{STOP_ON_FAILURE}}" != "true" ]]; then
            echo "AWS-A2 queue failed; leaving instance running for inspection. The hard safety shutdown remains scheduled."
            exit "${{status}}"
          fi
          if [[ "${{status}}" == "0" && "${{STOP_ON_SUCCESS}}" != "true" ]]; then
            echo "AWS-A2 queue succeeded; leaving instance running for inspection. The hard safety shutdown remains scheduled."
            exit "${{status}}"
          fi
          echo "AWS-A2 queue holding for ${{POST_QUEUE_INSPECTION_MINUTES}} minutes before shutdown"
          sleep "$((POST_QUEUE_INSPECTION_MINUTES * 60))"
          shutdown -h now
          exit "${{status}}"
        }}
        trap finish EXIT

        echo "AWS-A2 queue started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "QUEUE_ID=${{QUEUE_ID}}"
        echo "RUN_ID=${{RUN_ID}}"
        echo "IMAGE_REF=${{IMAGE_REF}}"
        echo "INPUTS_S3_URI=${{INPUTS_S3_URI}}"

        sudo shutdown -c || true
        sudo shutdown -h +${{MAX_LIFETIME_MINUTES}} "AWS-A2 queue safety stop"

        __CACHE_VOLUME_SETUP__

        systemctl start docker
        aws ecr get-login-password --region "${{AWS_REGION}}" \\
          | docker login --username AWS --password-stdin "${{ECR_REGISTRY}}"
        docker pull "${{IMAGE_REF}}"

        mkdir -p "${{HOST_DATA_DIR}}" "${{HOST_ARTIFACT_ROOT}}"
        if [[ ! -f "${{HOST_DATA_DIR}}/${{REQUIRED_INPUT_MANIFEST}}" ]]; then
          echo "Staging pinned inputs from ${{INPUTS_S3_URI}} to ${{HOST_DATA_DIR}}"
          aws s3 sync "${{INPUTS_S3_URI}}data/" "${{HOST_DATA_DIR}}/" \\
            --region "${{AWS_REGION}}" --only-show-errors
        fi
        test -f "${{HOST_DATA_DIR}}/${{REQUIRED_INPUT_MANIFEST}}"

        __ENV_EXPORTS__
        export MULTI_GPU_TRAINING_IMAGE_REF="${{IMAGE_REF}}"

        run_queue_unit() {{
          local run_unit="$1"
          local experiment_id="$2"
          local kind="$3"
          local visible_devices="$4"
          local runner="$5"
          local command_json="$6"
          local durable_uri="$7"
          local timeout_seconds="$8"
          local safe_experiment
          safe_experiment="$(echo "${{experiment_id}}" | tr '[:upper:]' '[:lower:]')"
          local container_name="${{CONTAINER_NAME_PREFIX}}-${{safe_experiment}}-${{run_unit,,}}-${{RUN_ID}}"
          local host_exp_dir="${{HOST_ARTIFACT_ROOT}}/${{experiment_id}}/${{RUN_ID}}"
          local container_run_dir="${{CONTAINER_ARTIFACT_ROOT}}/${{experiment_id}}/${{RUN_ID}}"
          mkdir -p "${{host_exp_dir}}/raw" "${{host_exp_dir}}/metrics"
          echo "Starting ${{run_unit}} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
          docker rm -f "${{container_name}}" >/dev/null 2>&1 || true
          set +e
          if [[ "${{kind}}" == "shell" ]]; then
            mapfile -t unit_command < <(COMMAND_JSON="${{command_json}}" python3 - <<'PY'
        import json
        import os
        for item in json.loads(os.environ["COMMAND_JSON"]):
            print(item)
        PY
        )
            timeout "${{timeout_seconds}}" docker run --rm --gpus all --ipc=host \\
              --name "${{container_name}}" \\
              --shm-size "${{SHM_SIZE}}" \\
              --ulimit memlock=-1 --ulimit stack=67108864 \\
              -e CUDA_VISIBLE_DEVICES="${{visible_devices}}" \\
              -e MULTI_GPU_TRAINING_IMAGE_REF="${{MULTI_GPU_TRAINING_IMAGE_REF}}" \\
              -e RUN_ID="${{RUN_ID}}" \\
              -e RUN_DIR="${{container_run_dir}}" \\
              -e NCCL_DEBUG="${{NCCL_DEBUG}}" \\
              -e NCCL_DEBUG_SUBSYS="${{NCCL_DEBUG_SUBSYS}}" \\
              -v "${{HOST_ARTIFACT_ROOT}}:${{CONTAINER_ARTIFACT_ROOT}}" \\
              -v "${{HOST_DATA_DIR}}:${{CONTAINER_DATA_DIR}}:ro" \\
              "${{IMAGE_REF}}" \\
              "${{unit_command[@]}}" \\
              2>&1 | tee "${{host_exp_dir}}/raw/docker-run-${{run_unit}}.log"
          else
            timeout "${{timeout_seconds}}" docker run --rm --gpus all --ipc=host \\
              --name "${{container_name}}" \\
              --shm-size "${{SHM_SIZE}}" \\
              --ulimit memlock=-1 --ulimit stack=67108864 \\
              -e CUDA_VISIBLE_DEVICES="${{visible_devices}}" \\
              -e MULTI_GPU_TRAINING_IMAGE_REF="${{MULTI_GPU_TRAINING_IMAGE_REF}}" \\
              -e TOKENIZERS_PARALLELISM="${{TOKENIZERS_PARALLELISM}}" \\
              -e NCCL_DEBUG="${{NCCL_DEBUG}}" \\
              -e NCCL_DEBUG_SUBSYS="${{NCCL_DEBUG_SUBSYS}}" \\
              -e TORCH_NCCL_ASYNC_ERROR_HANDLING="${{TORCH_NCCL_ASYNC_ERROR_HANDLING}}" \\
              -e TORCH_DISTRIBUTED_DEBUG="${{TORCH_DISTRIBUTED_DEBUG}}" \\
              -e CUDA_DEVICE_MAX_CONNECTIONS="${{CUDA_DEVICE_MAX_CONNECTIONS}}" \\
              -v "${{HOST_ARTIFACT_ROOT}}:${{CONTAINER_ARTIFACT_ROOT}}" \\
              -v "${{HOST_DATA_DIR}}:${{CONTAINER_DATA_DIR}}:ro" \\
              "${{IMAGE_REF}}" \\
              python "${{runner}}" --execute --run-unit "${{run_unit}}" --run-id "${{RUN_ID}}" \\
                --image-ref "${{IMAGE_REF}}" --output-root "${{CONTAINER_ARTIFACT_ROOT}}" \\
              2>&1 | tee "${{host_exp_dir}}/raw/docker-run-${{run_unit}}.log"
          fi
          local docker_status="${{PIPESTATUS[0]}}"
          set -e
          echo "${{docker_status}}" > "${{host_exp_dir}}/exit_status-${{run_unit}}.txt"
          date -u +%Y-%m-%dT%H:%M:%SZ > "${{host_exp_dir}}/finished-${{run_unit}}-utc.txt"
          aws s3 sync "${{host_exp_dir}}/" "${{durable_uri%/}}/runs/${{RUN_ID}}/" \\
            --region "${{AWS_REGION}}" --only-show-errors
          if [[ "${{docker_status}}" != "0" ]]; then
            echo "${{run_unit}} failed with status ${{docker_status}}" >&2
            return "${{docker_status}}"
          fi
          echo "Finished ${{run_unit}} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        }}

        queue_status=0
        __RUN_UNIT_INVOCATIONS__
        exit "${{queue_status}}"
        """
    )
    return (
        script.replace("__CACHE_VOLUME_SETUP__", cache_volume_user_data(launch_config))
        .replace("__ENV_EXPORTS__", env_exports(config))
        .replace("__RUN_UNIT_INVOCATIONS__", run_unit_invocations(config))
    )


def syntax_check_script(script: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sh") as handle:
        handle.write(script)
        handle.flush()
        result = subprocess.run(["bash", "-n", handle.name], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise QueueError(f"generated host script is not valid bash:\n{result.stderr}")


def launch_confirmation_phrase(config: dict[str, Any]) -> str:
    queue = config["queue"]
    return f"launch {queue['id']} {queue['compute_profile']} stop-after-{queue['maximum_lifetime_minutes']}m"


def queue_tags(config: dict[str, Any], launch_config: dict[str, Any], run_id: str) -> list[dict[str, str]]:
    tags = {
        "Project": "NCP-GENL",
        "Experiment": config["queue"]["id"],
        "ManagedBy": "codex-aws-a2-queue",
        "RunId": run_id,
        "ComputeProfile": config["queue"]["compute_profile"],
        "InstanceType": launch_config["compute"]["instance_type"],
        "ImageDigest": config["image"]["digest"],
        "GitCommit": git_commit(),
        "MaxLifetimeMinutes": str(config["queue"]["maximum_lifetime_minutes"]),
    }
    return [{"Key": key, "Value": str(value)} for key, value in sorted(tags.items())]


def build_launch_request(
    config: dict[str, Any],
    launch_config: dict[str, Any],
    script: str,
    run_id: str,
) -> dict[str, Any]:
    host = launch_config["host"]
    network = launch_config["network"]
    request: dict[str, Any] = {
        "ImageId": host["ami_id"],
        "InstanceType": launch_config["compute"]["instance_type"],
        "MinCount": 1,
        "MaxCount": 1,
        "IamInstanceProfile": {"Name": launch_config["identity"]["instance_profile"]},
        "InstanceInitiatedShutdownBehavior": "stop",
        "MetadataOptions": {
            "HttpEndpoint": launch_config["safety"]["metadata_options"]["http_endpoint"],
            "HttpTokens": launch_config["safety"]["metadata_options"]["http_tokens"],
            "InstanceMetadataTags": launch_config["safety"]["metadata_options"]["instance_metadata_tags"],
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
            {"ResourceType": resource, "Tags": queue_tags(config, launch_config, run_id)}
            for resource in ("instance", "volume", "network-interface")
        ],
        "UserData": base64.b64encode(script.encode("utf-8")).decode("ascii"),
    }
    if auto_select_subnet(launch_config):
        request["SecurityGroupIds"] = network["security_group_ids"]
    else:
        request["NetworkInterfaces"] = [
            {
                "DeviceIndex": 0,
                "SubnetId": selected_subnet(launch_config),
                "Groups": network["security_group_ids"],
                "AssociatePublicIpAddress": bool(network["associate_public_ip_address"]),
            }
        ]
    key_name = network.get("ssh_key_name")
    if key_name:
        request["KeyName"] = key_name
    return request


def launch_summary(
    config: dict[str, Any],
    launch_config: dict[str, Any],
    request: dict[str, Any],
    script: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "queue": config["queue"]["id"],
        "run_id": run_id,
        "aws_profile": config["aws"]["profile"],
        "region": config["aws"]["region"],
        "ami_id": request["ImageId"],
        "instance_type": request["InstanceType"],
        "compute_profile": config["queue"]["compute_profile"],
        "subnet_id": request.get("NetworkInterfaces", [{"SubnetId": "auto-default-vpc"}])[0]["SubnetId"],
        "security_group_ids": request.get("SecurityGroupIds") or request["NetworkInterfaces"][0]["Groups"],
        "instance_profile": request["IamInstanceProfile"]["Name"],
        "image": image_reference(config),
        "cache_volume": cache_volume_summary(launch_config),
        "maximum_lifetime_minutes": config["queue"]["maximum_lifetime_minutes"],
        "post_queue_inspection_minutes": config["queue"]["post_queue_inspection_minutes"],
        "stop_on_success": bool(config["queue"].get("stop_on_success", True)),
        "stop_on_failure": bool(config["queue"].get("stop_on_failure", True)),
        "shutdown_behavior": request["InstanceInitiatedShutdownBehavior"],
        "confirmation_phrase": launch_confirmation_phrase(config),
        "user_data_sha256": hashlib.sha256(script.encode("utf-8")).hexdigest(),
    }


def dry_run_launch(launch_config: dict[str, Any], request: dict[str, Any]) -> dict[str, str]:
    validate_cache_volume_for_launch(launch_config)
    result = run_aws(launch_config, request, dry_run=True)
    combined = f"{result.stdout}\n{result.stderr}"
    if "DryRunOperation" in combined:
        return {"status": "ok", "detail": "EC2 RunInstances dry run authorized"}
    raise QueueError(
        "EC2 RunInstances dry run failed\n"
        f"stdout: {result.stdout.strip()}\n"
        f"stderr: {result.stderr.strip()}"
    )


def real_launch(
    config: dict[str, Any],
    launch_config: dict[str, Any],
    request: dict[str, Any],
    confirm: str,
) -> dict[str, Any]:
    expected = launch_confirmation_phrase(config)
    if confirm != expected:
        raise QueueError(f"confirmation phrase mismatch; expected: {expected!r}")
    validate_cache_volume_for_launch(launch_config)
    result = run_aws(launch_config, request, dry_run=False)
    if result.returncode != 0:
        raise QueueError(
            "EC2 RunInstances failed\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    launch_response = json.loads(result.stdout)
    instances = launch_response.get("Instances", [])
    if len(instances) != 1:
        raise QueueError(f"expected one launched instance, got {len(instances)}")
    instance_id = instances[0]["InstanceId"]
    try:
        wait_instance_running(launch_config, instance_id)
        attach_response = attach_cache_volume(launch_config, instance_id)
    except Exception as error:
        run_aws_command(launch_config, "ec2", "stop-instances", "--instance-ids", instance_id)
        raise QueueError(f"launch created {instance_id}, but bootstrap attachment failed; stop requested") from error
    return {"run_instances": launch_response, "cache_volume_attachment": attach_response}


def render(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    try:
        config = load_yaml(args.queue_config)
        validate_queue(config)
        launch_config = load_launch_config(REPOSITORY_ROOT / config["queue"]["qualification_config"])
        run_id = validate_run_id(args.run_id or utc_run_id())
        if args.command == "plan":
            render(render_plan(config))
            return 0

        script = host_script(config, launch_config, run_id)
        syntax_check_script(script)
        if args.command == "host-script":
            print(script)
            return 0
        if args.command == "run":
            instance = select_running_instance(launch_config, args.instance_id)
            return send_ssm_command(
                launch_config,
                instance["InstanceId"],
                [script],
                args.timeout_seconds,
                f"{config['queue']['id']} sequential experiment queue",
            )

        request = build_launch_request(config, launch_config, script, run_id)
        summary = launch_summary(config, launch_config, request, script, run_id)
        if args.command == "launch-dry-run":
            result = dry_run_launch(launch_config, request)
            render({"mode": "launch-dry-run", "summary": summary, "result": result})
            return 0
        if args.command == "launch":
            result = real_launch(config, launch_config, request, args.confirm)
            render({"mode": "launch", "summary": summary, "result": {"status": "launched", "detail": result}})
            return 0
        raise QueueError(f"unsupported command: {args.command}")
    except QueueError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
