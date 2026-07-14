#!/usr/bin/env python3
"""Operator helpers for an already-running EXP-01 EC2 instance.

These commands are intentionally limited to inspection and manual interaction.
They do not launch, stop, or terminate EC2 instances.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "infra" / "aws" / "exp01_qualification.yaml"


class OpsError(RuntimeError):
    """Raised when an operator command cannot be completed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--instance-id", default="", help="Override automatic EXP-01 instance selection.")
    parser.add_argument("--run-id", default="", help="Override run ID for container helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="List matching EXP-01 EC2 instances and SSM state.")
    subparsers.add_parser("shell", help="Open an interactive SSM shell on the host.")
    subparsers.add_parser("container-shell", help="Open an interactive shell in the EXP-01 container.")
    subparsers.add_parser("logs", help="Collect key host, Docker, cloud-init, and experiment logs.")
    subparsers.add_parser("monitor", help="Collect a one-shot host/GPU/Docker monitoring snapshot.")
    subparsers.add_parser("artifacts", help="List EXP-01 S3 run artifacts.")

    host_command = subparsers.add_parser("command", help="Run a non-interactive shell command on the host.")
    host_command.add_argument("--shell-command", "-c", required=True)
    host_command.add_argument("--timeout-seconds", type=int, default=600)

    container_command = subparsers.add_parser(
        "container-command",
        help="Run a non-interactive shell command inside the EXP-01 container.",
    )
    container_command.add_argument("--shell-command", "-c", required=True)
    container_command.add_argument("--timeout-seconds", type=int, default=600)

    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise OpsError(f"configuration is not a mapping: {path}")
    return data


def run_command(command: list[str], allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0 and not allow_failure:
        raise OpsError(
            f"command failed: {' '.join(command)}\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
    return result


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


def aws_json(config: dict[str, Any], *args: str, allow_failure: bool = False) -> Any:
    result = run_command(aws_command(config, *args), allow_failure=allow_failure)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout or "{}")


def tags_to_dict(tags: list[dict[str, str]]) -> dict[str, str]:
    return {tag["Key"]: tag["Value"] for tag in tags}


def exp01_instance_filters(config: dict[str, Any], states: list[str]) -> list[str]:
    required = config["safety"]["required_tags"]
    return [
        f"Name=tag:Project,Values={required['Project']}",
        f"Name=tag:Experiment,Values={required['Experiment']}",
        f"Name=tag:ManagedBy,Values={required['ManagedBy']}",
        f"Name=instance-state-name,Values={','.join(states)}",
    ]


def matching_instances(config: dict[str, Any], states: list[str]) -> list[dict[str, Any]]:
    data = aws_json(
        config,
        "ec2",
        "describe-instances",
        "--filters",
        *exp01_instance_filters(config, states),
    )
    instances = [
        instance
        for reservation in data.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    return sorted(instances, key=lambda item: item.get("LaunchTime", ""))


def describe_instance(config: dict[str, Any], instance_id: str) -> dict[str, Any]:
    data = aws_json(
        config,
        "ec2",
        "describe-instances",
        "--instance-ids",
        instance_id,
    )
    instances = [
        instance
        for reservation in data.get("Reservations", [])
        for instance in reservation.get("Instances", [])
    ]
    if len(instances) != 1:
        raise OpsError(f"expected one instance for {instance_id}, got {len(instances)}")
    return instances[0]


def select_running_instance(config: dict[str, Any], instance_id: str = "") -> dict[str, Any]:
    if instance_id:
        instance = describe_instance(config, instance_id)
        state = instance.get("State", {}).get("Name")
        if state != "running":
            raise OpsError(f"{instance_id} is {state}, not running")
        return instance

    instances = matching_instances(config, ["running"])
    if not instances:
        raise OpsError("no running EXP-01 instance found")
    if len(instances) > 1:
        ids = ", ".join(instance["InstanceId"] for instance in instances)
        raise OpsError(f"multiple running EXP-01 instances found; pass INSTANCE_ID=... ({ids})")
    return instances[0]


def instance_run_id(instance: dict[str, Any], override: str = "") -> str:
    if override:
        return override
    tags = tags_to_dict(instance.get("Tags", []))
    run_id = tags.get("RunId")
    if not run_id:
        raise OpsError("run ID is not tagged on the instance; pass RUN_ID=...")
    return run_id


def container_name(run_id: str) -> str:
    return f"exp01-{run_id}"


def ssm_ping_status(config: dict[str, Any], instance_ids: list[str]) -> dict[str, str]:
    if not instance_ids:
        return {}
    data = aws_json(
        config,
        "ssm",
        "describe-instance-information",
        "--filters",
        f"Key=InstanceIds,Values={','.join(instance_ids)}",
        allow_failure=True,
    )
    if data is None:
        return {}
    return {
        item["InstanceId"]: item.get("PingStatus", "unknown")
        for item in data.get("InstanceInformationList", [])
    }


def render_status(config: dict[str, Any]) -> None:
    instances = matching_instances(config, ["pending", "running", "stopping", "stopped"])
    ssm_status = ssm_ping_status(config, [instance["InstanceId"] for instance in instances])
    if not instances:
        print("No EXP-01 instances found.")
        return

    print("InstanceId State SSM InstanceType AZ RunId LaunchTime")
    for instance in instances:
        tags = tags_to_dict(instance.get("Tags", []))
        placement = instance.get("Placement", {})
        print(
            instance["InstanceId"],
            instance.get("State", {}).get("Name", "unknown"),
            ssm_status.get(instance["InstanceId"], "not-registered"),
            instance.get("InstanceType", "unknown"),
            placement.get("AvailabilityZone", "unknown"),
            tags.get("RunId", "-"),
            instance.get("LaunchTime", "-"),
        )


def send_ssm_command(
    config: dict[str, Any],
    instance_id: str,
    commands: list[str],
    timeout_seconds: int,
    comment: str,
) -> int:
    payload = {
        "InstanceIds": [instance_id],
        "DocumentName": "AWS-RunShellScript",
        "Comment": comment,
        "Parameters": {
            "commands": commands,
            "executionTimeout": [str(timeout_seconds)],
        },
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json") as handle:
        json.dump(payload, handle)
        handle.flush()
        result = aws_json(
            config,
            "ssm",
            "send-command",
            "--cli-input-json",
            f"file://{handle.name}",
        )
    command_id = result["Command"]["CommandId"]
    wait_command = aws_command(
        config,
        "ssm",
        "wait",
        "command-executed",
        "--command-id",
        command_id,
        "--instance-id",
        instance_id,
    )
    run_command(wait_command, allow_failure=True)
    invocation = aws_json(
        config,
        "ssm",
        "get-command-invocation",
        "--command-id",
        command_id,
        "--instance-id",
        instance_id,
    )
    stdout = invocation.get("StandardOutputContent", "")
    stderr = invocation.get("StandardErrorContent", "")
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr)
    status = invocation.get("Status", "Unknown")
    if status != "Success":
        print(f"SSM command status: {status}", file=sys.stderr)
        return int(invocation.get("ResponseCode", 1) or 1)
    return 0


def require_session_plugin() -> None:
    if shutil.which("session-manager-plugin") is None:
        raise OpsError(
            "session-manager-plugin is not installed locally; install it before "
            "using interactive SSM shell targets. Non-interactive SSM command "
            "targets still work without the plugin."
        )


def start_session(config: dict[str, Any], instance_id: str, command: str = "") -> int:
    require_session_plugin()
    aws_args = aws_command(config, "ssm", "start-session", "--target", instance_id)
    if command:
        aws_args.extend(
            [
                "--document-name",
                "AWS-StartInteractiveCommand",
                "--parameters",
                json.dumps({"command": [command]}),
            ]
        )
    completed = subprocess.run(aws_args, check=False)
    return completed.returncode


def host_logs_command() -> str:
    return r"""
set -uo pipefail
echo "== host =="
date -u
hostname
uname -a
echo
echo "== uptime/memory/disk =="
uptime || true
free -h || true
df -h || true
echo
echo "== nvidia-smi =="
nvidia-smi || true
echo
echo "== docker ps =="
sudo docker ps -a --no-trunc || true
echo
echo "== docker images =="
sudo docker images || true
echo
echo "== exp01 run tree =="
sudo find /opt/multi-gpu-training/artifacts/runs/EXP-01 -maxdepth 4 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -80 || true
echo
echo "== exp01 user-data log =="
sudo tail -n 240 /var/log/exp01-user-data.log 2>/dev/null || true
echo
echo "== cloud-init output =="
sudo tail -n 160 /var/log/cloud-init-output.log 2>/dev/null || true
"""


def monitor_command() -> str:
    return r"""
set -uo pipefail
date -u
uptime || true
free -h || true
df -h / /opt /tmp 2>/dev/null || df -h || true
lsblk || true
echo
nvidia-smi || true
echo
nvidia-smi topo -m || true
echo
sudo docker ps --no-trunc || true
echo
sudo docker stats --no-stream --no-trunc 2>/dev/null || true
"""


def artifact_uri(config: dict[str, Any]) -> str:
    parsed = urlparse(config["artifacts"]["durable_uri"])
    if parsed.scheme != "s3" or not parsed.netloc:
        raise OpsError(f"invalid S3 artifact URI: {config['artifacts']['durable_uri']}")
    return f"s3://{parsed.netloc}/{parsed.path.lstrip('/')}"


def list_artifacts(config: dict[str, Any]) -> int:
    command = [
        "aws",
        "s3",
        "ls",
        f"{artifact_uri(config)}runs/",
        "--recursive",
        "--human-readable",
        "--summarize",
        "--profile",
        config["aws"]["profile"],
        "--region",
        config["aws"]["region"],
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr)
    if completed.returncode != 0 and "Total Objects: 0" in completed.stdout:
        return 0
    return completed.returncode


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        if args.command == "status":
            render_status(config)
            return 0
        if args.command == "artifacts":
            return list_artifacts(config)

        instance = select_running_instance(config, args.instance_id)
        instance_id = instance["InstanceId"]

        if args.command == "shell":
            return start_session(config, instance_id)
        if args.command == "container-shell":
            run_id = instance_run_id(instance, args.run_id)
            return start_session(
                config,
                instance_id,
                command=f"sudo docker exec -it {container_name(run_id)} bash",
            )
        if args.command == "command":
            return send_ssm_command(
                config,
                instance_id,
                [args.shell_command],
                args.timeout_seconds,
                "EXP-01 manual host command",
            )
        if args.command == "container-command":
            run_id = instance_run_id(instance, args.run_id)
            command = (
                f"sudo docker exec {container_name(run_id)} "
                f"bash -lc {shlex.quote(args.shell_command)}"
            )
            return send_ssm_command(
                config,
                instance_id,
                [command],
                args.timeout_seconds,
                "EXP-01 manual container command",
            )
        if args.command == "logs":
            return send_ssm_command(
                config,
                instance_id,
                [host_logs_command()],
                600,
                "EXP-01 host logs",
            )
        if args.command == "monitor":
            return send_ssm_command(
                config,
                instance_id,
                [monitor_command()],
                300,
                "EXP-01 monitor snapshot",
            )
    except OpsError as error:
        print(str(error), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
