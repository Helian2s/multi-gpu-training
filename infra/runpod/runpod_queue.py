#!/usr/bin/env python3
"""Runpod queue planning and container-side execution helpers.

This module deliberately does not create paid resources by default. It renders
the exact Pod create command and the container-side queue script so the operator
can review them before an approved launch.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import textwrap
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_A2_QUEUE_CONFIG = REPOSITORY_ROOT / "infra" / "runpod" / "a2_megatron_queue.yaml"


class RunpodQueueError(RuntimeError):
    """Raised when a Runpod queue is unsafe or incomplete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-config", type=Path, default=DEFAULT_A2_QUEUE_CONFIG)
    parser.add_argument("--run-id", default="")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Print launch/readiness plan without side effects.")
    subparsers.add_parser("container-script", help="Print the generated container queue script.")
    subparsers.add_parser(
        "pod-create-command",
        help="Print the reviewed runpodctl pod create command; does not execute it.",
    )
    subparsers.add_parser(
        "container-run",
        help="Run the generated queue script inside an already-started Runpod container.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RunpodQueueError(f"configuration is not a mapping: {path}")
    return data


def utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def validate_run_id(run_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise RunpodQueueError(f"run ID contains unsupported characters: {run_id!r}")
    return run_id


def image_reference(config: dict[str, Any], *, require_digest: bool = True) -> str:
    image = config["image"]
    digest = image.get("digest")
    if not digest:
        if require_digest:
            raise RunpodQueueError("image.digest is missing; publish the GHCR image first")
        tag = image.get("tag") or "unpublished"
        return f"{image['registry']}/{image['repository']}:{tag}"
    return f"{image['registry']}/{image['repository']}@{digest}"


def validate_queue(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise RunpodQueueError("queue schema_version must be 1")
    queue = config.get("queue")
    if not isinstance(queue, dict):
        raise RunpodQueueError("queue mapping is required")
    if not str(queue.get("id", "")).startswith("RUNPOD-"):
        raise RunpodQueueError("queue.id must be a RUNPOD-* queue label")
    if queue.get("compute_profile") not in {"RUNPOD-A100-SXM2", "RUNPOD-A100-SXM4"}:
        raise RunpodQueueError(f"unsupported compute profile: {queue.get('compute_profile')!r}")
    seen: set[str] = set()
    for item in config.get("run_units", []):
        run_unit = item.get("run_unit")
        experiment_id = item.get("experiment_id")
        if not isinstance(run_unit, str) or not run_unit.startswith(("EXP-", "QUAL-")):
            raise RunpodQueueError(f"invalid run_unit: {run_unit!r}")
        if run_unit in seen:
            raise RunpodQueueError(f"duplicate run unit: {run_unit}")
        seen.add(run_unit)
        if not isinstance(experiment_id, str) or not experiment_id.startswith(("EXP-", "QUAL-")):
            raise RunpodQueueError(f"invalid experiment_id: {experiment_id!r}")
        visible_devices = item.get("visible_devices")
        if not isinstance(visible_devices, str) or not visible_devices:
            raise RunpodQueueError(f"run unit {run_unit} has invalid visible_devices")
        timeout_seconds = item.get("timeout_seconds")
        if not isinstance(timeout_seconds, int) or timeout_seconds <= 0:
            raise RunpodQueueError(f"run unit {run_unit} has invalid timeout_seconds")
        kind = item.get("kind")
        if kind == "runner":
            runner = REPOSITORY_ROOT / item["runner"]
            if not runner.is_file():
                raise RunpodQueueError(f"runner does not exist: {runner}")
        elif kind == "shell":
            command = item.get("command")
            if not isinstance(command, list) or not command or not all(
                isinstance(part, str) for part in command
            ):
                raise RunpodQueueError(f"shell run unit {run_unit} must define a command list")
        else:
            raise RunpodQueueError(f"run unit {run_unit} has unsupported kind: {kind!r}")
    expected = queue.get("expected_run_units")
    if expected is not None:
        actual = [item["run_unit"] for item in config.get("run_units", [])]
        if actual != expected:
            raise RunpodQueueError(f"Runpod queue order must be {expected}; got {actual}")


def launch_ready(config: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = []
    if not config["image"].get("digest"):
        missing.append("image.digest")
    if not config["runpod"].get("registry_auth_id"):
        missing.append("runpod.registry_auth_id")
    if not config["runpod"].get("network_volume_id") and not config["runpod"].get("pod_volume_gb"):
        missing.append("runpod.network_volume_id or runpod.pod_volume_gb")
    return not missing, missing


def render_plan(config: dict[str, Any]) -> dict[str, Any]:
    ready, missing = launch_ready(config)
    return {
        "queue": config["queue"]["id"],
        "compute_profile": config["queue"]["compute_profile"],
        "image_ready": bool(config["image"].get("digest")),
        "image": image_reference(config, require_digest=False),
        "launch_ready": ready,
        "missing_for_launch": missing,
        "gpu_id": config["runpod"]["gpu_id"],
        "gpu_count": config["runpod"]["gpu_count"],
        "cloud_type": config["runpod"]["cloud_type"],
        "min_cuda_version": config["runpod"].get("min_cuda_version") or None,
        "network_volume_id": config["runpod"].get("network_volume_id") or None,
        "pod_volume_gb": config["runpod"].get("pod_volume_gb") or None,
        "network_volume_mount_path": config["runpod"]["network_volume_mount_path"],
        "ports": config["runpod"].get("ports", []),
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
                "timeout_seconds": item["timeout_seconds"],
            }
            for item in config["run_units"]
        ],
    }


def env_exports(config: dict[str, Any]) -> str:
    return "\n".join(
        f"export {key}={shlex.quote(str(value))}"
        for key, value in sorted(config["container"].get("environment", {}).items())
    )


def device_count(visible_devices: str) -> int:
    return len([item for item in visible_devices.split(",") if item.strip()])


def run_unit_script(config: dict[str, Any]) -> str:
    lines: list[str] = []
    for item in config["run_units"]:
        run_unit = item["run_unit"]
        experiment_id = item["experiment_id"]
        exports = {
            "RUN_UNIT": run_unit,
            "EXPERIMENT_ID": experiment_id,
            "CUDA_VISIBLE_DEVICES": item["visible_devices"],
            "EXPECTED_DEVICES": str(device_count(item["visible_devices"])),
        }
        if item["kind"] == "shell":
            command = " ".join(shlex.quote(part) for part in item["command"])
        else:
            command = (
                f"python {shlex.quote(item['runner'])} --execute "
                '"--run-id" "${RUN_ID}" '
                f"--run-unit {shlex.quote(run_unit)} "
                '"--output-root" "${ARTIFACT_ROOT}" '
                '"--image-ref" "${IMAGE_REF}"'
            )
        lines.extend(
            [
                'if [[ "${queue_status}" == "0" ]]; then',
                f"  echo 'Starting {run_unit} at '$(date -u +%Y-%m-%dT%H:%M:%SZ)",
                "  (",
                *(f"    export {key}={shlex.quote(value)}" for key, value in exports.items()),
                f"    export RUN_DIR=\"${{ARTIFACT_ROOT}}/{experiment_id}/${{RUN_ID}}\"",
                "    mkdir -p \"${RUN_DIR}\"",
                f"    timeout --preserve-status {item['timeout_seconds']} bash -lc {shlex.quote(command)}",
                f"  ) || queue_status=\"$?\"",
                f"  echo 'Finished {run_unit} with queue_status='\"${{queue_status}}\"' at '$(date -u +%Y-%m-%dT%H:%M:%SZ)",
                "fi",
            ]
        )
    return "\n".join(lines)


def container_script(config: dict[str, Any], run_id: str) -> str:
    queue = config["queue"]
    image = image_reference(config, require_digest=False)
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        QUEUE_ID={shlex.quote(queue['id'])}
        RUN_ID={shlex.quote(validate_run_id(run_id))}
        IMAGE_REF={shlex.quote(image)}
        REPO_ROOT={shlex.quote(config['container']['repo_root'])}
        ARTIFACT_ROOT={shlex.quote(config['artifacts']['network_volume_root'])}
        QUEUE_LOG_DIR="${{ARTIFACT_ROOT}}/${{QUEUE_ID}}/${{RUN_ID}}"
        QUEUE_LOG_FILE="${{QUEUE_LOG_DIR}}/queue.log"
        MAX_LIFETIME_MINUTES={shlex.quote(str(queue['maximum_lifetime_minutes']))}
        POST_QUEUE_INSPECTION_MINUTES={shlex.quote(str(queue['post_queue_inspection_minutes']))}
        STOP_ON_SUCCESS={shlex.quote(str(bool(queue.get('stop_on_success', True))).lower())}
        STOP_ON_FAILURE={shlex.quote(str(bool(queue.get('stop_on_failure', True))).lower())}

        mkdir -p "${{QUEUE_LOG_DIR}}" "${{ARTIFACT_ROOT}}"
        exec > >(tee -a "${{QUEUE_LOG_FILE}}") 2>&1

        cd "${{REPO_ROOT}}"
        export RUN_ID
        export IMAGE_REF
        export ARTIFACT_ROOT
        export MULTI_GPU_TRAINING_IMAGE_REF="${{IMAGE_REF}}"
        export PYTHONPATH="${{REPO_ROOT}}${{PYTHONPATH:+:${{PYTHONPATH}}}}"
        {env_exports(config)}

        echo "Runpod queue ${{QUEUE_ID}} started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "RUN_ID=${{RUN_ID}}"
        echo "IMAGE_REF=${{IMAGE_REF}}"
        echo "ARTIFACT_ROOT=${{ARTIFACT_ROOT}}"
        echo "Hard lifetime is enforced by the Runpod Pod terminate-after setting: ${{MAX_LIFETIME_MINUTES}} minutes."

        queue_status=0
        {run_unit_script(config)}

        echo "Runpod queue ${{QUEUE_ID}} completed with status ${{queue_status}} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        if [[ "${{queue_status}}" != "0" && "${{STOP_ON_FAILURE}}" != "true" ]]; then
          echo "Queue failed; leaving Pod available for inspection until the hard terminate-after guard."
        elif [[ "${{queue_status}}" == "0" && "${{STOP_ON_SUCCESS}}" == "true" ]]; then
          echo "Queue succeeded; operator should stop or delete the Pod after verifying artifacts."
        fi
        exit "${{queue_status}}"
        """
    )
    return script


def pod_create_command(config: dict[str, Any], run_id: str) -> str:
    ready, missing = launch_ready(config)
    if not ready:
        raise RunpodQueueError(
            "queue is not launch-ready; missing " + ", ".join(missing)
        )
    runpod = config["runpod"]
    minutes = int(config["queue"]["maximum_lifetime_minutes"])
    terminate_after = (datetime.now(UTC) + timedelta(minutes=minutes)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    command = [
        "runpodctl",
        "pod",
        "create",
        "--cloud-type",
        runpod["cloud_type"],
        "--image",
        image_reference(config),
        "--gpu-id",
        runpod["gpu_id"],
        "--gpu-count",
        str(runpod["gpu_count"]),
    ]
    if runpod.get("min_cuda_version"):
        command.extend(["--min-cuda-version", str(runpod["min_cuda_version"])])
    command.extend(
        [
            "--container-disk-in-gb",
            str(runpod["container_disk_gb"]),
            "--volume-mount-path",
            runpod["network_volume_mount_path"],
            "--registry-auth-id",
            runpod["registry_auth_id"],
            "--name",
            f"{config['queue']['id'].lower()}-{validate_run_id(run_id).lower()}",
            "--terminate-after",
            terminate_after,
        ]
    )
    if runpod.get("network_volume_id"):
        command.extend(["--network-volume-id", runpod["network_volume_id"]])
    else:
        command.extend(["--volume-in-gb", str(runpod["pod_volume_gb"])])
    ports = runpod.get("ports") or []
    if ports:
        command.extend(["--ports", ",".join(ports)])
    data_center_ids = runpod.get("data_center_ids") or []
    if data_center_ids:
        command.extend(["--data-center-ids", ",".join(data_center_ids)])
    env = json.dumps(config["container"].get("environment", {}), separators=(",", ":"))
    if env != "{}":
        command.extend(["--env", env])
    rendered = " ".join(shlex.quote(part) for part in command)
    return rendered


def main() -> int:
    args = parse_args()
    run_id = args.run_id or utc_run_id()
    try:
        config = load_yaml(args.queue_config)
        validate_queue(config)
        if args.command == "plan":
            print(json.dumps(render_plan(config), indent=2, sort_keys=True))
            return 0
        if args.command == "container-script":
            print(container_script(config, run_id))
            return 0
        if args.command == "pod-create-command":
            print(pod_create_command(config, run_id))
            return 0
        if args.command == "container-run":
            completed = subprocess.run(
                ["bash"],
                input=container_script(config, run_id),
                check=False,
                text=True,
            )
            return completed.returncode
        raise RunpodQueueError(f"unsupported command: {args.command}")
    except RunpodQueueError as error:
        print(f"runpod queue error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
