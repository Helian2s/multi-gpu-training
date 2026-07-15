#!/usr/bin/env python3
"""Operator tooling for the AWS-A1 PyTorch experiment queue.

This module never starts EC2 instances. It plans or sends a guarded shell script
to an already-running AWS-A1 host selected through SSM.
"""

from __future__ import annotations

import argparse
import json
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

from infra.aws.exp01_launch import cache_volume_user_data, load_config as load_launch_config
from infra.aws.exp01_ops import select_running_instance, send_ssm_command


DEFAULT_QUEUE_CONFIG = REPOSITORY_ROOT / "infra" / "aws" / "a1_experiment_queue.yaml"


class QueueError(RuntimeError):
    """Raised when the AWS-A1 queue is unsafe or incomplete."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-config", type=Path, default=DEFAULT_QUEUE_CONFIG)
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--run-id", default="")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan", help="Print the queue plan without contacting EC2.")
    subparsers.add_parser("host-script", help="Print the generated host script.")
    run = subparsers.add_parser("run", help="Run the queue on an already-running AWS-A1 host via SSM.")
    run.add_argument("--timeout-seconds", type=int, default=18000)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise QueueError(f"configuration is not a mapping: {path}")
    return data


def utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def image_reference(config: dict[str, Any]) -> str:
    image = config["image"]
    digest = image.get("digest")
    if not digest:
        raise QueueError("queue image.digest is missing; build and publish the AWS-A1 image first")
    return f"{image['registry']}/{image['repository']}@{digest}"


def validate_queue(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise QueueError("queue schema_version must be 1")
    if config["queue"]["compute_profile"] != "AWS-A1":
        raise QueueError("this runner only supports AWS-A1")
    seen: set[str] = set()
    for item in config.get("run_units", []):
        run_unit = item.get("run_unit")
        experiment_id = item.get("experiment_id")
        if not isinstance(run_unit, str) or not run_unit.endswith("-A1"):
            raise QueueError(f"invalid AWS-A1 run_unit: {run_unit!r}")
        if not isinstance(experiment_id, str) or not experiment_id.startswith("EXP-"):
            raise QueueError(f"invalid experiment_id: {experiment_id!r}")
        if run_unit in seen:
            raise QueueError(f"duplicate run unit: {run_unit}")
        seen.add(run_unit)
        runner = REPOSITORY_ROOT / item["runner"]
        if not runner.is_file():
            raise QueueError(f"runner does not exist: {runner}")
    expected = ["EXP-03-A1", "EXP-04-A1", "EXP-05-A1", "EXP-06-A1"]
    if [item["run_unit"] for item in config.get("run_units", [])] != expected:
        raise QueueError(f"AWS-A1 queue order must be {expected}")


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
        "run_units": [
            {
                "run_unit": item["run_unit"],
                "experiment_id": item["experiment_id"],
                "runner": item["runner"],
                "durable_uri": item["durable_uri"],
                "timeout_seconds": item["timeout_seconds"],
            }
            for item in config["run_units"]
        ],
    }


def bash_array(items: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in items)


def run_units_json(config: dict[str, Any]) -> str:
    return json.dumps(config["run_units"], separators=(",", ":"))


def host_script(config: dict[str, Any], launch_config: dict[str, Any], run_id: str) -> str:
    image = image_reference(config)
    queue = config["queue"]
    inputs = config["inputs"]
    artifacts = config["artifacts"]
    container = config["container"]
    env_exports = "\n".join(
        f"export {key}={shlex.quote(str(value))}"
        for key, value in sorted(container.get("environment", {}).items())
    )
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        exec > >(tee -a /var/log/aws-a1-queue.log) 2>&1

        QUEUE_ID="{queue['id']}"
        RUN_ID="{run_id}"
        AWS_REGION="{config['aws']['region']}"
        ECR_REGISTRY="{config['image']['registry']}"
        IMAGE_REF="{image}"
        INPUTS_S3_URI="{inputs['durable_uri'].rstrip('/')}/"
        HOST_DATA_DIR="{inputs['host_data_dir']}"
        CONTAINER_DATA_DIR="{inputs['container_data_dir']}"
        REQUIRED_INPUT_MANIFEST="{inputs['required_manifest']}"
        HOST_ARTIFACT_ROOT="{artifacts['host_root']}"
        CONTAINER_ARTIFACT_ROOT="{artifacts['container_root']}"
        CONTAINER_NAME_PREFIX="{container['name_prefix']}"
        CUDA_VISIBLE_DEVICES_VALUE="{container['cuda_visible_devices']}"
        SHM_SIZE="{container['shm_size']}"
        MAX_LIFETIME_MINUTES="{queue['maximum_lifetime_minutes']}"
        POST_QUEUE_INSPECTION_MINUTES="{queue['post_queue_inspection_minutes']}"

        echo "AWS-A1 queue started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "QUEUE_ID=${{QUEUE_ID}}"
        echo "RUN_ID=${{RUN_ID}}"
        echo "IMAGE_REF=${{IMAGE_REF}}"
        echo "INPUTS_S3_URI=${{INPUTS_S3_URI}}"

        sudo shutdown -c || true
        sudo shutdown -h +${{MAX_LIFETIME_MINUTES}} "AWS-A1 queue safety stop"

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

        run_experiment() {{
          local run_unit="$1"
          local experiment_id="$2"
          local runner="$3"
          local durable_uri="$4"
          local timeout_seconds="$5"
          local safe_experiment
          safe_experiment="$(echo "${{experiment_id}}" | tr '[:upper:]' '[:lower:]')"
          local container_name="${{CONTAINER_NAME_PREFIX}}-${{safe_experiment}}-${{RUN_ID}}"
          local host_exp_dir="${{HOST_ARTIFACT_ROOT}}/${{experiment_id}}/${{RUN_ID}}"
          mkdir -p "${{host_exp_dir}}/raw" "${{host_exp_dir}}/metrics"
          echo "Starting ${{run_unit}} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
          docker rm -f "${{container_name}}" >/dev/null 2>&1 || true
          set +e
          timeout "${{timeout_seconds}}" docker run --rm --gpus all --ipc=host \\
            --name "${{container_name}}" \\
            --shm-size "${{SHM_SIZE}}" \\
            --ulimit memlock=-1 --ulimit stack=67108864 \\
            -e CUDA_VISIBLE_DEVICES="${{CUDA_VISIBLE_DEVICES_VALUE}}" \\
            -e MULTI_GPU_TRAINING_IMAGE_REF="${{MULTI_GPU_TRAINING_IMAGE_REF}}" \\
            -e TOKENIZERS_PARALLELISM="${{TOKENIZERS_PARALLELISM}}" \\
            -e NCCL_DEBUG="${{NCCL_DEBUG}}" \\
            -v "${{HOST_ARTIFACT_ROOT}}:${{CONTAINER_ARTIFACT_ROOT}}" \\
            -v "${{HOST_DATA_DIR}}:${{CONTAINER_DATA_DIR}}:ro" \\
            "${{IMAGE_REF}}" \\
            python "${{runner}}" --execute --run-unit "${{run_unit}}" --run-id "${{RUN_ID}}" --image-ref "${{IMAGE_REF}}" \\
            2>&1 | tee "${{host_exp_dir}}/raw/docker-run-${{run_unit}}.log"
          local docker_status="${{PIPESTATUS[0]}}"
          set -e
          echo "${{docker_status}}" > "${{host_exp_dir}}/exit_status.txt"
          date -u +%Y-%m-%dT%H:%M:%SZ > "${{host_exp_dir}}/finished_utc.txt"
          aws s3 sync "${{host_exp_dir}}/" "${{durable_uri%/}}/runs/${{RUN_ID}}/" \\
            --region "${{AWS_REGION}}" --only-show-errors
          if [[ "${{docker_status}}" != "0" ]]; then
            echo "${{run_unit}} failed with status ${{docker_status}}" >&2
            return "${{docker_status}}"
          fi
          echo "Finished ${{run_unit}} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        }}

        python3 - <<'PY' >/tmp/aws-a1-run-units.tsv
        import json
        for item in json.loads('''__RUN_UNITS_JSON__'''):
            print("\\t".join([
                item["run_unit"],
                item["experiment_id"],
                item["runner"],
                item["durable_uri"],
                str(item["timeout_seconds"]),
            ]))
        PY

        while IFS=$'\\t' read -r run_unit experiment_id runner durable_uri timeout_seconds; do
          run_experiment "${{run_unit}}" "${{experiment_id}}" "${{runner}}" "${{durable_uri}}" "${{timeout_seconds}}"
        done </tmp/aws-a1-run-units.tsv

        cp /var/log/aws-a1-queue.log "${{HOST_ARTIFACT_ROOT}}/AWS-A1-PyTorch-${{RUN_ID}}.log" 2>/dev/null || true
        aws s3 cp "${{HOST_ARTIFACT_ROOT}}/AWS-A1-PyTorch-${{RUN_ID}}.log" \\
          "s3://finetuning-lab-1-037678282394-us-west-2-an/artifacts/AWS-A1-PyTorch/runs/${{RUN_ID}}/queue.log" \\
          --region "${{AWS_REGION}}" --only-show-errors || true

        echo "AWS-A1 queue finished; holding for ${{POST_QUEUE_INSPECTION_MINUTES}} minutes"
        sleep "$((POST_QUEUE_INSPECTION_MINUTES * 60))"
        shutdown -h now
        """
    )
    return (
        script.replace("__CACHE_VOLUME_SETUP__", cache_volume_user_data(launch_config))
        .replace("__ENV_EXPORTS__", env_exports)
        .replace("__RUN_UNITS_JSON__", run_units_json(config))
    )


def syntax_check_script(script: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sh") as handle:
        handle.write(script)
        handle.flush()
        result = subprocess.run(["bash", "-n", handle.name], check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise QueueError(f"generated host script is not valid bash:\n{result.stderr}")


def main() -> int:
    args = parse_args()
    try:
        config = load_yaml(args.queue_config)
        validate_queue(config)
        launch_config = load_launch_config(REPOSITORY_ROOT / config["queue"]["qualification_config"])
        run_id = args.run_id or utc_run_id()
        if args.command == "plan":
            print(json.dumps(render_plan(config), indent=2, sort_keys=True))
            return 0
        script = host_script(config, launch_config, run_id)
        syntax_check_script(script)
        if args.command == "host-script":
            print(script)
            return 0
        instance = select_running_instance(launch_config, args.instance_id)
        return send_ssm_command(
            launch_config,
            instance["InstanceId"],
            [script],
            args.timeout_seconds,
            f"{config['queue']['id']} sequential experiment queue",
        )
    except QueueError as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
