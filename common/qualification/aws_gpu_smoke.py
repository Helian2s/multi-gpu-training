#!/usr/bin/env python3
"""Container-side AWS GPU qualification smoke check."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any


class QualificationError(RuntimeError):
    """Raised when the selected GPU environment does not qualify."""


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_capture(command: list[str], output_path: Path, *, required: bool = True) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which(command[0]) is None:
        message = f"missing command: {command[0]}\n"
        output_path.write_text(message, encoding="utf-8")
        if required:
            raise QualificationError(message.strip())
        return 127
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    output_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if required and completed.returncode != 0:
        raise QualificationError(
            f"{' '.join(command)} failed with exit code {completed.returncode}; see {output_path}"
        )
    return completed.returncode


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def import_state(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        return {"available": False, "error": repr(error)}
    return {"available": True, "path": getattr(module, "__file__", None)}


def runtime_versions(torch: Any) -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": getattr(torch, "__version__", None),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "megatron_core": package_version("megatron-core"),
        "megatron_bridge": package_version("megatron-bridge"),
        "transformer_engine": package_version("transformer-engine"),
        "megatron_core_import": import_state("megatron.core"),
        "megatron_pipeline_schedules_import": import_state(
            "megatron.core.pipeline_parallel.schedules"
        ),
    }


def selected_environment() -> dict[str, str | None]:
    keys = [
        "CUDA_VISIBLE_DEVICES",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "TORCH_DISTRIBUTED_DEBUG",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "MULTI_GPU_TRAINING_IMAGE_REF",
        "RUN_ID",
        "RUN_DIR",
    ]
    return {key: os.environ.get(key) for key in keys}


def require_torch() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise QualificationError("PyTorch is required in the runtime image") from error
    return torch


def worker(output_dir: Path, expected_devices: int) -> int:
    torch = require_torch()
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != expected_devices:
        raise QualificationError(f"expected world_size={expected_devices}, got {world_size}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    device = torch.device(f"cuda:{local_rank}")
    value = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(value)
    expected_sum = float(world_size * (world_size + 1) / 2)
    if float(value.item()) != expected_sum:
        raise QualificationError(f"all_reduce sum mismatch: got {float(value.item())}")
    matrix = torch.randn((256, 256), device=device)
    product = matrix @ matrix
    torch.cuda.synchronize(device)
    payload = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device_name": torch.cuda.get_device_name(local_rank),
        "device_capability": torch.cuda.get_device_capability(local_rank),
        "all_reduce_sum": float(value.item()),
        "matmul_checksum": float(product.float().sum().item()),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    write_json(output_dir / f"rank_{rank}.json", payload)
    dist.barrier()
    dist.destroy_process_group()
    return 0


def execute(output_dir: Path, expected_devices: int) -> int:
    torch = require_torch()
    raw_dir = output_dir / "raw" / "qualification"
    metrics_dir = output_dir / "metrics"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    write_json(raw_dir / "environment.json", selected_environment())
    run_capture(["nvidia-smi", "-L"], raw_dir / "nvidia-smi-L.txt")
    run_capture(["nvidia-smi", "topo", "-m"], raw_dir / "nvidia-smi-topo-m.txt")
    run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total",
            "--format=csv",
        ],
        raw_dir / "nvidia-smi-gpu-query.csv",
    )

    versions = runtime_versions(torch)
    write_json(raw_dir / "runtime_versions.json", versions)
    if not torch.cuda.is_available():
        raise QualificationError("CUDA is not available")
    if torch.cuda.device_count() != expected_devices:
        raise QualificationError(
            f"expected {expected_devices} visible CUDA devices, got {torch.cuda.device_count()}"
        )
    if not versions["megatron_core_import"]["available"]:
        raise QualificationError("Megatron Core is not importable")
    if not versions["megatron_pipeline_schedules_import"]["available"]:
        raise QualificationError("Megatron pipeline schedules are not importable")

    if expected_devices > 1:
        worker_dir = raw_dir / "torchrun"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(expected_devices),
            str(Path(__file__).resolve()),
            "--worker",
            "--expected-devices",
            str(expected_devices),
            "--output-dir",
            str(worker_dir),
        ]
        log_path = raw_dir / "torchrun.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=600,
            )
        if completed.returncode != 0:
            raise QualificationError(
                f"torch distributed smoke failed with exit code {completed.returncode}; see {log_path}"
            )
    else:
        device = torch.device("cuda:0")
        tensor = torch.randn((256, 256), device=device)
        result = tensor @ tensor
        torch.cuda.synchronize(device)
        write_json(
            raw_dir / "single_gpu_smoke.json",
            {
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": torch.cuda.get_device_capability(0),
                "matmul_checksum": float(result.float().sum().item()),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            },
        )

    write_json(
        output_dir / "manifest.json",
        {
            "run_id": os.environ.get("RUN_ID"),
            "qualification": "AWS GPU container smoke",
            "expected_devices": expected_devices,
            "status": "completed",
        },
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("RUN_DIR", ".")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        output_dir = args.output_dir.resolve()
        if args.worker:
            return worker(output_dir, args.expected_devices)
        if args.execute:
            return execute(output_dir, args.expected_devices)
        raise QualificationError("use --execute or --worker")
    except (QualificationError, subprocess.TimeoutExpired) as error:
        print(f"qualification failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
