"""Measured PyTorch executor for accepted AWS-A2 experiments.

The executor is intentionally provider-neutral and assumes it is already running
inside the selected container on a qualified host. Cloud lifecycle, storage
attachment, and artifact stage-out remain under ``infra/``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from array import array
from datetime import timedelta
from pathlib import Path
from typing import Any

from common.experiment_runner import (
    RunnerError,
    configured_variants,
    load_yaml,
    select_variants,
    visible_devices,
)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def variant_by_id(config: dict[str, Any], variant_id: str) -> dict[str, Any]:
    matches = select_variants(configured_variants(config), variant_ids=[variant_id])
    return matches[0]


def repo_root_from_config(config_path: Path) -> Path:
    # experiments/exp_NN_slug/experiment.yaml -> repository root.
    return config_path.resolve().parents[2]


def input_lock_path(config_path: Path, config: dict[str, Any]) -> Path:
    try:
        value = config["workload"]["overrides"]["input_lock"]
    except KeyError as error:
        raise RunnerError("workload.overrides.input_lock is required") from error
    return (config_path.parent / value).resolve()


def load_input_paths(config_path: Path, config: dict[str, Any]) -> dict[str, Path]:
    from scripts.prepare_inputs import input_paths, load_config

    lock = load_config(input_lock_path(config_path, config))
    paths = input_paths(lock)
    manifest = paths["processed"] / "manifest.json"
    if not manifest.is_file():
        raise RunnerError(
            f"processed input manifest is missing: {manifest}; run or stage make prepare-inputs first"
        )
    if not paths["model"].is_dir():
        raise RunnerError(f"model directory is missing: {paths['model']}")
    return paths


def torch_modules() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RunnerError("PyTorch is required inside the runtime image") from error
    return torch


def require_cuda(torch: Any, expected_devices: list[int]) -> None:
    if not torch.cuda.is_available():
        raise RunnerError("CUDA is not available; measured AWS-A2 runs require NVIDIA GPUs")
    available = torch.cuda.device_count()
    if available < len(expected_devices):
        raise RunnerError(
            f"expected at least {len(expected_devices)} visible CUDA device(s), got {available}"
        )


def distributed_context(torch: Any, variant: dict[str, Any]) -> tuple[int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        import torch.distributed as dist

        timeout_seconds = int(variant.get("timeout_seconds", 1800))
        dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_seconds))
    return rank, local_rank, is_distributed


def cleanup_distributed() -> None:
    try:
        import torch.distributed as dist
    except (ImportError, RuntimeError):
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def precision_dtype(torch: Any, precision: str | None) -> Any:
    if precision in {None, "fp32", "tf32"}:
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise RunnerError(f"unsupported measured precision for this executor: {precision}")


def configure_precision(torch: Any, precision: str | None) -> None:
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    torch.backends.cudnn.allow_tf32 = precision == "tf32"


def read_tokens(tokens_path: Path, start_token: int, count: int) -> list[int]:
    values = array("I")
    with tokens_path.open("rb") as handle:
        handle.seek(start_token * 4)
        values.frombytes(handle.read(count * 4))
    if len(values) != count:
        raise RunnerError(f"not enough tokens in {tokens_path}")
    if sys.byteorder != "little":
        values.byteswap()
    return values.tolist()


def token_batch(
    *,
    torch: Any,
    processed_dir: Path,
    step: int,
    batch_size: int,
    sequence_length: int,
    device: Any,
) -> tuple[Any, Any]:
    tokens_path = processed_dir / "train.tokens.bin"
    token_count = tokens_path.stat().st_size // 4
    sample_width = sequence_length + 1
    required = batch_size * sample_width
    if token_count <= required:
        raise RunnerError(
            f"processed train token stream is too small for batch={batch_size}, seq={sequence_length}"
        )
    start = (step * required) % (token_count - required)
    values = read_tokens(tokens_path, start, required)
    tensor = torch.tensor(values, dtype=torch.long, device=device).view(batch_size, sample_width)
    return tensor[:, :-1].contiguous(), tensor[:, 1:].contiguous()


def load_model(torch: Any, model_dir: Path, precision: str | None, device: Any) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ModuleNotFoundError as error:
        raise RunnerError("transformers is required inside the runtime image") from error

    dtype = precision_dtype(torch, precision)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    model.train()
    return model.to(device)


def train_steps(
    *,
    config_path: Path,
    config: dict[str, Any],
    variant: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    torch = torch_modules()
    devices = visible_devices(variant)
    require_cuda(torch, devices)
    rank, local_rank, is_distributed = distributed_context(torch, variant)
    device = torch.device(f"cuda:{local_rank}")
    precision = variant.get("precision") or config["workload"]["overrides"].get("precision")
    if precision == "bf16_candidate":
        precision = "bf16"
    configure_precision(torch, precision)

    try:
        paths = load_input_paths(config_path, config)
        model = load_model(torch, paths["model"], precision, device)
        strategy = variant.get("strategy")
        if is_distributed and strategy == "fsdp":
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy

            sharding = variant.get("sharding")
            strategy_value = (
                ShardingStrategy.FULL_SHARD
                if sharding == "parameters_gradients_and_optimizer_state"
                else ShardingStrategy.SHARD_GRAD_OP
            )
            model = FSDP(model, sharding_strategy=strategy_value, device_id=device)
            fsdp_api = "torch.distributed.fsdp.FullyShardedDataParallel"
        elif is_distributed:
            from torch.nn.parallel import DistributedDataParallel

            bucket_cap_mb = variant.get("bucket_cap_mb")
            kwargs = {"device_ids": [local_rank], "output_device": local_rank}
            if bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = int(bucket_cap_mb)
            model = DistributedDataParallel(model, **kwargs)
            fsdp_api = None
        else:
            fsdp_api = None

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        sequence_length = int(config["workload"]["overrides"].get("sequence_length", 1024))
        batch_size = int(variant.get("per_rank_microbatch_size", 1))
        accumulation = int(variant.get("gradient_accumulation_steps", 1))
        warmup_steps = int(config["workload"]["overrides"].get("warmup_optimizer_steps", 1))
        measured_steps = int(config["workload"]["overrides"].get("measured_optimizer_steps", 3))
        scaler_enabled = precision == "fp16" and variant.get("gradient_scaling") == "enabled"
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        timings: list[float] = []
        losses: list[float] = []
        peak_allocated = 0
        total_steps = warmup_steps + measured_steps
        no_sync_requested = variant.get("synchronization") == "no_sync_until_optimizer_step"

        for step in range(total_steps):
            optimizer.zero_grad(set_to_none=True)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            total_loss = None
            for microstep in range(accumulation):
                input_ids, labels = token_batch(
                    torch=torch,
                    processed_dir=paths["processed"],
                    step=step * accumulation + microstep + rank,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    device=device,
                )
                should_sync = microstep == accumulation - 1
                context = (
                    model.no_sync()
                    if is_distributed and no_sync_requested and not should_sync
                    else contextlib.nullcontext()
                )
                with context:
                    with torch.autocast(
                        "cuda",
                        dtype=precision_dtype(torch, precision),
                        enabled=precision in {"bf16", "fp16"},
                    ):
                        loss = model(input_ids=input_ids, labels=labels).loss / accumulation
                    if scaler_enabled:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    total_loss = loss.detach() if total_loss is None else total_loss + loss.detach()
            if scaler_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            end_event.record()
            torch.cuda.synchronize(device)
            peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
            if step >= warmup_steps:
                timings.append(start_event.elapsed_time(end_event))
                if total_loss is not None:
                    losses.append(float(total_loss.detach().float().cpu()))

        result = {
            "variant_id": variant["id"],
            "rank": rank,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "precision": precision,
            "strategy": strategy or ("ddp" if is_distributed else "single"),
            "fsdp_api": fsdp_api,
            "measured_optimizer_steps": measured_steps,
            "mean_step_ms": sum(timings) / len(timings) if timings else None,
            "final_loss": losses[-1] if losses else None,
            "finite_loss": all(torch.isfinite(torch.tensor(losses)).tolist()) if losses else False,
            "peak_allocated_bytes": peak_allocated,
        }
        write_json(output_dir / f"rank_{rank}.json", result)
        return result
    finally:
        cleanup_distributed()


def gemm_microbenchmark(variant: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    torch = torch_modules()
    require_cuda(torch, visible_devices(variant))
    device = torch.device("cuda:0")
    precision = variant.get("precision")
    configure_precision(torch, precision)
    dtype = precision_dtype(torch, precision)
    aligned = variant.get("shape") == "tensor_core_aligned"
    size = int(os.environ.get("EXP_GEMM_SIZE", "4096" if aligned else "4097"))
    warmup = int(os.environ.get("EXP_GEMM_WARMUP", "10"))
    iterations = int(os.environ.get("EXP_GEMM_ITERS", "50"))
    a = torch.randn((size, size), device=device, dtype=dtype)
    b = torch.randn((size, size), device=device, dtype=dtype)
    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize(device)
    elapsed_ms = start.elapsed_time(end)
    flops = 2 * (size**3) * iterations
    result = {
        "variant_id": variant["id"],
        "precision": precision,
        "shape": variant.get("shape"),
        "matrix_size": size,
        "iterations": iterations,
        "elapsed_ms": elapsed_ms,
        "tflops": flops / (elapsed_ms / 1000.0) / 1e12,
        "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    write_json(output_dir / "rank_0.json", result)
    return result


def fault_case(variant: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    torch = torch_modules()
    require_cuda(torch, visible_devices(variant))
    rank, local_rank, is_distributed = distributed_context(torch, variant)
    device = torch.device(f"cuda:{local_rank}")
    case = variant.get("case")
    start = time.monotonic()
    status = "pass"
    detail = ""
    try:
        if case == "healthy_training_smoke":
            x = torch.ones((1024, 1024), device=device)
            y = (x @ x).sum()
            y.backward() if y.requires_grad else None
            detail = "healthy CUDA smoke completed"
        elif case == "bounded_oom_or_fragmentation":
            if os.environ.get("EXP09_REAL_OOM") == "1":
                torch.empty((1024, 1024, 1024, 16), dtype=torch.float32, device=device)
            raise RuntimeError("controlled CUDA out of memory diagnostic")
        elif case == "fp16_nonfinite_gradient":
            x = torch.full((1024,), 65504.0, dtype=torch.float16, device=device)
            y = x * x
            if torch.isfinite(y).all():
                raise RunnerError("expected non-finite fp16 result was finite")
            raise FloatingPointError("controlled fp16 non-finite diagnostic")
        elif case == "input_pipeline_starvation":
            sleep_seconds = float(os.environ.get("EXP09_INPUT_SLEEP_SECONDS", "5"))
            time.sleep(sleep_seconds)
            detail = f"controlled input wait of {sleep_seconds} seconds"
        elif case == "mismatched_collective":
            import torch.distributed as dist

            tensor = torch.ones(1, device=device)
            if rank == 0:
                dist.all_reduce(tensor)
            else:
                dist.broadcast(tensor, src=1)
        elif case == "artificial_rank_straggler":
            import torch.distributed as dist

            if rank == 1:
                time.sleep(float(os.environ.get("EXP09_STRAGGLER_SECONDS", "10")))
            tensor = torch.ones(1, device=device)
            dist.all_reduce(tensor)
            detail = "straggler collective completed"
        elif case == "nccl_timeout_signature":
            if rank == 0:
                time.sleep(float(os.environ.get("EXP09_NCCL_TIMEOUT_SLEEP_SECONDS", "180")))
            else:
                import torch.distributed as dist

                dist.all_reduce(torch.ones(1, device=device))
        else:
            raise RunnerError(f"unsupported EXP-09 fault case: {case}")
    except (RuntimeError, FloatingPointError) as error:
        expected = variant.get("expected_outcome") == "controlled_failure"
        status = "expected_failure" if expected else "unexpected_failure"
        detail = str(error)
    finally:
        cleanup_distributed()

    result = {
        "variant_id": variant["id"],
        "case": case,
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "status": status,
        "detail": detail,
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json(output_dir / f"rank_{rank}.json", result)
    return result


def run_worker(config_path: Path, variant_id: str, run_dir: Path) -> int:
    config = load_yaml(config_path)
    variant = variant_by_id(config, variant_id)
    output_dir = run_dir / "raw" / variant_id
    workload = variant.get("workload")
    case = variant.get("case")
    strategy = variant.get("strategy")
    if workload == "gemm_microbenchmark":
        result = gemm_microbenchmark(variant, output_dir)
    elif case is not None:
        result = fault_case(variant, output_dir)
    elif workload in {"transformer_training", "transformer_training_ddp"} or strategy in {"ddp", "fsdp"} or "world_size" in variant:
        result = train_steps(config_path=config_path, config=config, variant=variant, output_dir=output_dir)
    else:
        raise RunnerError(f"no measured executor route for variant: {variant_id}")
    append_jsonl(run_dir / "metrics" / "variant_results.jsonl", result)
    return 0


def worker_command(config_path: Path, variant_id: str, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "common.pytorch_executor",
        "--worker",
        "--config",
        str(config_path),
        "--variant",
        variant_id,
        "--run-dir",
        str(run_dir),
    ]


def run_variant_subprocess(plan: dict[str, Any], variant: dict[str, Any]) -> None:
    run_dir = Path(plan["run_dir"])
    config_path = Path(plan["config_path"])
    variant_id = variant["id"]
    devices = ",".join(str(device) for device in variant.get("visible_devices", []))
    env = os.environ.copy()
    if devices:
        env["CUDA_VISIBLE_DEVICES"] = devices
    world_size = int(variant.get("world_size") or len(visible_devices(variant)) or 1)
    timeout_seconds = int(variant.get("timeout_seconds") or 1800)
    if world_size > 1:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes",
            "1",
            "--nproc_per_node",
            str(world_size),
            "-m",
            "common.pytorch_executor",
            "--worker",
            "--config",
            str(config_path),
            "--variant",
            variant_id,
            "--run-dir",
            str(run_dir),
        ]
    else:
        command = worker_command(config_path, variant_id, run_dir)
    log_path = run_dir / "raw" / variant_id / "subprocess.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root_from_config(config_path),
                env=env,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise RunnerError(
                f"variant {variant_id} exceeded timeout {timeout_seconds}s; see {log_path}"
            ) from error
    if completed.returncode != 0:
        raise RunnerError(
            f"variant {variant_id} failed with exit code {completed.returncode}; see {log_path}"
        )


def execute_pytorch_plan(plan: dict[str, Any], config: dict[str, Any]) -> int:
    run_dir = Path(plan["run_dir"])
    write_json(run_dir / "manifest.json", {"plan": plan, "status": "running"})
    variants = {variant["id"]: variant for variant in configured_variants(config)}
    for planned in plan["variants"]:
        variant = variants[planned["id"]]
        run_variant_subprocess(plan, variant)
    write_json(run_dir / "manifest.json", {"plan": plan, "status": "completed"})
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Worker process for measured PyTorch variants.")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run_worker(args.config.resolve(), args.variant, args.run_dir.resolve())
    except RunnerError as error:
        print(f"pytorch executor failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
