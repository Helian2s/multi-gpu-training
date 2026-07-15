"""Measured executor for NeMo/Megatron schedule experiments.

This module is provider-neutral. It assumes it is already running inside the
selected NeMo/Megatron container on a qualified GPU host.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
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


def torch_modules() -> Any:
    try:
        import torch
    except ModuleNotFoundError as error:
        raise RunnerError("PyTorch is required inside the NeMo/Megatron runtime image") from error
    return torch


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def megatron_environment() -> dict[str, Any]:
    try:
        core = importlib.import_module("megatron.core")
        schedules = importlib.import_module("megatron.core.pipeline_parallel.schedules")
    except Exception as error:
        raise RunnerError("Megatron Core pipeline modules are not importable") from error
    return {
        "megatron_core_version": package_version("megatron-core"),
        "megatron_bridge_version": package_version("megatron-bridge"),
        "megatron_core_path": getattr(core, "__file__", None),
        "pipeline_schedules_path": getattr(schedules, "__file__", None),
        "has_forward_backward_no_pipelining": hasattr(
            schedules, "forward_backward_no_pipelining"
        ),
        "has_forward_backward_pipelining_without_interleaving": hasattr(
            schedules, "forward_backward_pipelining_without_interleaving"
        ),
    }


def require_cuda(torch: Any, expected_devices: list[int]) -> None:
    if not torch.cuda.is_available():
        raise RunnerError("CUDA is not available; measured Megatron runs require NVIDIA GPUs")
    available = torch.cuda.device_count()
    if available < len(expected_devices):
        raise RunnerError(
            f"expected at least {len(expected_devices)} visible CUDA device(s), got {available}"
        )


def precision_dtype(torch: Any, precision: str | None) -> Any:
    if precision in {None, "fp32", "tf32"}:
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise RunnerError(f"unsupported precision for Megatron executor: {precision}")


def distributed_context(torch: Any, variant: dict[str, Any]) -> tuple[int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        torch.cuda.set_device(local_rank)
        import torch.distributed as dist

        timeout_seconds = int(variant.get("timeout_seconds", 1800))
        dist.init_process_group("nccl", timeout=timedelta(seconds=timeout_seconds))
    return rank, local_rank, world_size, is_distributed


def cleanup_distributed() -> None:
    # The measured workers are short-lived torchrun subprocesses. Explicit
    # NCCL destruction can hang with multiple custom TP/DP groups on some
    # runtimes, while process exit reliably tears the groups down.
    if os.environ.get("MGT_DESTROY_PROCESS_GROUP") != "1":
        return
    try:
        import torch.distributed as dist
    except (ImportError, RuntimeError):
        return
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def repo_root_from_config(config_path: Path) -> Path:
    return config_path.resolve().parents[2]


def variant_by_id(config: dict[str, Any], variant_id: str) -> dict[str, Any]:
    matches = select_variants(configured_variants(config), variant_ids=[variant_id])
    return matches[0]


def make_stage(torch: Any, hidden_size: int, layer_count: int, device: Any) -> Any:
    modules: list[Any] = []
    for _ in range(layer_count):
        modules.append(torch.nn.Linear(hidden_size, hidden_size, bias=False))
        modules.append(torch.nn.GELU())
    model = torch.nn.Sequential(*modules)
    model.train()
    return model.to(device)


def stage_layers_for_rank(variant: dict[str, Any], rank: int, world_size: int) -> int:
    layers = variant.get("stage_layers")
    if isinstance(layers, list) and layers:
        if rank < len(layers):
            return int(layers[rank])
        return int(layers[-1])
    total_layers = int(variant.get("total_layers", world_size * 2))
    return max(1, total_layers // max(1, world_size))


def input_shape(variant: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(variant.get("microbatch_size", 1)),
        int(variant.get("sequence_length", 512)),
        int(variant.get("hidden_size", 1024)),
    )


def new_input(torch: Any, shape: tuple[int, int, int], device: Any, seed: int) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.randn(shape, device=device, generator=generator)


def timed_cuda_step(torch: Any, device: Any, step: Any) -> float:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    step()
    end_event.record()
    torch.cuda.synchronize(device)
    return float(start_event.elapsed_time(end_event))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def make_parameters(torch: Any, shapes: list[tuple[int, ...]], device: Any) -> list[Any]:
    parameters = []
    for index, shape in enumerate(shapes):
        generator = torch.Generator(device=device)
        generator.manual_seed(50_000 + index)
        tensor = torch.randn(shape, device=device, generator=generator) / math.sqrt(max(1, shape[0]))
        parameters.append(torch.nn.Parameter(tensor))
    return parameters


def all_reduce_if_needed(torch: Any, tensor: Any, group: Any | None) -> None:
    if group is None:
        return
    import torch.distributed as dist

    dist.all_reduce(tensor, group=group)


def all_gather_first_dim(torch: Any, tensor: Any, group: Any | None, group_size: int) -> Any:
    if group is None or group_size == 1:
        return tensor
    import torch.distributed as dist

    gathered = [torch.empty_like(tensor) for _ in range(group_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=0)


def create_rank_groups(
    *,
    world_size: int,
    rank: int,
    group_specs: list[list[int]],
) -> dict[tuple[int, ...], Any | None]:
    if world_size == 1:
        return {tuple(spec): None for spec in group_specs}
    import torch.distributed as dist

    groups: dict[tuple[int, ...], Any | None] = {}
    for spec in group_specs:
        key = tuple(spec)
        groups[key] = dist.new_group(ranks=spec) if len(spec) > 1 else None
    return groups


def group_for_rank(
    groups: dict[tuple[int, ...], Any | None],
    rank: int,
) -> tuple[list[int], Any | None]:
    for spec, group in groups.items():
        if rank in spec:
            return list(spec), group
    raise RunnerError(f"rank {rank} is not present in any process group")


def tensor_sequence_parallel_benchmark(
    *,
    config_path: Path,
    config: dict[str, Any],
    variant: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    torch = torch_modules()
    devices = visible_devices(variant)
    require_cuda(torch, devices)
    megatron_info = megatron_environment()
    rank, local_rank, world_size, is_distributed = distributed_context(torch, variant)
    expected_world = int(variant.get("world_size", len(devices) or 1))
    if world_size != expected_world:
        raise RunnerError(f"expected world_size={expected_world}, got {world_size}")
    tensor_parallel_size = int(variant.get("tensor_parallel_size", world_size))
    if tensor_parallel_size != world_size:
        raise RunnerError("EXP-11 synthetic TP/SP variants expect one TP group spanning all ranks")

    device = torch.device(f"cuda:{local_rank}")
    dtype = precision_dtype(torch, variant.get("precision") or config["workload"]["overrides"].get("precision"))
    hidden_size = int(variant.get("hidden_size", 1024))
    expansion = int(variant.get("mlp_expansion", 4))
    intermediate = hidden_size * expansion
    if intermediate % tensor_parallel_size != 0:
        raise RunnerError("intermediate size must be divisible by tensor_parallel_size")
    shard_intermediate = intermediate // tensor_parallel_size
    sequence_length = int(variant.get("sequence_length", 512))
    microbatch_size = int(variant.get("microbatch_size", 1))
    sequence_parallel = bool(variant.get("sequence_parallel", False))
    local_sequence = sequence_length // tensor_parallel_size if sequence_parallel else sequence_length
    if sequence_parallel and sequence_length % tensor_parallel_size != 0:
        raise RunnerError("sequence_length must be divisible by tensor_parallel_size")

    parameters = make_parameters(
        torch,
        [(hidden_size, shard_intermediate), (shard_intermediate, hidden_size)],
        device,
    )
    optimizer = torch.optim.AdamW(parameters, lr=float(variant.get("learning_rate", 1e-5)))
    group = None
    if is_distributed:
        group = create_rank_groups(
            world_size=world_size,
            rank=rank,
            group_specs=[list(range(world_size))],
        )[tuple(range(world_size))]
        import torch.distributed as dist

        dist.barrier()

    warmup_steps = int(variant.get("warmup_steps", 1))
    measured_steps = int(variant.get("measured_steps", 5))
    timings: list[float] = []
    collective_bytes = 0
    torch.cuda.reset_peak_memory_stats(device)

    for step_index in range(warmup_steps + measured_steps):
        if is_distributed:
            import torch.distributed as dist

            dist.barrier()

        def step() -> None:
            nonlocal collective_bytes
            optimizer.zero_grad(set_to_none=True)
            x = new_input(
                torch,
                (microbatch_size, local_sequence, hidden_size),
                device,
                seed=80_000 + step_index + rank,
            ).reshape(-1, hidden_size)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                hidden = torch.nn.functional.gelu(x @ parameters[0])
                partial = hidden @ parameters[1]
            if tensor_parallel_size > 1:
                all_reduce_if_needed(torch, partial, group)
                collective_bytes += partial.numel() * partial.element_size() * tensor_parallel_size
            loss = partial.float().pow(2).mean()
            loss.backward()
            optimizer.step()

        elapsed = timed_cuda_step(torch, device, step)
        if step_index >= warmup_steps:
            timings.append(elapsed)

    if is_distributed:
        import torch.distributed as dist

        dist.barrier()
    mean_step_ms = mean(timings)
    rank_means = gather_float(torch, float(mean_step_ms or 0.0), device, world_size)
    max_mean_step_ms = max(rank_means) if rank_means else None
    tokens_per_step = microbatch_size * sequence_length
    result = {
        "variant_id": variant["id"],
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "workload": variant.get("workload"),
        "tensor_parallel_size": tensor_parallel_size,
        "sequence_parallel": sequence_parallel,
        "sequence_length": sequence_length,
        "local_sequence_length": local_sequence,
        "microbatch_size": microbatch_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate,
        "shard_intermediate_size": shard_intermediate,
        "precision": variant.get("precision") or config["workload"]["overrides"].get("precision"),
        "measured_steps": measured_steps,
        "mean_step_ms": mean_step_ms,
        "rank_mean_step_ms": rank_means,
        "max_rank_mean_step_ms": max_mean_step_ms,
        "tokens_per_second": (
            tokens_per_step / (max_mean_step_ms / 1000.0) if max_mean_step_ms else None
        ),
        "estimated_collective_bytes": collective_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "megatron": megatron_info,
    }
    write_json(output_dir / f"rank_{rank}.json", result)
    return result


def context_parallel_benchmark(
    *,
    config_path: Path,
    config: dict[str, Any],
    variant: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    torch = torch_modules()
    devices = visible_devices(variant)
    require_cuda(torch, devices)
    megatron_info = megatron_environment()
    rank, local_rank, world_size, is_distributed = distributed_context(torch, variant)
    expected_world = int(variant.get("world_size", len(devices) or 1))
    if world_size != expected_world:
        raise RunnerError(f"expected world_size={expected_world}, got {world_size}")
    context_parallel_size = int(variant.get("context_parallel_size", world_size))
    if context_parallel_size != world_size:
        raise RunnerError("EXP-13 synthetic CP variants expect one CP group spanning all ranks")

    device = torch.device(f"cuda:{local_rank}")
    dtype = precision_dtype(torch, variant.get("precision") or config["workload"]["overrides"].get("precision"))
    hidden_size = int(variant.get("hidden_size", 512))
    sequence_length = int(variant.get("sequence_length", 1024))
    microbatch_size = int(variant.get("microbatch_size", 1))
    if sequence_length % context_parallel_size != 0:
        raise RunnerError("sequence_length must be divisible by context_parallel_size")
    local_sequence = sequence_length // context_parallel_size
    checkpoint_attention = bool(variant.get("activation_checkpointing", False))
    parameters = make_parameters(
        torch,
        [
            (hidden_size, hidden_size),
            (hidden_size, hidden_size),
            (hidden_size, hidden_size),
            (hidden_size, hidden_size),
        ],
        device,
    )
    optimizer = torch.optim.AdamW(parameters, lr=float(variant.get("learning_rate", 1e-5)))
    group = None
    if is_distributed:
        group = create_rank_groups(
            world_size=world_size,
            rank=rank,
            group_specs=[list(range(world_size))],
        )[tuple(range(world_size))]
        import torch.distributed as dist

        dist.barrier()

    def attention_forward(x: Any) -> Any:
        q = x @ parameters[0]
        k = x @ parameters[1]
        v = x @ parameters[2]
        flat_k = k.reshape(-1, hidden_size)
        flat_v = v.reshape(-1, hidden_size)
        gathered_k = all_gather_first_dim(torch, flat_k, group, context_parallel_size).reshape(
            microbatch_size, sequence_length, hidden_size
        )
        gathered_v = all_gather_first_dim(torch, flat_v, group, context_parallel_size).reshape(
            microbatch_size, sequence_length, hidden_size
        )
        scores = torch.matmul(q, gathered_k.transpose(-1, -2)) / math.sqrt(hidden_size)
        weights = torch.softmax(scores.float(), dim=-1).to(dtype if dtype != torch.float32 else q.dtype)
        context = torch.matmul(weights, gathered_v)
        return context @ parameters[3]

    warmup_steps = int(variant.get("warmup_steps", 1))
    measured_steps = int(variant.get("measured_steps", 4))
    timings: list[float] = []
    collective_bytes = 0
    torch.cuda.reset_peak_memory_stats(device)

    for step_index in range(warmup_steps + measured_steps):
        if is_distributed:
            import torch.distributed as dist

            dist.barrier()

        def step() -> None:
            nonlocal collective_bytes
            optimizer.zero_grad(set_to_none=True)
            x = new_input(
                torch,
                (microbatch_size, local_sequence, hidden_size),
                device,
                seed=90_000 + step_index + rank,
            )
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                if checkpoint_attention:
                    from torch.utils.checkpoint import checkpoint

                    y = checkpoint(attention_forward, x, use_reentrant=False)
                else:
                    y = attention_forward(x)
            if context_parallel_size > 1:
                bytes_per_kv = microbatch_size * local_sequence * hidden_size * x.element_size()
                collective_bytes += 2 * bytes_per_kv * context_parallel_size
            loss = y.float().pow(2).mean()
            loss.backward()
            optimizer.step()

        elapsed = timed_cuda_step(torch, device, step)
        if step_index >= warmup_steps:
            timings.append(elapsed)

    if is_distributed:
        import torch.distributed as dist

        dist.barrier()
    mean_step_ms = mean(timings)
    rank_means = gather_float(torch, float(mean_step_ms or 0.0), device, world_size)
    max_mean_step_ms = max(rank_means) if rank_means else None
    tokens_per_step = microbatch_size * sequence_length
    result = {
        "variant_id": variant["id"],
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "workload": variant.get("workload"),
        "context_parallel_size": context_parallel_size,
        "sequence_length": sequence_length,
        "local_sequence_length": local_sequence,
        "microbatch_size": microbatch_size,
        "hidden_size": hidden_size,
        "activation_checkpointing": checkpoint_attention,
        "precision": variant.get("precision") or config["workload"]["overrides"].get("precision"),
        "measured_steps": measured_steps,
        "mean_step_ms": mean_step_ms,
        "rank_mean_step_ms": rank_means,
        "max_rank_mean_step_ms": max_mean_step_ms,
        "tokens_per_second": (
            tokens_per_step / (max_mean_step_ms / 1000.0) if max_mean_step_ms else None
        ),
        "estimated_collective_bytes": collective_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "megatron": megatron_info,
    }
    write_json(output_dir / f"rank_{rank}.json", result)
    return result


def hybrid_rank_groups(layout: str, world_size: int) -> tuple[list[list[int]], list[list[int]]]:
    if layout == "dp4":
        if world_size != 4:
            raise RunnerError("DP=4 layout requires world_size=4")
        return [[0], [1], [2], [3]], [[0, 1, 2, 3]]
    if layout == "tp4":
        if world_size != 4:
            raise RunnerError("TP=4 layout requires world_size=4")
        return [[0, 1, 2, 3]], [[0], [1], [2], [3]]
    if layout == "tp2_dp2":
        if world_size != 4:
            raise RunnerError("TP=2 x DP=2 layout requires world_size=4")
        return [[0, 1], [2, 3]], [[0, 2], [1, 3]]
    raise RunnerError(f"unsupported EXP-14 hybrid layout: {layout}")


def hybrid_tp_dp_benchmark(
    *,
    config_path: Path,
    config: dict[str, Any],
    variant: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    torch = torch_modules()
    devices = visible_devices(variant)
    require_cuda(torch, devices)
    megatron_info = megatron_environment()
    rank, local_rank, world_size, is_distributed = distributed_context(torch, variant)
    expected_world = int(variant.get("world_size", len(devices) or 1))
    if world_size != expected_world or world_size != 4:
        raise RunnerError(f"EXP-14 requires world_size=4, got {world_size}")
    if not is_distributed:
        raise RunnerError("EXP-14 requires distributed execution")

    layout = str(variant.get("layout"))
    tp_specs, dp_specs = hybrid_rank_groups(layout, world_size)
    tp_groups = create_rank_groups(world_size=world_size, rank=rank, group_specs=tp_specs)
    dp_groups = create_rank_groups(world_size=world_size, rank=rank, group_specs=dp_specs)
    tp_ranks, tp_group = group_for_rank(tp_groups, rank)
    dp_ranks, dp_group = group_for_rank(dp_groups, rank)
    tensor_parallel_size = len(tp_ranks)
    data_parallel_size = len(dp_ranks)

    device = torch.device(f"cuda:{local_rank}")
    dtype = precision_dtype(torch, variant.get("precision") or config["workload"]["overrides"].get("precision"))
    hidden_size = int(variant.get("hidden_size", 1024))
    expansion = int(variant.get("mlp_expansion", 4))
    intermediate = hidden_size * expansion
    if intermediate % tensor_parallel_size != 0:
        raise RunnerError("intermediate size must be divisible by tensor_parallel_size")
    shard_intermediate = intermediate // tensor_parallel_size
    microbatch_size = int(variant.get("microbatch_size", 1))
    sequence_length = int(variant.get("sequence_length", 512))
    parameters = make_parameters(
        torch,
        [(hidden_size, shard_intermediate), (shard_intermediate, hidden_size)],
        device,
    )
    optimizer = torch.optim.AdamW(parameters, lr=float(variant.get("learning_rate", 1e-5)))
    import torch.distributed as dist

    dist.barrier()
    warmup_steps = int(variant.get("warmup_steps", 1))
    measured_steps = int(variant.get("measured_steps", 5))
    timings: list[float] = []
    collective_bytes = 0
    torch.cuda.reset_peak_memory_stats(device)

    for step_index in range(warmup_steps + measured_steps):
        dist.barrier()

        def step() -> None:
            nonlocal collective_bytes
            optimizer.zero_grad(set_to_none=True)
            x = new_input(
                torch,
                (microbatch_size, sequence_length, hidden_size),
                device,
                seed=100_000 + step_index + rank,
            ).reshape(-1, hidden_size)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                hidden = torch.nn.functional.gelu(x @ parameters[0])
                partial = hidden @ parameters[1]
            if tensor_parallel_size > 1:
                dist.all_reduce(partial, group=tp_group)
                collective_bytes += partial.numel() * partial.element_size() * tensor_parallel_size
            loss = partial.float().pow(2).mean()
            loss.backward()
            if data_parallel_size > 1:
                for parameter in parameters:
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad, group=dp_group)
                        parameter.grad.div_(data_parallel_size)
                        collective_bytes += (
                            parameter.grad.numel()
                            * parameter.grad.element_size()
                            * data_parallel_size
                        )
            optimizer.step()

        elapsed = timed_cuda_step(torch, device, step)
        if step_index >= warmup_steps:
            timings.append(elapsed)

    dist.barrier()
    mean_step_ms = mean(timings)
    rank_means = gather_float(torch, float(mean_step_ms or 0.0), device, world_size)
    max_mean_step_ms = max(rank_means) if rank_means else None
    tokens_per_step = microbatch_size * sequence_length * data_parallel_size
    result = {
        "variant_id": variant["id"],
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "workload": variant.get("workload"),
        "layout": layout,
        "tensor_parallel_size": tensor_parallel_size,
        "data_parallel_size": data_parallel_size,
        "tp_ranks": tp_ranks,
        "dp_ranks": dp_ranks,
        "all_tp_groups": tp_specs,
        "all_dp_groups": dp_specs,
        "sequence_length": sequence_length,
        "microbatch_size": microbatch_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate,
        "shard_intermediate_size": shard_intermediate,
        "precision": variant.get("precision") or config["workload"]["overrides"].get("precision"),
        "measured_steps": measured_steps,
        "mean_step_ms": mean_step_ms,
        "rank_mean_step_ms": rank_means,
        "max_rank_mean_step_ms": max_mean_step_ms,
        "tokens_per_second": (
            tokens_per_step / (max_mean_step_ms / 1000.0) if max_mean_step_ms else None
        ),
        "estimated_collective_bytes": collective_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "megatron": megatron_info,
    }
    write_json(output_dir / f"rank_{rank}.json", result)
    return result


def run_no_pipeline_step(
    *,
    torch: Any,
    variant: dict[str, Any],
    model: Any,
    optimizer: Any,
    device: Any,
    dtype: Any,
    step_index: int,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    microbatches = int(variant.get("num_microbatches", 1))
    shape = input_shape(variant)
    for microbatch in range(microbatches):
        x = new_input(torch, shape, device, seed=10_000 * step_index + microbatch)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            y = model(x)
            loss = y.float().pow(2).mean() / microbatches
        loss.backward()
    optimizer.step()


def run_flush_pipeline_step(
    *,
    torch: Any,
    variant: dict[str, Any],
    model: Any,
    optimizer: Any,
    device: Any,
    dtype: Any,
    rank: int,
    step_index: int,
) -> int:
    import torch.distributed as dist

    optimizer.zero_grad(set_to_none=True)
    microbatches = int(variant.get("num_microbatches", 1))
    shape = input_shape(variant)
    p2p_bytes = 0
    if rank == 0:
        outputs = []
        for microbatch in range(microbatches):
            x = new_input(torch, shape, device, seed=10_000 * step_index + microbatch)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                y = model(x)
            outputs.append(y)
            dist.send(y.detach(), dst=1)
            p2p_bytes += y.numel() * y.element_size()
        for y in reversed(outputs):
            grad = torch.empty_like(y)
            dist.recv(grad, src=1)
            torch.autograd.backward(y, grad)
            p2p_bytes += grad.numel() * grad.element_size()
    else:
        activations = []
        losses = []
        for _ in range(microbatches):
            activation = torch.empty(shape, device=device, dtype=dtype)
            dist.recv(activation, src=0)
            activation.requires_grad_(True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                y = model(activation)
                loss = y.float().pow(2).mean() / microbatches
            activations.append(activation)
            losses.append(loss)
            p2p_bytes += activation.numel() * activation.element_size()
        for activation, loss in reversed(list(zip(activations, losses, strict=True))):
            loss.backward()
            if activation.grad is None:
                raise RunnerError("pipeline activation gradient was not produced")
            dist.send(activation.grad, dst=0)
            p2p_bytes += activation.grad.numel() * activation.grad.element_size()
    optimizer.step()
    return p2p_bytes


def run_one_f_one_b_step(
    *,
    torch: Any,
    variant: dict[str, Any],
    model: Any,
    optimizer: Any,
    device: Any,
    dtype: Any,
    rank: int,
    step_index: int,
) -> int:
    import torch.distributed as dist

    optimizer.zero_grad(set_to_none=True)
    microbatches = int(variant.get("num_microbatches", 1))
    shape = input_shape(variant)
    p2p_bytes = 0
    if rank == 0:
        pending = []
        for microbatch in range(microbatches):
            x = new_input(torch, shape, device, seed=10_000 * step_index + microbatch)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                y = model(x)
            send_tensor = y.detach().contiguous()
            send_request = dist.isend(send_tensor, dst=1)
            pending.append((y, send_tensor, send_request))
            p2p_bytes += send_tensor.numel() * send_tensor.element_size()
            if microbatch > 0:
                previous, _previous_send_tensor, previous_send_request = pending[microbatch - 1]
                previous_send_request.wait()
                grad = torch.empty_like(previous)
                recv_request = dist.irecv(grad, src=1)
                recv_request.wait()
                torch.autograd.backward(previous, grad)
                p2p_bytes += grad.numel() * grad.element_size()
        last, _last_send_tensor, last_send_request = pending[-1]
        last_send_request.wait()
        grad = torch.empty_like(last)
        recv_request = dist.irecv(grad, src=1)
        recv_request.wait()
        torch.autograd.backward(last, grad)
        p2p_bytes += grad.numel() * grad.element_size()
    else:
        for _ in range(microbatches):
            activation = torch.empty(shape, device=device, dtype=dtype)
            recv_request = dist.irecv(activation, src=0)
            recv_request.wait()
            activation.requires_grad_(True)
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                y = model(activation)
                loss = y.float().pow(2).mean() / microbatches
            loss.backward()
            if activation.grad is None:
                raise RunnerError("pipeline activation gradient was not produced")
            grad = activation.grad.contiguous()
            send_request = dist.isend(grad, dst=0)
            send_request.wait()
            p2p_bytes += (
                activation.numel() * activation.element_size()
                + grad.numel() * grad.element_size()
            )
    optimizer.step()
    return p2p_bytes


def gather_float(torch: Any, value: float, device: Any, world_size: int) -> list[float]:
    if world_size == 1:
        return [value]
    import torch.distributed as dist

    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return [float(item.cpu().item()) for item in gathered]


def pipeline_schedule_benchmark(
    *,
    config_path: Path,
    config: dict[str, Any],
    variant: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    torch = torch_modules()
    devices = visible_devices(variant)
    require_cuda(torch, devices)
    megatron_info = megatron_environment()
    rank, local_rank, world_size, is_distributed = distributed_context(torch, variant)
    expected_world = int(variant.get("world_size", len(devices) or 1))
    if world_size != expected_world:
        raise RunnerError(f"expected world_size={expected_world}, got {world_size}")
    device = torch.device(f"cuda:{local_rank}")
    dtype = precision_dtype(torch, variant.get("precision") or config["workload"]["overrides"].get("precision"))
    stage_layers = stage_layers_for_rank(variant, rank, world_size)
    model = make_stage(torch, int(variant.get("hidden_size", 1024)), stage_layers, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(variant.get("learning_rate", 1e-5)))
    schedule = str(variant.get("schedule", "no_pipeline"))
    warmup_steps = int(variant.get("warmup_steps", 1))
    measured_steps = int(variant.get("measured_steps", 4))
    total_steps = warmup_steps + measured_steps
    timings: list[float] = []
    p2p_bytes = 0
    torch.cuda.reset_peak_memory_stats(device)
    if is_distributed:
        import torch.distributed as dist

        dist.barrier()
    for step in range(total_steps):
        if is_distributed:
            import torch.distributed as dist

            dist.barrier()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        if world_size == 1:
            run_no_pipeline_step(
                torch=torch,
                variant=variant,
                model=model,
                optimizer=optimizer,
                device=device,
                dtype=dtype,
                step_index=step,
            )
        elif schedule == "flush":
            p2p_bytes += run_flush_pipeline_step(
                torch=torch,
                variant=variant,
                model=model,
                optimizer=optimizer,
                device=device,
                dtype=dtype,
                rank=rank,
                step_index=step,
            )
        elif schedule == "one_f_one_b":
            p2p_bytes += run_one_f_one_b_step(
                torch=torch,
                variant=variant,
                model=model,
                optimizer=optimizer,
                device=device,
                dtype=dtype,
                rank=rank,
                step_index=step,
            )
        else:
            raise RunnerError(f"unsupported EXP-12 schedule: {schedule}")
        end_event.record()
        torch.cuda.synchronize(device)
        if step >= warmup_steps:
            timings.append(start_event.elapsed_time(end_event))
    if is_distributed:
        import torch.distributed as dist

        dist.barrier()
    mean_step_ms = sum(timings) / len(timings) if timings else None
    rank_means = gather_float(torch, float(mean_step_ms or 0.0), device, world_size)
    max_mean_step_ms = max(rank_means) if rank_means else None
    tokens_per_step = (
        int(variant.get("num_microbatches", 1))
        * int(variant.get("microbatch_size", 1))
        * int(variant.get("sequence_length", 512))
    )
    pipeline_stages = int(variant.get("pipeline_parallel_size", world_size))
    bubble_fraction_estimate = (
        (pipeline_stages - 1) / (int(variant.get("num_microbatches", 1)) + pipeline_stages - 1)
        if pipeline_stages > 1
        else 0.0
    )
    result = {
        "variant_id": variant["id"],
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "workload": variant.get("workload"),
        "schedule": schedule,
        "pipeline_parallel_size": pipeline_stages,
        "num_microbatches": int(variant.get("num_microbatches", 1)),
        "microbatch_size": int(variant.get("microbatch_size", 1)),
        "sequence_length": int(variant.get("sequence_length", 512)),
        "hidden_size": int(variant.get("hidden_size", 1024)),
        "stage_layers": stage_layers,
        "precision": variant.get("precision") or config["workload"]["overrides"].get("precision"),
        "measured_steps": measured_steps,
        "mean_step_ms": mean_step_ms,
        "rank_mean_step_ms": rank_means,
        "max_rank_mean_step_ms": max_mean_step_ms,
        "tokens_per_second": (
            tokens_per_step / (max_mean_step_ms / 1000.0) if max_mean_step_ms else None
        ),
        "estimated_pipeline_bubble_fraction": bubble_fraction_estimate,
        "p2p_bytes": p2p_bytes,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "megatron": megatron_info,
    }
    write_json(output_dir / f"rank_{rank}.json", result)
    return result


def run_worker(config_path: Path, variant_id: str, run_dir: Path) -> int:
    config = load_yaml(config_path)
    variant = variant_by_id(config, variant_id)
    output_dir = run_dir / "raw" / variant_id
    workload = variant.get("workload")
    try:
        if workload == "pipeline_schedule_benchmark":
            result = pipeline_schedule_benchmark(
                config_path=config_path,
                config=config,
                variant=variant,
                output_dir=output_dir,
            )
        elif workload == "tensor_sequence_parallel_benchmark":
            result = tensor_sequence_parallel_benchmark(
                config_path=config_path,
                config=config,
                variant=variant,
                output_dir=output_dir,
            )
        elif workload == "context_parallel_benchmark":
            result = context_parallel_benchmark(
                config_path=config_path,
                config=config,
                variant=variant,
                output_dir=output_dir,
            )
        elif workload == "hybrid_tp_dp_benchmark":
            result = hybrid_tp_dp_benchmark(
                config_path=config_path,
                config=config,
                variant=variant,
                output_dir=output_dir,
            )
        else:
            raise RunnerError(f"no measured Megatron executor route for workload: {workload}")
    finally:
        cleanup_distributed()
    append_jsonl(run_dir / "metrics" / "variant_results.jsonl", result)
    return 0


def worker_command(config_path: Path, variant_id: str, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "common.megatron_executor",
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
            "common.megatron_executor",
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


def execute_megatron_plan(plan: dict[str, Any], config: dict[str, Any]) -> int:
    run_dir = Path(plan["run_dir"])
    write_json(run_dir / "manifest.json", {"plan": plan, "status": "running"})
    variants = {variant["id"]: variant for variant in configured_variants(config)}
    for planned in plan["variants"]:
        variant = variants[planned["id"]]
        run_variant_subprocess(plan, variant)
    write_json(run_dir / "manifest.json", {"plan": plan, "status": "completed"})
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Worker process for measured Megatron variants.")
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
        print(f"megatron executor failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
