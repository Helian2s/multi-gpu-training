"""Shared runner contract for accepted experiment entry points.

The helpers in this module are intentionally provider-neutral. They prepare and
validate experiment plans inside the repository or container; AWS/Runpod
resource lifecycle remains under ``infra/``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunnerError(RuntimeError):
    """Raised when a runner contract is invalid or unsafe to execute."""


PlanExecutor = Callable[[dict[str, Any], dict[str, Any]], int | None]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RunnerError(
            "PyYAML is required by experiment runners; install the runtime "
            "requirements before launching an experiment"
        ) from error

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise RunnerError(f"cannot read experiment config: {path}") from error
    if not isinstance(data, dict):
        raise RunnerError(f"experiment config is not a YAML mapping: {path}")
    return data


def experiment_id(config: dict[str, Any]) -> str:
    try:
        value = config["experiment"]["id"]
    except KeyError as error:
        raise RunnerError("experiment config has no experiment.id") from error
    if not isinstance(value, str) or not value.startswith("EXP-"):
        raise RunnerError(f"invalid experiment.id: {value!r}")
    return value


def stack_image(config: dict[str, Any]) -> str | None:
    stack = config.get("stack")
    if not isinstance(stack, dict):
        raise RunnerError("experiment config has no stack mapping")
    image = stack.get("image")
    if image is None:
        return None
    if not isinstance(image, str) or "@sha256:" not in image:
        raise RunnerError("stack.image must be an immutable digest reference")
    return image


def apply_image_override(config: dict[str, Any], image_ref: str | None) -> None:
    if not image_ref:
        return
    if "@sha256:" not in image_ref:
        raise RunnerError("--image-ref must be an immutable digest reference")
    stack = config.get("stack")
    if not isinstance(stack, dict):
        raise RunnerError("experiment config has no stack mapping")
    stack["image"] = image_ref


def configured_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = config.get("sweep")
    if not isinstance(sweep, dict):
        raise RunnerError("experiment config has no sweep mapping")
    variants = sweep.get("variants")
    if not isinstance(variants, list):
        raise RunnerError("experiment config sweep.variants must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise RunnerError(f"variant {index} is not a mapping")
        variant_id = variant.get("id")
        if not isinstance(variant_id, str) or not variant_id:
            raise RunnerError(f"variant {index} has no string id")
        if variant_id in seen:
            raise RunnerError(f"duplicate variant id: {variant_id}")
        seen.add(variant_id)
        normalized.append(variant)
    return normalized


def variant_run_unit(variant: dict[str, Any]) -> str:
    run_unit = variant.get("run_unit")
    if not isinstance(run_unit, str) or not run_unit:
        raise RunnerError(f"variant {variant.get('id', '<unknown>')} has no run_unit")
    return run_unit


def select_variants(
    variants: Sequence[dict[str, Any]],
    *,
    variant_ids: Iterable[str] = (),
    run_units: Iterable[str] = (),
) -> list[dict[str, Any]]:
    selected_variant_ids = set(variant_ids)
    selected_run_units = set(run_units)
    known_variant_ids = {variant["id"] for variant in variants}
    known_run_units = {variant_run_unit(variant) for variant in variants}
    unknown_variants = sorted(selected_variant_ids - known_variant_ids)
    unknown_run_units = sorted(selected_run_units - known_run_units)
    if unknown_variants:
        raise RunnerError(f"unknown variant id(s): {', '.join(unknown_variants)}")
    if unknown_run_units:
        raise RunnerError(f"unknown run unit(s): {', '.join(unknown_run_units)}")

    result = []
    for variant in variants:
        if selected_variant_ids and variant["id"] not in selected_variant_ids:
            continue
        if selected_run_units and variant_run_unit(variant) not in selected_run_units:
            continue
        result.append(variant)
    if not result:
        raise RunnerError("no variants selected")
    return result


def visible_devices(variant: dict[str, Any]) -> list[int]:
    devices = variant.get("visible_devices", [])
    if not isinstance(devices, list) or any(not isinstance(item, int) for item in devices):
        raise RunnerError(f"variant {variant['id']} has invalid visible_devices")
    return devices


def resolve_output_root(config_path: Path, config: dict[str, Any], override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise RunnerError("experiment config has no outputs mapping")
    root = outputs.get("root")
    if not isinstance(root, str) or not root:
        raise RunnerError("experiment config outputs.root must be a string")
    return (config_path.parent / root).resolve()


def build_plan(
    *,
    config_path: Path,
    config: dict[str, Any],
    selected: Sequence[dict[str, Any]],
    run_id: str,
    output_root: Path,
    mode: str,
) -> dict[str, Any]:
    exp_id = experiment_id(config)
    run_dir = output_root / exp_id / run_id
    image = stack_image(config)
    run_units = sorted({variant_run_unit(variant) for variant in selected})
    plan_variants = []
    for variant in selected:
        plan_variants.append(
            {
                "id": variant["id"],
                "run_unit": variant_run_unit(variant),
                "visible_devices": visible_devices(variant),
                "world_size": variant.get("world_size", len(visible_devices(variant)) or 1),
                "workload": (
                    variant.get("workload")
                    or variant.get("strategy")
                    or variant.get("case")
                    or variant.get("view")
                ),
            }
        )
    return {
        "schema_version": 1,
        "mode": mode,
        "experiment_id": exp_id,
        "config_path": str(config_path),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "image": image,
        "image_ready": image is not None,
        "run_units": run_units,
        "variant_count": len(plan_variants),
        "variants": plan_variants,
    }


def write_plan(plan: dict[str, Any]) -> Path:
    run_dir = Path(plan["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def list_variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "id": variant["id"],
            "run_unit": variant_run_unit(variant),
            "visible_devices": visible_devices(variant),
            "world_size": variant.get("world_size", len(visible_devices(variant)) or 1),
            "workload": (
                variant.get("workload")
                or variant.get("strategy")
                or variant.get("case")
                or variant.get("view")
            ),
        }
        for variant in configured_variants(config)
    ]


def fail_closed_executor(plan: dict[str, Any], _config: dict[str, Any]) -> int:
    raise RunnerError(
        f"{plan['experiment_id']} has a validated dry-run plan but no measured "
        "GPU workload executor yet; refusing to create placeholder results"
    )


def build_parser(description: str, default_config: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID") or utc_timestamp())
    parser.add_argument("--run-unit", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--output-root")
    parser.add_argument(
        "--image-ref",
        default=os.environ.get("MULTI_GPU_TRAINING_IMAGE_REF"),
        help="Runtime immutable image reference supplied by provider queue tooling.",
    )
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--list-variants", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-unpublished-image",
        action="store_true",
        help="Allow execute mode before stack.image has an immutable digest. Intended only for local smoke testing.",
    )
    return parser


def runner_main(
    *,
    experiment: str,
    default_config: Path,
    description: str,
    executor: PlanExecutor = fail_closed_executor,
    argv: Sequence[str] | None = None,
) -> int:
    parser = build_parser(description, default_config)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    try:
        config = load_yaml(config_path)
        apply_image_override(config, args.image_ref)
        if experiment_id(config) != experiment:
            raise RunnerError(
                f"runner is for {experiment}, config is for {experiment_id(config)}"
            )
        if args.list_variants:
            print(json.dumps(list_variants(config), indent=2, sort_keys=True))
            return 0
        selected = select_variants(
            configured_variants(config),
            variant_ids=args.variant,
            run_units=args.run_unit,
        )
        mode = "dry-run" if args.dry_run else "execute"
        plan = build_plan(
            config_path=config_path,
            config=config,
            selected=selected,
            run_id=args.run_id,
            output_root=resolve_output_root(config_path, config, args.output_root),
            mode=mode,
        )
        if args.dry_run:
            if args.write_plan:
                path = write_plan(plan)
                plan = {**plan, "plan_path": str(path)}
            print(json.dumps(plan, indent=2, sort_keys=True))
            return 0
        if not plan["image_ready"] and not args.allow_unpublished_image:
            raise RunnerError(
                "execute mode requires stack.image to contain an immutable digest; "
                "rebuild and publish the image, then record the digest first"
            )
        path = write_plan(plan)
        print(f"wrote execution plan: {path}", flush=True)
        result = executor(plan, config)
        return 0 if result is None else result
    except RunnerError as error:
        parser.exit(2, f"{parser.prog}: error: {error}\n")
