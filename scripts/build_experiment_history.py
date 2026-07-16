#!/usr/bin/env python3
"""Build RAG-friendly experiment history markdown from specs and artifacts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
HISTORY_DIR = ROOT / "docs" / "experiment_history"
ARTIFACT_ROOTS = [
    ROOT / "artifacts" / "runs" / "aws-s3-mirror",
    ROOT / "artifacts" / "runs" / "runpod-volume-mirror",
]


EXPERIMENT_NOTES: dict[str, dict[str, str | list[str]]] = {
    "EXP-01": {
        "conclusion": (
            "AWS-A2 G7e exposed two RTX PRO 6000 Blackwell GPUs over PIX/PCIe. "
            "CUDA peer access worked in both directions. P2P writes reached about "
            "55 GB/s per direction and about 105 GB/s bidirectional, while the "
            "NCCL collectives showed materially lower bus bandwidth than the "
            "Runpod NVLink baseline."
        ),
        "interpretation": (
            "PCIe P2P is usable and much faster than host-staged fallback for "
            "latency, but it is not equivalent to NVLink. The AWS logs show NCCL "
            "building local rings over PIX and using direct CUDA P2P for the "
            "two local GPUs."
        ),
        "takeaway": (
            "Always qualify topology before interpreting distributed-training "
            "speed. Two GPUs on one host can have peer access while still being "
            "limited by PCIe-class bandwidth."
        ),
    },
    "EXP-02": {
        "conclusion": (
            "Reduced precision accelerated eligible matrix and training paths on "
            "AWS-A2. BF16 was the strongest measured training mode: one-rank "
            "BF16 cut step time from about 322 ms to 113 ms and peak memory from "
            "about 32.1 GiB to 16.1 GiB versus FP32. FP8 was not admitted because "
            "the native Transformer Engine SM120 path was not proven."
        ),
        "interpretation": (
            "TF32 improved FP32-compatible matrix work, and BF16/FP16 GEMMs were "
            "the fastest microbenchmarks. In the bounded training workload, BF16 "
            "also reduced activation/parameter memory, while FP16 used FP32 "
            "parameters with autocast/GradScaler after the local executor fix."
        ),
        "takeaway": (
            "Do not choose a precision mode from marketing claims alone: record "
            "kernel eligibility, finite loss, memory, and distributed behavior. "
            "BF16 is the practical default here; unsupported FP8 remains excluded."
        ),
    },
    "EXP-03": {
        "conclusion": (
            "With fixed effective global batch 8 on one AWS GPU, larger "
            "microbatches reduced optimizer-step time until microbatch 4, then "
            "memory pressure rose sharply at microbatch 8. All variants produced "
            "finite loss."
        ),
        "interpretation": (
            "More accumulation steps lower peak memory but increase repeated "
            "forward/backward overhead per optimizer update. The best point in "
            "this short run was microbatch 4 with accumulation 2, not the largest "
            "microbatch."
        ),
        "takeaway": (
            "Tune microbatch and accumulation together. Gradient accumulation "
            "controls effective batch without requiring the largest possible "
            "microbatch."
        ),
    },
    "EXP-04": {
        "conclusion": (
            "Activation checkpointing reduced peak memory at both sequence "
            "lengths and increased step time. At sequence length 4096, peak memory "
            "fell from about 28.7 GiB to 17.1 GiB, while step time rose from about "
            "269 ms to 318 ms."
        ),
        "interpretation": (
            "The run shows the expected recomputation trade: store fewer "
            "activations during forward, recompute during backward, and pay extra "
            "compute time to fit longer or larger batches."
        ),
        "takeaway": (
            "Use checkpointing when memory is the binding constraint; avoid it "
            "when the same workload already fits and throughput is the priority."
        ),
    },
    "EXP-05": {
        "conclusion": (
            "Automatic/FlashAttention SDPA was dramatically faster and smaller "
            "than the math backend for the Qwen-like GQA shape. Automatic SDPA "
            "measured about 0.171 ms/iteration versus 4.777 ms/iteration for math."
        ),
        "interpretation": (
            "Backend selection dominated this attention microbenchmark. "
            "`torch.compile` reduced math-backend steady-state time but added "
            "compile warmup, while automatic compiled SDPA was slightly slower "
            "than automatic eager in this short measurement."
        ),
        "takeaway": (
            "Verify which attention backend actually ran. A fused attention path "
            "can matter more than small code-level changes."
        ),
    },
    "EXP-06": {
        "conclusion": (
            "Profiler triangulation produced PyTorch Profiler traces/key averages "
            "for math and automatic attention backends, and confirmed Nsight "
            "Systems and Nsight Compute were available with prepared commands."
        ),
        "interpretation": (
            "The run separates profiler roles: PyTorch Profiler captured operator "
            "and timeline artifacts directly, while Nsight tools were validated "
            "for deeper host/CUDA/kernel inspection from an interactive host shell."
        ),
        "takeaway": (
            "Use PyTorch Profiler for framework-level triage first, then Nsight "
            "Systems/Compute when the question requires CUDA timeline or kernel "
            "counter evidence."
        ),
    },
    "EXP-07": {
        "conclusion": (
            "The two-rank DDP variants completed on AWS-A2. One rank measured "
            "about 113 ms/step, while two-rank DDP with batch 2 measured about "
            "240 ms/step; this is not a throughput speedup for the small "
            "one-sample-per-rank workload. A larger bucket variant was slightly "
            "faster than the default/small bucket variants."
        ),
        "interpretation": (
            "For this bounded workload, communication/synchronization overhead "
            "and small local work dominated the benefit of adding a second GPU. "
            "No-sync gradient accumulation traded fewer synchronizations for "
            "larger effective batch and longer optimizer-step windows."
        ),
        "takeaway": (
            "DDP only scales when each rank has enough local compute to amortize "
            "gradient synchronization. Bucket settings and accumulation change "
            "that balance."
        ),
    },
    "EXP-08": {
        "conclusion": (
            "FSDP completed on two AWS GPUs and reduced peak memory versus DDP "
            "from about 19.3 GiB to 14.5 GiB per rank while also reducing step "
            "time in this short run from about 239 ms to about 205 ms. The "
            "prefetch variant was skipped because its admission gate was not met."
        ),
        "interpretation": (
            "The measured FSDP variants sharded model/gradient/optimizer state "
            "enough to lower memory. The result is encouraging but remains a "
            "short controlled run, not a general claim that FSDP is always faster."
        ),
        "takeaway": (
            "Use sharding to trade extra distributed machinery for lower per-rank "
            "state memory. Validate correctness and state-dict behavior before "
            "treating FSDP as a production recipe."
        ),
    },
    "EXP-09": {
        "conclusion": (
            "Controlled failure cases produced the intended signatures: healthy "
            "smoke passed, OOM and FP16 overflow were expected failures, input "
            "starvation produced a timed wait, mismatched collectives surfaced "
            "an explicit collective fingerprint mismatch, and a timeout case "
            "recorded bounded timeout evidence."
        ),
        "interpretation": (
            "The experiment created known-bad cases so future troubleshooting can "
            "distinguish memory pressure, numerical overflow, input stalls, rank "
            "stragglers, collective mismatches, and NCCL timeout symptoms."
        ),
        "takeaway": (
            "Good distributed debugging starts from the failure signature: OOM, "
            "non-finite gradients, input wait, rank mismatch, and timeout each "
            "point to different first checks."
        ),
    },
    "EXP-10": {
        "conclusion": (
            "Runpod A2 exposed two A100-SXM4-80GB GPUs with NV12 topology. P2P "
            "writes reached about 269-274 GB/s per direction and about "
            "518-525 GB/s bidirectional. NCCL collective bus bandwidth was much "
            "higher than AWS PCIe for comparable two-GPU collectives."
        ),
        "interpretation": (
            "The A100 SXM host provided NVLink/NVS topology with 12 links per "
            "GPU, so it is the right environment for the topology-sensitive "
            "Megatron TP/SP/CP experiments."
        ),
        "takeaway": (
            "NVLink qualification should record both `nvidia-smi topo -m` and "
            "measured P2P/NCCL behavior. Do not infer NVLink just from GPU name."
        ),
    },
    "EXP-11": {
        "conclusion": (
            "Tensor parallelism reduced per-rank memory from about 190 MiB to "
            "106 MiB and introduced non-zero collective traffic. Sequence "
            "parallelism reduced local sequence length from 1024 to 512 and "
            "slightly lowered memory further, but step time increased in this "
            "small synthetic workload."
        ),
        "interpretation": (
            "TP=2 split the intermediate dimension and changed communication. "
            "SP changed token placement within the TP group, which is useful "
            "when sequence activation memory is the constraint, even if the small "
            "benchmark does not show a speed gain."
        ),
        "takeaway": (
            "Tensor parallelism is primarily a model-sharding tool; sequence "
            "parallelism targets activation placement. Measure both memory and "
            "collective cost."
        ),
    },
    "EXP-12": {
        "conclusion": (
            "Pipeline schedule mechanics were validated on AWS-A2. More "
            "microbatches reduced estimated bubble fraction, PP=2 introduced "
            "point-to-point activation/gradient traffic, and imbalanced stages "
            "were slower than balanced stages."
        ),
        "interpretation": (
            "The AWS PCIe run is valid for schedule/bubble mechanics but not as "
            "a Runpod NVLink Megatron throughput claim."
        ),
        "takeaway": (
            "Pipeline parallelism needs enough microbatches and balanced stage "
            "work. Splitting layers alone does not guarantee utilization."
        ),
    },
    "EXP-13": {
        "conclusion": (
            "Context parallel variants completed on Runpod A2. CP=2 halved local "
            "sequence length for the same global sequence and lowered memory "
            "versus comparable one-rank long-sequence cases, while adding "
            "collective traffic."
        ),
        "interpretation": (
            "The run demonstrates the intended CP geometry: sequence context is "
            "partitioned across ranks, reducing local token/activation pressure "
            "at the cost of communication."
        ),
        "takeaway": (
            "Use context parallelism when long context memory is the constraint; "
            "expect communication cost and validate attention/backend support."
        ),
    },
    "EXP-14": {
        "conclusion": (
            "The four-GPU Runpod A4 hybrid run completed. DP=4 gave the highest "
            "aggregate tokens/s in the bounded workload, TP=4 used the least "
            "memory, and TP=2 x DP=2 landed between them on throughput, memory, "
            "and collective volume."
        ),
        "interpretation": (
            "The experiment validated non-trivial tensor-parallel and "
            "data-parallel process groups on one NVLink-connected four-GPU host. "
            "The result is process-group and trade-off evidence, not a full Qwen3 "
            "Megatron recipe benchmark."
        ),
        "takeaway": (
            "TP=2 x DP=2 is the smallest four-rank layout that exercises both "
            "model sharding and data replication. Choose it when a model needs "
            "some sharding but still benefits from replicated throughput."
        ),
    },
}


OPERATIONAL_TIMELINE = [
    (
        "AWS-A1 capacity and user-data fix",
        "Initial AWS-A1 capacity probes across us-west-2 AZs failed with "
        "InsufficientInstanceCapacity. A later AWS-A1 launch in us-west-2b "
        "reached SSM and showed one RTX PRO 6000 GPU, but bootstrap failed "
        "because generated heredoc delimiters were indented. Tests and launch "
        "generation were fixed before the successful AWS-A1 queue run.",
    ),
    (
        "AWS-A2 launch and queue generator fixes",
        "AWS-A2 capacity was scarce. The first created AWS-A2 instance failed "
        "before measurement because queue user-data wrote shell commands through "
        "a TSV file and lost JSON command fields. The queue generator was fixed "
        "to emit shell-quoted run-unit calls.",
    ),
    (
        "EXP-02 FP16 executor fix",
        "A first AWS-A2 retry reached EXP-02 and failed because FP16 training "
        "loaded model parameters as FP16 while also using GradScaler. The "
        "executor was changed so FP16+GradScaler keeps model parameters in FP32 "
        "and uses FP16 autocast.",
    ),
    (
        "AWS-A2 PyTorch completion",
        "The run aws-a2-full-fp16fix-20260715T023252Z completed EXP-01, EXP-02, "
        "EXP-07, EXP-08, and EXP-09 with exit status 0. The same run provided "
        "the AWS two-GPU communication, precision, DDP, FSDP, and failure "
        "diagnosis measurements.",
    ),
    (
        "AWS-A2 Megatron completion",
        "EXP-12 first hit capacity failures in fixed AZ attempts. The successful "
        "AWS-selected placement launched in us-west-2b, ran QUAL-A2 and "
        "EXP-12-A2V1/A2V2, and stopped after the success hold.",
    ),
    (
        "Runpod registry and image readiness",
        "Early Runpod attempts exposed GHCR authorization/image-start issues. "
        "The usable path became the GHCR NeMo image with Runpod SSH startup and "
        "CUDA 12.8 support, pulled by digest through a read-only registry auth.",
    ),
    (
        "Runpod A2 completion",
        "The run runpod-a2-megatron-20260715T203057Z completed QUAL-RUNPOD-A2, "
        "EXP-10, EXP-11, and EXP-13. Artifacts were copied from the Pod volume "
        "into the local runpod-volume mirror before the Pod was stopped.",
    ),
    (
        "EXP-14 cleanup hotfix",
        "The first Runpod A4 run wrote metrics but hung during explicit NCCL "
        "process-group destruction. The successful rerun used a container-side "
        "hotpatch, now committed in source, to skip explicit destroy by default "
        "and rely on short-lived subprocess exit for NCCL cleanup.",
    ),
]


@dataclass(frozen=True)
class ExperimentSpec:
    exp_id: str
    title: str
    directory: Path
    config: dict[str, Any]
    hypothesis_sections: dict[str, str]
    educational_goal: str


def strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def parse_markdown_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = defaultdict(list)
    current = "Preamble"
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            continue
        if line.startswith("# "):
            continue
        sections[current].append(line)
    return {
        key: "\n".join(value).strip()
        for key, value in sections.items()
        if "\n".join(value).strip()
    }


def catalog_detail_sections() -> dict[str, str]:
    text = read_text(ROOT / "EXPERIMENT_CATALOG.md")
    matches = list(re.finditer(r"^### (EXP-\d{2}): .*$", text, re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        exp_id = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[exp_id] = text[start:end].strip()
    return sections


def extract_educational_goals() -> dict[str, str]:
    goals: dict[str, str] = {}
    for exp_id, section in catalog_detail_sections().items():
        match = re.search(
            r"\*\*Educational goal:\*\*\s*(.+?)(?:\n\n|\n\*\*|\Z)",
            section,
            flags=re.DOTALL,
        )
        if match:
            goals[exp_id] = re.sub(r"\s+", " ", match.group(1)).strip()
    return goals


def load_specs() -> list[ExperimentSpec]:
    goals = extract_educational_goals()
    specs: list[ExperimentSpec] = []
    for directory in sorted((ROOT / "experiments").glob("exp_[0-9][0-9]_*")):
        config = load_yaml(directory / "experiment.yaml")
        experiment = config.get("experiment", {})
        exp_id = experiment.get("id", directory.name.split("_", maxsplit=2)[1].upper())
        title = experiment.get("title", directory.name)
        sections = parse_markdown_sections(read_text(directory / "hypothesis.md"))
        specs.append(
            ExperimentSpec(
                exp_id=exp_id,
                title=title,
                directory=directory,
                config=config,
                hypothesis_sections=sections,
                educational_goal=goals.get(exp_id, ""),
            )
        )
    return specs


def run_dirs_for(exp_id: str) -> list[Path]:
    paths: list[Path] = []
    for root in ARTIFACT_ROOTS:
        direct = root / exp_id
        if not direct.exists():
            continue
        runs_child = direct / "runs"
        if runs_child.is_dir():
            paths.extend(p for p in runs_child.iterdir() if p.is_dir())
        else:
            paths.extend(p for p in direct.iterdir() if p.is_dir())
    return sorted(paths)


def qualification_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in ARTIFACT_ROOTS:
        if not root.is_dir():
            continue
        for path in root.glob("QUAL-*"):
            if path.is_dir():
                runs_child = path / "runs"
                if runs_child.is_dir():
                    dirs.extend(p for p in runs_child.iterdir() if p.is_dir())
                else:
                    dirs.extend(p for p in path.iterdir() if p.is_dir())
    return sorted(dirs)


def queue_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in ARTIFACT_ROOTS:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if not path.is_dir() or path.name.startswith("EXP-") or path.name.startswith("QUAL-"):
                continue
            runs_child = path / "runs"
            if runs_child.is_dir():
                dirs.extend(p for p in runs_child.iterdir() if p.is_dir())
            else:
                dirs.extend(p for p in path.iterdir() if p.is_dir())
    return sorted(dirs)


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def bytes_to_gib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{value / (1024 ** 3):.2f}"


def bytes_to_mib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{value / (1024 ** 2):.1f}"


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = []
        for value in row:
            cell = str(value).replace("\n", "<br>").replace("|", "\\|")
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def group_rows_by_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("variant_id", "unknown")].append(row)

    out: list[dict[str, Any]] = []
    for variant_id, items in grouped.items():
        first = dict(items[0])
        first["variant_id"] = variant_id
        ranks = sorted({item.get("rank") for item in items if item.get("rank") is not None})
        if ranks:
            first["ranks"] = ",".join(str(rank) for rank in ranks)
        for key in [
            "mean_step_ms",
            "max_rank_mean_step_ms",
            "peak_allocated_bytes",
            "elapsed_seconds",
            "tokens_per_second",
            "estimated_collective_bytes",
            "p2p_bytes",
        ]:
            values = [item.get(key) for item in items if isinstance(item.get(key), (int, float))]
            if values:
                if key in {"mean_step_ms", "peak_allocated_bytes", "elapsed_seconds"}:
                    first[key] = max(values)
                else:
                    first[key] = values[0]
        losses = [
            f"r{item.get('rank')}={fmt_num(item.get('final_loss'))}"
            for item in items
            if item.get("final_loss") is not None
        ]
        if losses:
            first["losses"] = "; ".join(losses)
        statuses = sorted({str(item.get("status")) for item in items if item.get("status")})
        if statuses:
            first["status"] = ",".join(statuses)
        out.append(first)
    return out


def metrics_table(exp_id: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    grouped = group_rows_by_variant(rows)
    if exp_id in {"EXP-02"}:
        gemm = [row for row in grouped if row.get("workload") is None and row.get("tflops") is not None]
        train = [row for row in grouped if row.get("strategy")]
        skipped = [row for row in grouped if row.get("status") == "skipped"]
        parts = []
        if gemm:
            parts.append("GEMM microbenchmark variants:\n\n" + md_table(
                ["Variant", "Precision", "Shape", "Matrix", "Elapsed ms", "TFLOP/s"],
                [
                    [
                        row.get("variant_id"),
                        row.get("precision"),
                        row.get("shape"),
                        row.get("matrix_size"),
                        fmt_num(row.get("elapsed_ms")),
                        fmt_num(row.get("tflops")),
                    ]
                    for row in gemm
                ],
            ))
        if train:
            parts.append("Training variants:\n\n" + training_table(train))
        if skipped:
            parts.append("Skipped variants:\n\n" + md_table(
                ["Variant", "Status", "Detail"],
                [[row.get("variant_id"), row.get("status"), row.get("detail")] for row in skipped],
            ))
        return "\n\n".join(parts)
    if exp_id in {"EXP-03", "EXP-04", "EXP-07", "EXP-08"}:
        return training_table(grouped)
    if exp_id == "EXP-05":
        return md_table(
            [
                "Variant",
                "Backend",
                "Compile",
                "Mean iteration ms",
                "Peak MiB",
                "Warmup/compile s",
                "Checksum",
            ],
            [
                [
                    row.get("variant_id"),
                    row.get("backend"),
                    fmt_num(row.get("torch_compile")),
                    fmt_num(row.get("mean_iteration_ms")),
                    bytes_to_mib(row.get("peak_allocated_bytes")),
                    fmt_num(row.get("compile_or_first_warmup_seconds")),
                    fmt_num(row.get("output_checksum")),
                ]
                for row in grouped
            ],
        )
    if exp_id == "EXP-06":
        return md_table(
            ["Variant", "Profiler", "Target", "Status", "Artifact or command"],
            [
                [
                    row.get("variant_id"),
                    row.get("profiler"),
                    row.get("target_backend"),
                    row.get("status"),
                    row.get("key_averages") or row.get("prepared_command") or row.get("trace") or "",
                ]
                for row in grouped
            ],
        )
    if exp_id == "EXP-09":
        return md_table(
            ["Variant", "Case", "World", "Status", "Elapsed s", "Detail"],
            [
                [
                    row.get("variant_id"),
                    row.get("case"),
                    row.get("world_size"),
                    row.get("status"),
                    fmt_num(row.get("elapsed_seconds")),
                    truncate(row.get("detail", ""), 180),
                ]
                for row in grouped
            ],
        )
    if exp_id == "EXP-11":
        return md_table(
            [
                "Variant",
                "TP",
                "SP",
                "World",
                "Local seq",
                "Shard intermediate",
                "Max step ms",
                "Tokens/s",
                "Peak MiB",
                "Collective MiB",
            ],
            [
                [
                    row.get("variant_id"),
                    row.get("tensor_parallel_size"),
                    fmt_num(row.get("sequence_parallel")),
                    row.get("world_size"),
                    row.get("local_sequence_length"),
                    row.get("shard_intermediate_size"),
                    fmt_num(row.get("max_rank_mean_step_ms") or row.get("mean_step_ms")),
                    fmt_num(row.get("tokens_per_second")),
                    bytes_to_mib(row.get("peak_allocated_bytes")),
                    bytes_to_mib(row.get("estimated_collective_bytes")),
                ]
                for row in grouped
            ],
        )
    if exp_id == "EXP-12":
        return md_table(
            [
                "Variant",
                "Schedule",
                "PP",
                "Microbatches",
                "Bubble",
                "Max step ms",
                "Tokens/s",
                "Peak MiB",
                "P2P MiB",
                "Stages",
            ],
            [
                [
                    row.get("variant_id"),
                    row.get("schedule"),
                    row.get("pipeline_parallel_size"),
                    row.get("num_microbatches"),
                    fmt_num(row.get("estimated_pipeline_bubble_fraction")),
                    fmt_num(row.get("max_rank_mean_step_ms") or row.get("mean_step_ms")),
                    fmt_num(row.get("tokens_per_second")),
                    bytes_to_mib(row.get("peak_allocated_bytes")),
                    bytes_to_mib(row.get("p2p_bytes")),
                    row.get("stage_layers"),
                ]
                for row in grouped
            ],
        )
    if exp_id == "EXP-13":
        return md_table(
            [
                "Variant",
                "CP",
                "World",
                "Seq",
                "Local seq",
                "Checkpointing",
                "Max step ms",
                "Tokens/s",
                "Peak MiB",
                "Collective MiB",
            ],
            [
                [
                    row.get("variant_id"),
                    row.get("context_parallel_size"),
                    row.get("world_size"),
                    row.get("sequence_length"),
                    row.get("local_sequence_length"),
                    fmt_num(row.get("activation_checkpointing")),
                    fmt_num(row.get("max_rank_mean_step_ms") or row.get("mean_step_ms")),
                    fmt_num(row.get("tokens_per_second")),
                    bytes_to_mib(row.get("peak_allocated_bytes")),
                    bytes_to_mib(row.get("estimated_collective_bytes")),
                ]
                for row in grouped
            ],
        )
    if exp_id == "EXP-14":
        return md_table(
            [
                "Variant",
                "Layout",
                "TP",
                "DP",
                "World",
                "TP ranks",
                "DP ranks",
                "Max step ms",
                "Tokens/s",
                "Peak MiB",
                "Collective MiB",
            ],
            [
                [
                    row.get("variant_id"),
                    row.get("layout"),
                    row.get("tensor_parallel_size"),
                    row.get("data_parallel_size"),
                    row.get("world_size"),
                    row.get("tp_ranks"),
                    row.get("dp_ranks"),
                    fmt_num(row.get("max_rank_mean_step_ms") or row.get("mean_step_ms")),
                    fmt_num(row.get("tokens_per_second")),
                    bytes_to_mib(row.get("peak_allocated_bytes")),
                    bytes_to_mib(row.get("estimated_collective_bytes")),
                ]
                for row in grouped
            ],
        )
    return md_table(
        sorted({key for row in grouped for key in row.keys()})[:12],
        [[row.get(key, "") for key in sorted({key for row in grouped for key in row.keys()})[:12]] for row in grouped],
    )


def training_table(rows: list[dict[str, Any]]) -> str:
    return md_table(
        [
            "Variant",
            "Strategy",
            "Precision",
            "World",
            "Microbatch",
            "Accum",
            "Global batch",
            "Seq",
            "Mean step ms",
            "Peak GiB",
            "Loss",
            "Finite",
        ],
        [
            [
                row.get("variant_id"),
                row.get("strategy"),
                row.get("precision"),
                row.get("world_size"),
                row.get("per_rank_microbatch_size"),
                row.get("gradient_accumulation_steps"),
                row.get("effective_global_batch_size"),
                row.get("sequence_length"),
                fmt_num(row.get("mean_step_ms")),
                bytes_to_gib(row.get("peak_allocated_bytes")),
                row.get("losses") or fmt_num(row.get("final_loss")),
                fmt_num(row.get("finite_loss")),
            ]
            for row in rows
        ],
    )


def truncate(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def parse_nccl_avg_busbw(raw_dir: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    names = [
        ("all_reduce", "nccl_all_reduce.txt"),
        ("all_gather", "nccl_all_gather.txt"),
        ("broadcast", "nccl_broadcast.txt"),
        ("reduce_scatter", "nccl_reduce_scatter.txt"),
        ("all_to_all", "nccl_all_to_all.txt"),
    ]
    for label, filename in names:
        text = read_text(raw_dir / filename)
        match = re.search(r"# Avg bus bandwidth\s*:\s*([0-9.]+)", text)
        if match:
            rows.append([label, match.group(1), rel(raw_dir / filename)])
    return rows


def parse_p2p_summary(raw_dir: Path) -> list[str]:
    text = read_text(raw_dir / "p2pBandwidthLatencyTest.txt")
    if not text:
        return []
    summary: list[str] = []
    if "Device=0 CAN Access Peer Device=1" in text and "Device=1 CAN Access Peer Device=0" in text:
        summary.append("CUDA peer access is enabled in both directions.")
    uni = extract_matrix_pair(text, "Unidirectional P2P=Enabled Bandwidth")
    bi = extract_matrix_pair(text, "Bidirectional P2P=Enabled Bandwidth")
    lat = extract_matrix_pair(text, "P2P=Enabled Latency")
    if uni:
        summary.append(f"Unidirectional P2P write bandwidth: GPU0->GPU1 {uni[0]} GB/s, GPU1->GPU0 {uni[1]} GB/s.")
    if bi:
        summary.append(f"Bidirectional P2P bandwidth: GPU0<->GPU1 {bi[0]} and {bi[1]} GB/s.")
    if lat:
        summary.append(f"Enabled GPU P2P latency: GPU0->GPU1 {lat[0]} us, GPU1->GPU0 {lat[1]} us.")
    return summary


def extract_matrix_pair(text: str, label: str) -> tuple[str, str] | None:
    start = text.find(label)
    if start == -1:
        return None
    block = text[start : start + 450]
    rows = []
    for line in block.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] in {"0", "1"}:
            rows.append(parts)
    if len(rows) < 2:
        return None
    return rows[0][2], rows[1][1]


def communication_section(run_dir: Path) -> str:
    raw_dir = run_dir / "raw"
    rows = parse_nccl_avg_busbw(raw_dir)
    p2p = parse_p2p_summary(raw_dir)
    topo = strip_ansi(read_text(raw_dir / "nvidia-smi-topo-m.txt")).strip()
    parts = []
    if p2p:
        parts.append("\n".join(f"- {item}" for item in p2p))
    if rows:
        parts.append("NCCL average bus bandwidth:\n\n" + md_table(
            ["Collective", "Avg bus bandwidth GB/s", "Raw log"],
            rows,
        ))
    if topo:
        topo_lines = "\n".join(topo.splitlines()[:12])
        parts.append("Topology excerpt:\n\n```text\n" + topo_lines + "\n```")
    return "\n\n".join(parts)


def run_inventory(run_dir: Path) -> str:
    manifest = read_json(run_dir / "manifest.json")
    plan = read_json(run_dir / "plan.json") or manifest.get("plan", {})
    run_units = plan.get("run_units", [])
    variants = plan.get("variants", [])
    status_files = sorted(run_dir.glob("exit_status*.txt"))
    finished_files = sorted(run_dir.glob("finished*.txt"))
    raw_files = sorted((run_dir / "raw").glob("**/*")) if (run_dir / "raw").is_dir() else []
    metric_file = run_dir / "metrics" / "variant_results.jsonl"
    bullets = [
        f"- Run ID: `{run_dir.name}`",
        f"- Artifact directory: `{rel(run_dir)}`",
    ]
    if plan.get("image"):
        bullets.append(f"- Image: `{plan.get('image')}`")
    if manifest.get("status"):
        bullets.append(f"- Manifest status: `{manifest.get('status')}`")
    if run_units:
        bullets.append("- Run units: " + ", ".join(f"`{unit}`" for unit in run_units))
    if variants:
        bullets.append(f"- Planned variant count in manifest/plan: `{len(variants)}`")
    if metric_file.is_file():
        bullets.append(f"- Metrics JSONL: `{rel(metric_file)}`")
    if status_files:
        statuses = [f"`{path.name}`={read_text(path).strip()}" for path in status_files]
        bullets.append("- Exit statuses: " + ", ".join(statuses))
    if finished_files:
        finishes = [f"`{path.name}`={read_text(path).strip()}" for path in finished_files]
        bullets.append("- Finished UTC markers: " + ", ".join(finishes))
    if raw_files:
        bullets.append(f"- Raw artifact file count below `raw/`: `{sum(1 for p in raw_files if p.is_file())}`")
    return "\n".join(bullets)


def spec_summary(spec: ExperimentSpec) -> str:
    cfg = spec.config
    stack = cfg.get("stack", {})
    hardware = cfg.get("hardware", {})
    workload = cfg.get("workload", {})
    sweep = cfg.get("sweep", {})
    rows = [
        ["Framework/image family", stack.get("family", "")],
        ["Image", f"`{stack.get('image', '')}`" if stack.get("image") else ""],
        ["Provider", hardware.get("provider", "")],
        ["Compute profile", hardware.get("compute_profile", "")],
        ["Resource type", hardware.get("resource_type", "")],
        ["GPU type", hardware.get("gpu_type", "")],
        ["Physical GPUs", hardware.get("physical_gpu_count", "")],
        ["Normal visible GPUs", hardware.get("visible_gpu_count", "") or "variant-dependent"],
        ["Workload profile", workload.get("profile", "")],
        ["Baseline", sweep.get("baseline", "")],
        ["Declared variants", len(sweep.get("variants", []))],
    ]
    return md_table(["Field", "Value"], rows)


def write_experiment_doc(spec: ExperimentSpec) -> None:
    notes = EXPERIMENT_NOTES.get(spec.exp_id, {})
    run_dirs = run_dirs_for(spec.exp_id)
    lines: list[str] = [
        f"# {spec.exp_id}: {spec.title}",
        "",
        "Generated experiment-history file for RAG and cross-workstation continuity.",
        "",
        "Source priority: measured artifact summaries in this file, then the raw "
        "artifact paths listed here, then the formal experiment report if it has "
        "already been completed.",
        "",
        "## Tags",
        "",
        f"`{spec.exp_id}`, `{spec.config.get('stack', {}).get('family', '')}`, "
        f"`{spec.config.get('hardware', {}).get('provider', '')}`, "
        f"`{spec.config.get('hardware', {}).get('compute_profile', '')}`",
        "",
        "## Goal And Design",
        "",
    ]
    if spec.educational_goal:
        lines += ["Educational goal: " + spec.educational_goal, ""]
    for section in ["Scenario", "Question", "Hypothesis", "Decision rule"]:
        text = spec.hypothesis_sections.get(section)
        if text:
            lines += [f"### {section}", "", text, ""]
    lines += ["### Configuration Snapshot", "", spec_summary(spec), ""]
    if notes.get("conclusion"):
        lines += ["## Current Conclusion", "", str(notes["conclusion"]), ""]
    if notes.get("interpretation"):
        lines += ["## Interpretation", "", str(notes["interpretation"]), ""]

    if not run_dirs:
        lines += ["## Run Inventory", "", "No local artifact mirror was found for this experiment.", ""]
    else:
        lines += ["## Run Inventory", ""]
        for run_dir in run_dirs:
            lines += [f"### {run_dir.name}", "", run_inventory(run_dir), ""]

    for run_dir in run_dirs:
        metric_rows = read_jsonl(run_dir / "metrics" / "variant_results.jsonl")
        comm = communication_section(run_dir)
        if metric_rows:
            lines += [f"## Measured Results From `{run_dir.name}`", "", metrics_table(spec.exp_id, metric_rows), ""]
        if comm:
            lines += [f"## Communication And Topology From `{run_dir.name}`", "", comm, ""]

    report_text = read_text(spec.directory / "report.md")
    status_match = re.search(r"Report status:\s*(.+)", report_text)
    report_status = status_match.group(1).strip() if status_match else "not recorded"
    lines += [
        "## Formal Report Status",
        "",
        f"- Report file: `{rel(spec.directory / 'report.md')}`",
        f"- Report status line: {report_status}",
        "",
    ]
    if notes.get("takeaway"):
        lines += ["## Exam / Study Takeaway", "", str(notes["takeaway"]), ""]
    lines += [
        "## Raw Artifact Transfer Note",
        "",
        "The raw files referenced above are intentionally ignored by Git. To move "
        "them to another workstation, transfer the compressed artifact archive "
        "recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before "
        "deeper analysis.",
        "",
    ]
    (HISTORY_DIR / f"{spec.exp_id}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_interim_report(spec: ExperimentSpec) -> None:
    if spec.exp_id in {"EXP-12", "EXP-14"}:
        return
    run_dirs = run_dirs_for(spec.exp_id)
    if not run_dirs:
        return
    notes = EXPERIMENT_NOTES.get(spec.exp_id, {})
    report = spec.directory / "report.md"
    status = "Raw execution complete; formal validation pending"
    lines: list[str] = [
        f"# {spec.exp_id}: {spec.title} - interim report",
        "",
        f"Report status: {status}",
        "",
        "This interim report replaces the old placeholder so repository search "
        "and RAG do not incorrectly report the experiment as not run. It is a "
        "derived summary from local artifact mirrors, not a final validated "
        "publication report. The detailed RAG-oriented history is in "
        f"[docs/experiment_history/{spec.exp_id}.md](../../docs/experiment_history/{spec.exp_id}.md).",
        "",
        "## Executive conclusion",
        "",
        str(notes.get("conclusion", "Raw execution artifacts are present.")),
        "",
        "## Run inventory",
        "",
    ]
    for run_dir in run_dirs:
        lines += [f"### {run_dir.name}", "", run_inventory(run_dir), ""]
    lines += ["## Measured results", ""]
    for run_dir in run_dirs:
        metric_rows = read_jsonl(run_dir / "metrics" / "variant_results.jsonl")
        comm = communication_section(run_dir)
        if metric_rows:
            lines += [f"### `{run_dir.name}` metrics", "", metrics_table(spec.exp_id, metric_rows), ""]
        if comm:
            lines += [f"### `{run_dir.name}` communication/topology", "", comm, ""]
    lines += [
        "## Interpretation",
        "",
        str(notes.get("interpretation", "Interpretation pending final report validation.")),
        "",
        "## Limitations and anomalies",
        "",
        "- This is a short bounded lab measurement, not model training to convergence.",
        "- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.",
        "- Catalog lifecycle status remains `accepted` until final validation and completed report review.",
        "",
        "## Exam takeaway",
        "",
        str(notes.get("takeaway", "Pending final report validation.")),
        "",
        "## Reproduction",
        "",
        "Use the experiment spec, queue configuration, and immutable image digest "
        "recorded in this report and in `docs/experiment_history/`. Restore "
        "`artifacts/runs/` from the artifact archive before rerunning local "
        "analysis.",
        "",
    ]
    report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_index(specs: list[ExperimentSpec]) -> None:
    rows = []
    for spec in specs:
        runs = run_dirs_for(spec.exp_id)
        notes = EXPERIMENT_NOTES.get(spec.exp_id, {})
        status = "artifact mirror present" if runs else "no local artifacts"
        if spec.exp_id in {"EXP-12", "EXP-14"}:
            status = "completed report present"
        rows.append([
            f"[{spec.exp_id}]({spec.exp_id}.md)",
            spec.title,
            spec.config.get("stack", {}).get("family", ""),
            spec.config.get("hardware", {}).get("provider", ""),
            spec.config.get("hardware", {}).get("compute_profile", ""),
            ", ".join(run.name for run in runs) or "",
            status,
            truncate(str(notes.get("conclusion", "")), 140),
        ])
    text = [
        "# Experiment History For RAG",
        "",
        "This directory is a tracked, text-first summary of the project experiment "
        "history. It is intentionally redundant with `EXPERIMENT_CATALOG.md`, "
        "`PROJECT_DECISIONS.md`, experiment specs, reports, and artifact mirrors "
        "so another workstation can answer questions about experiment goals, "
        "methods, results, and known anomalies without relying on Codex chat "
        "history.",
        "",
        "The files here are derived from tracked experiment specifications plus "
        "the local ignored artifact mirrors under `artifacts/runs/`. Raw logs, "
        "profiler traces, checkpoints, credentials, and model/data caches remain "
        "outside Git.",
        "",
        "## How To Use",
        "",
        "- Start with this README for the table of all experiments.",
        "- Use `EXP-NN.md` for RAG chunks about a single experiment.",
        "- Use `operational_timeline.md` for launch, image, provider, and bug-fix history.",
        "- Restore the compressed artifact archive when exact raw logs or profiler traces are needed.",
        "- Prefer these history files over stale placeholder report text for experiments whose formal reports are still pending.",
        "",
        "## Experiment Index",
        "",
        md_table(
            [
                "ID",
                "Experiment",
                "Framework",
                "Provider",
                "Compute profile",
                "Run IDs",
                "State",
                "Current conclusion",
            ],
            rows,
        ),
        "",
        "## Artifact Mirrors Used",
        "",
        md_table(
            ["Mirror", "Exists", "Files"],
            [
                [
                    rel(root),
                    "yes" if root.exists() else "no",
                    sum(1 for path in root.glob("**/*") if path.is_file()) if root.exists() else 0,
                ]
                for root in ARTIFACT_ROOTS
            ],
        ),
        "",
        "## Regeneration",
        "",
        "After restoring `artifacts/runs/` on another workstation, regenerate this directory with:",
        "",
        "```bash",
        "python3 scripts/build_experiment_history.py",
        "```",
        "",
    ]
    (HISTORY_DIR / "README.md").write_text("\n".join(text).rstrip() + "\n", encoding="utf-8")


def write_timeline() -> None:
    lines = [
        "# Operational Experiment Timeline",
        "",
        "This file records the experiment execution history that is easy to lose in "
        "chat: capacity attempts, image/auth issues, launch bugs, hotfixes, and "
        "which run IDs are the measurements to trust. It is intentionally concise "
        "and points to the per-experiment history files for measurements.",
        "",
    ]
    for title, body in OPERATIONAL_TIMELINE:
        lines += [f"## {title}", "", body, ""]
    lines += [
        "## Qualification And Queue Artifact Directories",
        "",
        md_table(
            ["Kind", "Directory"],
            [["qualification", rel(path)] for path in qualification_dirs()]
            + [["queue", rel(path)] for path in queue_dirs()],
        ),
        "",
    ]
    (HISTORY_DIR / "operational_timeline.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    specs = load_specs()
    for spec in specs:
        write_interim_report(spec)
    for spec in specs:
        write_experiment_doc(spec)
    write_index(specs)
    write_timeline()
    print(f"wrote {len(specs) + 2} markdown files under {rel(HISTORY_DIR)}")


if __name__ == "__main__":
    main()
