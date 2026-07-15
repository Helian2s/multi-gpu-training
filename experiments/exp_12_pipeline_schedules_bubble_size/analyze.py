#!/usr/bin/env python3
"""Summarize EXP-12 pipeline schedule metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(run_dir: Path) -> list[dict]:
    metrics = run_dir / "metrics" / "variant_results.jsonl"
    if not metrics.is_file():
        raise SystemExit(f"missing metrics file: {metrics}")
    return [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line]


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant_id"]].append(row)
    summary = []
    for variant_id, ranks in grouped.items():
        first = ranks[0]
        peak_mib = max(row["peak_allocated_bytes"] for row in ranks) / 1024 / 1024
        stages = [row["stage_layers"] for row in sorted(ranks, key=lambda item: item["rank"])]
        summary.append(
            {
                "variant_id": variant_id,
                "schedule": first["schedule"],
                "microbatches": first["num_microbatches"],
                "bubble_fraction": first["estimated_pipeline_bubble_fraction"],
                "max_step_ms": first["max_rank_mean_step_ms"],
                "tokens_per_second": first["tokens_per_second"],
                "peak_mib": peak_mib,
                "stages": stages,
            }
        )
    return sorted(summary, key=lambda item: item["variant_id"])


def print_markdown(summary: list[dict]) -> None:
    print("| Variant | Schedule | Microbatches | Bubble estimate | Max step ms | Tokens/s | Peak MiB | Stages |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in summary:
        print(
            "| {variant_id} | {schedule} | {microbatches} | {bubble_fraction:.3f} | "
            "{max_step_ms:.3f} | {tokens_per_second:.0f} | {peak_mib:.1f} | {stages} |".format(
                **row
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print_markdown(summarize(load_rows(args.run_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
