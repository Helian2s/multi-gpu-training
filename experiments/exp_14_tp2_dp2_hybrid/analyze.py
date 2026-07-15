#!/usr/bin/env python3
"""Summarize EXP-14 hybrid TP/DP metrics."""

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
        collective_mib = max(row["estimated_collective_bytes"] for row in ranks) / 1024 / 1024
        summary.append(
            {
                "variant_id": variant_id,
                "layout": first["layout"],
                "tensor_parallel_size": first["tensor_parallel_size"],
                "data_parallel_size": first["data_parallel_size"],
                "ranks": [row["rank"] for row in sorted(ranks, key=lambda item: item["rank"])],
                "max_step_ms": first["max_rank_mean_step_ms"],
                "tokens_per_second": first["tokens_per_second"],
                "peak_mib": peak_mib,
                "collective_mib": collective_mib,
            }
        )
    order = {"dp4": 0, "tp4": 1, "tp2_dp2": 2}
    return sorted(summary, key=lambda item: order[item["layout"]])


def print_markdown(summary: list[dict]) -> None:
    print("| Variant | Layout | TP | DP | Ranks | Max step ms | Tokens/s | Peak MiB | Collective MiB/rank |")
    print("| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
    for row in summary:
        print(
            "| {variant_id} | {layout} | {tensor_parallel_size} | {data_parallel_size} | "
            "{ranks} | {max_step_ms:.3f} | {tokens_per_second:.0f} | "
            "{peak_mib:.1f} | {collective_mib:.1f} |".format(**row)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print_markdown(summarize(load_rows(args.run_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
