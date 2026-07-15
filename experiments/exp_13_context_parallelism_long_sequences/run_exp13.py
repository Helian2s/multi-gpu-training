#!/usr/bin/env python3
"""EXP-13 runner entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.experiment_runner import runner_main
from common.megatron_executor import execute_megatron_plan


DEFAULT_CONFIG = Path(__file__).with_name("experiment.yaml")


def main() -> int:
    return runner_main(
        experiment="EXP-13",
        default_config=DEFAULT_CONFIG,
        description="Plan or execute EXP-13 context parallel variants.",
        executor=execute_megatron_plan,
    )


if __name__ == "__main__":
    raise SystemExit(main())
