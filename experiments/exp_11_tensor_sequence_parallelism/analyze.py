#!/usr/bin/env python3
"""Experiment-specific analysis entry point.

Replace the template error only after the expected metrics and decision rules
have been declared. Keep raw inputs immutable and write derived artifacts to the
run's analysis directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_directory",
        type=Path,
        help="Path to artifacts/runs/<experiment-id>/<run-id>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run_directory.is_dir():
        raise SystemExit(f"run directory does not exist: {args.run_directory}")
    raise SystemExit(
        "analysis is not implemented; define the experiment's metrics and "
        "decision rules before replacing this template"
    )


if __name__ == "__main__":
    main()
