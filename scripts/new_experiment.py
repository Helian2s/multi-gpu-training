#!/usr/bin/env python3
"""Create an experiment directory from the canonical template."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DIRECTORY = REPOSITORY_ROOT / "experiments" / "_template"
CATALOG_PATH = REPOSITORY_ROOT / "EXPERIMENT_CATALOG.md"
EXPERIMENT_ID_PATTERN = re.compile(r"EXP-(\d{2})")
SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create experiments/exp_NN_<slug> from experiments/_template."
    )
    parser.add_argument("experiment_id", help="Canonical ID such as EXP-01")
    parser.add_argument("slug", help="Lowercase underscore-separated directory slug")
    parser.add_argument("title", help="Human-readable experiment title")
    return parser.parse_args()


def validated_destination(experiment_id: str, slug: str) -> Path:
    id_match = EXPERIMENT_ID_PATTERN.fullmatch(experiment_id)
    if id_match is None:
        raise SystemExit("experiment_id must match EXP-NN")
    if SLUG_PATTERN.fullmatch(slug) is None:
        raise SystemExit("slug must contain lowercase letters, digits, and underscores")
    return REPOSITORY_ROOT / "experiments" / f"exp_{id_match.group(1)}_{slug}"


def require_accepted_catalog_entry(experiment_id: str) -> None:
    for line in CATALOG_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith(f"| {experiment_id} |"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        status = cells[-1].lower()
        if status != "accepted":
            raise SystemExit(
                f"{experiment_id} is not accepted in EXPERIMENT_CATALOG.md "
                f"(Status={cells[-1]})"
            )
        return
    raise SystemExit(f"{experiment_id} was not found in EXPERIMENT_CATALOG.md")


def replace_placeholders(directory: Path, experiment_id: str, slug: str, title: str) -> None:
    replacements = {
        "EXP-XX": experiment_id,
        "experiment_slug": slug,
        "Experiment title": title,
    }
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        for old, new in replacements.items():
            content = content.replace(old, new)
        path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    destination = validated_destination(args.experiment_id, args.slug)
    require_accepted_catalog_entry(args.experiment_id)
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")
    shutil.copytree(
        TEMPLATE_DIRECTORY,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    replace_placeholders(destination, args.experiment_id, args.slug, args.title)
    print(destination.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
