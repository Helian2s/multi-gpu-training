#!/usr/bin/env python3
"""Validate repository contracts that commonly drift during planning edits."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "EXPERIMENT_CATALOG.md"
DECISIONS = ROOT / "PROJECT_DECISIONS.md"


class ValidationError(RuntimeError):
    """Raised when a repository contract is violated."""


def repository_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise ValidationError(f"cannot list active repository files: {detail}")
    return sorted(
        ROOT / raw_path.decode()
        for raw_path in completed.stdout.split(b"\0")
        if raw_path
    )


def validate_yaml() -> int:
    paths = [path for path in repository_files() if path.suffix == ".yaml"]
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        ruby = shutil.which("ruby")
        if ruby is None:
            raise ValidationError(
                "YAML validation requires PyYAML or a bootstrap Ruby installation"
            )
        ruby_program = (
            'require "yaml"; '
            "ARGV.each { |f| YAML.safe_load(File.read(f), "
            "permitted_classes: [], permitted_symbols: [], aliases: false, "
            "filename: f) }"
        )
        completed = subprocess.run(
            [ruby, "-e", ruby_program, *(str(path) for path in paths)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValidationError(f"YAML parsing failed: {detail}")
    else:
        for path in paths:
            try:
                yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError as error:
                relative_path = path.relative_to(ROOT)
                raise ValidationError(
                    f"invalid YAML in {relative_path}: {error}"
                ) from error
    return len(paths)


def validate_markdown_links() -> int:
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    checked = 0
    documents = [path for path in repository_files() if path.suffix == ".md"]
    for document in documents:
        document_text = document.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(document_text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_text = target.split("#", maxsplit=1)[0]
            if not path_text:
                continue
            checked += 1
            resolved = (document.parent / path_text).resolve()
            if not resolved.exists():
                relative_document = document.relative_to(ROOT)
                raise ValidationError(
                    f"broken local Markdown link in {relative_document}: {target}"
                )
    return checked


def validate_no_legacy_experiment_ids() -> int:
    legacy_pattern = re.compile(
        r"\bEXP-\d{3}\b|\bEXP-(?:NNN|XXX)\b|\bexp_(?:NNN|\d{3})\b"
    )
    checked = 0
    candidate_paths = [
        path
        for path in repository_files()
        if (
            path.suffix in {".md", ".py", ".yaml"}
            or path.name in {"AGENTS.md", "Makefile"}
        )
    ]
    for path in sorted(candidate_paths):
        text = path.read_text(encoding="utf-8")
        if path == DECISIONS:
            text = text.split("## Decision log", maxsplit=1)[0]
        checked += 1
        match = legacy_pattern.search(text)
        if match is not None:
            relative_path = path.relative_to(ROOT)
            raise ValidationError(
                f"legacy experiment ID {match.group(0)} in active content: "
                f"{relative_path}"
            )
    return checked


def expand_profile_reference(reference: str) -> set[str]:
    if "/" not in reference:
        return {reference}
    first, *suffixes = reference.split("/")
    match = re.fullmatch(r"(.+?)(\d+)", first)
    if match is None or any(not suffix.isdigit() for suffix in suffixes):
        raise ValidationError(
            f"unsupported combined compute-profile reference: {reference}"
        )
    prefix, first_number = match.groups()
    return {f"{prefix}{number}" for number in (first_number, *suffixes)}


def validate_catalog() -> int:
    text = CATALOG.read_text(encoding="utf-8")
    workload_section = text.split(
        "## Proposed shared workload contract", maxsplit=1
    )[1].split("### Why this model", maxsplit=1)[0]
    workload_statuses = re.findall(
        r"^\| [^|]+ \| [^|]+ \| ([a-z]+) \|$",
        workload_section,
        flags=re.MULTILINE,
    )
    if len(workload_statuses) != 8:
        raise ValidationError(
            "shared workload contract must contain exactly eight decision rows"
        )
    unsupported_workload_statuses = sorted(
        set(workload_statuses) - {"proposed", "accepted"}
    )
    if unsupported_workload_statuses:
        raise ValidationError(
            "shared workload contract has unsupported statuses: "
            f"{unsupported_workload_statuses}"
        )

    summary_rows = re.findall(
        r"^\| (EXP-\d{2}) \|(.+)$", text, flags=re.MULTILINE
    )
    summary_ids = [experiment_id for experiment_id, _ in summary_rows]
    detail_ids = re.findall(r"^### (EXP-\d{2}):", text, flags=re.MULTILINE)
    expected_ids = [
        f"EXP-{number:02d}" for number in range(1, len(summary_ids) + 1)
    ]

    if summary_ids != expected_ids:
        raise ValidationError(
            "catalog summary IDs are not continuous: "
            f"expected {expected_ids}, got {summary_ids}"
        )
    if detail_ids != expected_ids:
        raise ValidationError(
            "catalog detail IDs do not match the summary: "
            f"expected {expected_ids}, got {detail_ids}"
        )

    defined_profiles = set(
        re.findall(
            r"^\| `((?:AWS|RUNPOD)-[^`]+)` \|", text, flags=re.MULTILINE
        )
    )
    detail_sections: dict[str, str] = {}
    detail_matches = list(
        re.finditer(r"^### (EXP-\d{2}): (.+)$", text, flags=re.MULTILINE)
    )
    detail_titles = {
        match.group(1): match.group(2).strip() for match in detail_matches
    }
    for index, match in enumerate(detail_matches):
        section_end = (
            detail_matches[index + 1].start()
            if index + 1 < len(detail_matches)
            else len(text)
        )
        detail_sections[match.group(1)] = text[match.end() : section_end]

    referenced_profiles: set[str] = set()
    for experiment_id, remainder in summary_rows:
        cells = [cell.strip() for cell in remainder.strip(" |").split("|")]
        if len(cells) < 4:
            raise ValidationError(f"malformed catalog summary row for {experiment_id}")
        summary_title = cells[0]
        if detail_titles[experiment_id] != summary_title:
            raise ValidationError(
                f"{experiment_id} title mismatch: summary has {summary_title!r}, "
                f"detail has {detail_titles[experiment_id]!r}"
            )
        status = cells[-1].lower()
        allowed_statuses = {"proposed", "accepted", "deferred", "completed"}
        if status not in allowed_statuses:
            raise ValidationError(
                f"{experiment_id} has unsupported status {status!r}; "
                f"expected one of {sorted(allowed_statuses)}"
            )
        provider_cell = cells[3]
        providers = {name for name in ("AWS", "Runpod") if name in provider_cell}
        if len(providers) != 1:
            raise ValidationError(
                f"{experiment_id} must reference exactly one provider, "
                f"got: {provider_cell}"
            )
        expected_provider = "AWS" if int(experiment_id[-2:]) <= 9 else "Runpod"
        if providers != {expected_provider}:
            raise ValidationError(
                f"{experiment_id} must be in the {expected_provider} phase, "
                f"got: {provider_cell}"
            )
        references = re.findall(
            r"`((?:AWS|RUNPOD)-[^`]+)`", provider_cell
        )
        if not references:
            raise ValidationError(
                f"{experiment_id} has no compute-profile reference"
            )
        summary_profiles: set[str] = set()
        for reference in references:
            summary_profiles.update(expand_profile_reference(reference))
        referenced_profiles.update(summary_profiles)

        section = detail_sections[experiment_id]
        planned_compute = re.search(
            r"\*\*Planned compute:\*\*(.*?)(?:\n\n)",
            section,
            flags=re.DOTALL,
        )
        if planned_compute is None:
            raise ValidationError(
                f"{experiment_id} detail has no planned-compute paragraph"
            )
        detail_references = re.findall(
            r"`((?:AWS|RUNPOD)-[^`]+)`", planned_compute.group(1)
        )
        detail_profiles: set[str] = set()
        for reference in detail_references:
            detail_profiles.update(expand_profile_reference(reference))
        if detail_profiles != summary_profiles:
            raise ValidationError(
                f"{experiment_id} compute-profile mismatch: summary has "
                f"{sorted(summary_profiles)}, detail has {sorted(detail_profiles)}"
            )

    unknown_profiles = sorted(referenced_profiles - defined_profiles)
    if unknown_profiles:
        raise ValidationError(
            "catalog references undefined compute profiles: "
            f"{', '.join(unknown_profiles)}"
        )
    return len(summary_ids)


def main() -> int:
    try:
        yaml_count = validate_yaml()
        link_count = validate_markdown_links()
        active_file_count = validate_no_legacy_experiment_ids()
        experiment_count = validate_catalog()
    except ValidationError as error:
        print(f"repository validation failed: {error}", file=sys.stderr)
        return 1

    print(
        "repository validation passed: "
        f"{yaml_count} YAML files, {link_count} local links, "
        f"{active_file_count} active files without legacy IDs, "
        f"{experiment_count} catalog experiments"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
