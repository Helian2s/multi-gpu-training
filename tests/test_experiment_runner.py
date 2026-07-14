from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from common.experiment_runner import (
    build_plan,
    configured_variants,
    load_yaml,
    resolve_output_root,
    runner_main,
    select_variants,
    write_plan,
)


ROOT = Path(__file__).resolve().parents[1]
EXP02_CONFIG = ROOT / "experiments" / "exp_02_mixed_precision_tensor_cores" / "experiment.yaml"
EXP08_CONFIG = ROOT / "experiments" / "exp_08_fsdp_sharding_zero_memory_tradeoffs" / "experiment.yaml"


class ExperimentRunnerContractTest(unittest.TestCase):
    def test_selects_run_unit_variants(self) -> None:
        config = load_yaml(EXP02_CONFIG)
        variants = configured_variants(config)
        selected = select_variants(variants, run_units=["EXP-02-A2V2"])

        self.assertEqual(
            [variant["id"] for variant in selected],
            [
                "EXP-02-A2V2-ddp-fp32-reference",
                "EXP-02-A2V2-ddp-bf16",
                "EXP-02-A2V2-ddp-fp16",
            ],
        )

    def test_rejects_unknown_variant(self) -> None:
        config = load_yaml(EXP02_CONFIG)
        with self.assertRaisesRegex(Exception, "unknown variant"):
            select_variants(configured_variants(config), variant_ids=["missing"])

    def test_builds_dry_run_plan_without_image_digest(self) -> None:
        config = load_yaml(EXP08_CONFIG)
        selected = select_variants(configured_variants(config), run_units=["EXP-08-A2V2"])
        output_root = resolve_output_root(EXP08_CONFIG, config, None)
        plan = build_plan(
            config_path=EXP08_CONFIG,
            config=config,
            selected=selected,
            run_id="unit-test",
            output_root=output_root,
            mode="dry-run",
        )

        self.assertEqual(plan["experiment_id"], "EXP-08")
        self.assertEqual(plan["run_units"], ["EXP-08-A2V2"])
        self.assertFalse(plan["image_ready"])
        self.assertEqual(plan["variant_count"], 4)

    def test_write_plan_creates_manifest(self) -> None:
        config = load_yaml(EXP08_CONFIG)
        selected = select_variants(configured_variants(config), variant_ids=["EXP-08-A2V2-ddp-replicated-baseline"])
        with tempfile.TemporaryDirectory() as temporary_directory:
            plan = build_plan(
                config_path=EXP08_CONFIG,
                config=config,
                selected=selected,
                run_id="unit-test",
                output_root=Path(temporary_directory),
                mode="dry-run",
            )
            path = write_plan(plan)
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["run_id"], "unit-test")
        self.assertEqual(loaded["variants"][0]["id"], "EXP-08-A2V2-ddp-replicated-baseline")

    def test_runner_execute_fails_closed_without_image_digest(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                runner_main(
                    experiment="EXP-08",
                    default_config=EXP08_CONFIG,
                    description="test runner",
                    argv=["--execute", "--variant", "EXP-08-A2V2-ddp-replicated-baseline"],
                )

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("immutable digest", stderr.getvalue())

    def test_runner_dry_run_outputs_json(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = runner_main(
                experiment="EXP-08",
                default_config=EXP08_CONFIG,
                description="test runner",
                argv=["--dry-run", "--variant", "EXP-08-A2V2-ddp-replicated-baseline"],
            )

        self.assertEqual(result, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["experiment_id"], "EXP-08")
        self.assertEqual(payload["variant_count"], 1)


if __name__ == "__main__":
    unittest.main()

