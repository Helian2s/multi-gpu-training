from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

from infra.aws.a1_queue import host_script, load_yaml, render_plan, validate_queue
from infra.aws.exp01_launch import load_config as load_launch_config


class AwsA1QueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = load_yaml(Path("infra/aws/a1_experiment_queue.yaml"))
        self.launch_config = load_launch_config(Path("infra/aws/a1_qualification.yaml"))

    def test_queue_order_is_exp03_through_exp06(self) -> None:
        validate_queue(self.queue)
        plan = render_plan(self.queue)

        self.assertTrue(plan["image_ready"])
        self.assertIn("@sha256:e12af417e7e905f30182122a95d73610e3acc9cb41829093d0265dfd6cca4225", plan["image"])
        self.assertEqual(
            [item["run_unit"] for item in plan["run_units"]],
            ["EXP-03-A1", "EXP-04-A1", "EXP-05-A1", "EXP-06-A1"],
        )

    def test_generated_host_script_is_valid_bash_with_digest(self) -> None:
        queue = copy.deepcopy(self.queue)
        queue["image"]["digest"] = (
            "sha256:1111111111111111111111111111111111111111111111111111111111111111"
        )
        script = host_script(queue, self.launch_config, "unit-test")
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".sh") as handle:
            handle.write(script)
            handle.flush()
            completed = subprocess.run(
                ["bash", "-n", handle.name],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("here-document", completed.stderr)
        self.assertIn("EXP-03-A1", script)
        self.assertIn("MULTI_GPU_TRAINING_IMAGE_REF", script)


if __name__ == "__main__":
    unittest.main()
