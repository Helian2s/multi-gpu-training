from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

from infra.aws.a2_queue import (
    build_launch_request,
    host_script,
    launch_confirmation_phrase,
    load_yaml,
    render_plan,
    validate_queue,
)
from infra.aws.exp01_launch import load_config as load_launch_config


class AwsA2QueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = load_yaml(Path("infra/aws/a2_experiment_queue.yaml"))
        self.launch_config = load_launch_config(Path("infra/aws/exp01_qualification.yaml"))

    def queue_with_digest(self) -> dict:
        queue = copy.deepcopy(self.queue)
        queue["image"]["digest"] = (
            "sha256:1111111111111111111111111111111111111111111111111111111111111111"
        )
        return queue

    def test_queue_order_includes_exp08(self) -> None:
        queue = self.queue_with_digest()
        validate_queue(queue)
        plan = render_plan(queue)

        self.assertTrue(plan["image_ready"])
        self.assertFalse(plan["stop_on_failure"])
        self.assertTrue(plan["stop_on_success"])
        self.assertEqual(
            [item["run_unit"] for item in plan["run_units"]],
            [
                "EXP-01-A2",
                "EXP-02-A2V1",
                "EXP-02-A2V2",
                "EXP-07-A2V1",
                "EXP-07-A2V2",
                "EXP-08-A2V2",
                "EXP-09-A2V1",
                "EXP-09-A2V2",
            ],
        )

    def test_generated_host_script_is_valid_bash_with_digest(self) -> None:
        script = host_script(self.queue_with_digest(), self.launch_config, "unit-test")
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
        self.assertIn("EXP-01-A2", script)
        self.assertIn("EXP-08-A2V2", script)
        self.assertIn("EXP-09-A2V2", script)
        self.assertIn("collect_exp01.sh", script)
        self.assertIn("--run-unit", script)
        self.assertIn("run_queue_unit EXP-01-A2 EXP-01 shell 0,1 ''", script)
        self.assertIn("STOP_ON_FAILURE=\"false\"", script)
        self.assertIn("leaving instance running for inspection", script)
        self.assertNotIn("aws-a2-run-units.tsv", script)
        self.assertNotIn("while IFS=$'\\t'", script)

    def test_launch_request_uses_stop_and_fixed_west_2b_subnet(self) -> None:
        queue = self.queue_with_digest()
        script = host_script(queue, self.launch_config, "unit-test")
        request = build_launch_request(queue, self.launch_config, script, "unit-test")

        self.assertEqual(request["InstanceType"], "g7e.12xlarge")
        self.assertEqual(request["InstanceInitiatedShutdownBehavior"], "stop")
        self.assertEqual(request["NetworkInterfaces"][0]["SubnetId"], "subnet-0d50d4374d2149a57")
        self.assertEqual(request["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"], True)
        self.assertIn("UserData", request)

    def test_confirmation_phrase_is_explicit(self) -> None:
        self.assertEqual(
            launch_confirmation_phrase(self.queue),
            "launch AWS-A2-PyTorch AWS-A2 stop-after-300m",
        )


if __name__ == "__main__":
    unittest.main()
