from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from infra.aws.a2_queue import (
    build_launch_request,
    host_script,
    launch_confirmation_phrase,
    load_yaml,
    QueueError,
    real_launch,
    render_plan,
    validate_queue,
)
from infra.aws.exp01_launch import load_config as load_launch_config


class AwsA2MegatronQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.queue = load_yaml(Path("infra/aws/a2_megatron_queue.yaml"))
        self.launch_config = load_launch_config(Path("infra/aws/exp01_qualification.yaml"))

    def queue_with_digest(self) -> dict:
        queue = copy.deepcopy(self.queue)
        queue["image"]["tag"] = "unit-test"
        queue["image"]["digest"] = (
            "sha256:2222222222222222222222222222222222222222222222222222222222222222"
        )
        return queue

    def test_queue_order_is_exp12_only(self) -> None:
        queue = self.queue_with_digest()
        validate_queue(queue)
        plan = render_plan(queue)

        self.assertTrue(plan["image_ready"])
        self.assertEqual(plan["queue"], "AWS-A2-Megatron")
        self.assertFalse(plan["stop_on_failure"])
        self.assertEqual(
            [item["run_unit"] for item in plan["run_units"]],
            ["QUAL-A2", "EXP-12-A2V1", "EXP-12-A2V2"],
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
        self.assertIn("AWS-A2-Megatron", script)
        self.assertIn("aws-a2-megatron-queue.log", script)
        self.assertIn("QUAL-A2", script)
        self.assertIn("aws_gpu_smoke.py", script)
        self.assertIn("EXP-12-A2V1", script)
        self.assertIn("EXP-12-A2V2", script)
        self.assertIn("run_exp12.py", script)
        self.assertIn("STOP_ON_FAILURE=\"false\"", script)

    def test_launch_request_reuses_a2_host_profile(self) -> None:
        queue = self.queue_with_digest()
        script = host_script(queue, self.launch_config, "unit-test")
        request = build_launch_request(queue, self.launch_config, script, "unit-test")

        self.assertEqual(request["InstanceType"], "g7e.12xlarge")
        self.assertEqual(request["InstanceInitiatedShutdownBehavior"], "stop")
        if self.launch_config["network"].get("auto_select_subnet"):
            self.assertNotIn("NetworkInterfaces", request)
            self.assertEqual(
                request["SecurityGroupIds"],
                self.launch_config["network"]["security_group_ids"],
            )
        else:
            self.assertEqual(
                request["NetworkInterfaces"][0]["SubnetId"],
                self.launch_config["network"]["subnet_ids"][0],
            )

    def test_confirmation_phrase_is_explicit(self) -> None:
        self.assertEqual(
            launch_confirmation_phrase(self.queue),
            "launch AWS-A2-Megatron AWS-A2 stop-after-300m",
        )

    def test_launch_attachment_failure_honors_inspection_policy(self) -> None:
        queue = self.queue_with_digest()
        request = {"unused": "request"}
        run_instances = {"Instances": [{"InstanceId": "i-0123456789abcdef0"}]}

        with (
            mock.patch("infra.aws.a2_queue.validate_cache_volume_for_launch"),
            mock.patch("infra.aws.a2_queue.run_aws") as run_aws,
            mock.patch("infra.aws.a2_queue.wait_instance_running"),
            mock.patch("infra.aws.a2_queue.attach_cache_volume") as attach_cache_volume,
            mock.patch("infra.aws.a2_queue.run_aws_command") as run_aws_command,
        ):
            run_aws.return_value = SimpleNamespace(
                returncode=0,
                stdout=json.dumps(run_instances),
                stderr="",
            )
            attach_cache_volume.side_effect = RuntimeError("attachment failed")

            with self.assertRaisesRegex(QueueError, "left running for inspection"):
                real_launch(
                    queue,
                    self.launch_config,
                    request,
                    "launch AWS-A2-Megatron AWS-A2 stop-after-300m",
                )

        run_aws_command.assert_not_called()


if __name__ == "__main__":
    unittest.main()
