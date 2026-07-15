from __future__ import annotations

import copy
import subprocess
import tempfile
import unittest
from pathlib import Path

from infra.runpod.runpod_queue import (
    RunpodQueueError,
    container_script,
    load_yaml,
    pod_create_command,
    render_plan,
    validate_queue,
)


class RunpodQueueTest(unittest.TestCase):
    def setUp(self) -> None:
        self.a2_queue = load_yaml(Path("infra/runpod/a2_megatron_queue.yaml"))
        self.a4_queue = load_yaml(Path("infra/runpod/a4_megatron_queue.yaml"))

    def launch_ready_queue(self, queue: dict) -> dict:
        result = copy.deepcopy(queue)
        result["image"]["tag"] = "unit-test"
        result["image"]["digest"] = (
            "sha256:2222222222222222222222222222222222222222222222222222222222222222"
        )
        result["runpod"]["registry_auth_id"] = "registry-auth-id"
        result["runpod"]["network_volume_id"] = ""
        result["runpod"]["pod_volume_gb"] = 50
        return result

    def test_a2_queue_order(self) -> None:
        validate_queue(self.a2_queue)
        plan = render_plan(self.a2_queue)

        self.assertEqual(plan["queue"], "RUNPOD-A2-Megatron")
        self.assertTrue(plan["launch_ready"])
        self.assertEqual(plan["missing_for_launch"], [])
        self.assertEqual(
            [item["run_unit"] for item in plan["run_units"]],
            [
                "QUAL-RUNPOD-A2-Megatron",
                "EXP-10-RUNPOD-A2-Megatron",
                "EXP-11-RUNPOD-A2-Megatron-V1",
                "EXP-11-RUNPOD-A2-Megatron-V2",
                "EXP-13-RUNPOD-A2-Megatron-V1",
                "EXP-13-RUNPOD-A2-Megatron-V2",
            ],
        )

    def test_a4_queue_order(self) -> None:
        validate_queue(self.a4_queue)
        plan = render_plan(self.a4_queue)

        self.assertEqual(plan["queue"], "RUNPOD-A4-Megatron")
        self.assertEqual(
            [item["run_unit"] for item in plan["run_units"]],
            ["QUAL-RUNPOD-A4-Megatron", "EXP-14-RUNPOD-A4-Megatron"],
        )

    def test_generated_container_scripts_are_valid_bash(self) -> None:
        for queue in (self.a2_queue, self.a4_queue):
            script = container_script(queue, "unit-test")
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

    def test_generated_container_script_exports_worker_context(self) -> None:
        script = container_script(self.a2_queue, "unit-test")

        self.assertIn("export RUN_ID\n", script)
        self.assertIn("export IMAGE_REF\n", script)
        self.assertIn("export ARTIFACT_ROOT\n", script)
        self.assertIn('export MULTI_GPU_TRAINING_IMAGE_REF="${IMAGE_REF}"', script)
        self.assertIn('"--output-root" "${ARTIFACT_ROOT}"', script)
        self.assertIn('"--image-ref" "${IMAGE_REF}"', script)

    def test_pod_create_command_requires_launch_ready_state(self) -> None:
        queue = copy.deepcopy(self.a2_queue)
        queue["image"]["digest"] = ""
        queue["runpod"]["registry_auth_id"] = ""
        queue["runpod"]["network_volume_id"] = ""
        queue["runpod"]["pod_volume_gb"] = 0
        with self.assertRaisesRegex(
            RunpodQueueError,
            "image.digest, runpod.registry_auth_id, runpod.network_volume_id or runpod.pod_volume_gb",
        ):
            pod_create_command(queue, "unit-test")

    def test_pod_create_command_renders_guarded_command(self) -> None:
        queue = self.launch_ready_queue(self.a2_queue)
        command = pod_create_command(queue, "unit-test")

        self.assertIn("runpodctl pod create", command)
        self.assertIn("--terminate-after", command)
        self.assertRegex(command, r"--terminate-after [0-9]{4}-[0-9]{2}-[0-9]{2}T")
        self.assertNotIn("${TERMINATE_AFTER}", command)
        self.assertIn("ghcr.io/helian2s/multi-gpu-training-nemo@sha256:", command)
        self.assertIn("registry-auth-id", command)
        self.assertIn("--volume-in-gb 50", command)
        self.assertIn("--ports 22/tcp", command)
        self.assertNotIn("--network-volume-id", command)
        self.assertNotIn("--docker-args", command)


if __name__ == "__main__":
    unittest.main()
