from __future__ import annotations

import base64
import unittest

from infra.aws.exp01_launch import (
    DEFAULT_CONFIG,
    build_run_instances_request,
    container_name,
    confirmation_phrase,
    cache_volume_summary,
    image_reference,
    load_config,
    request_summary,
    validate_run_id,
)


A1_CONFIG = DEFAULT_CONFIG.with_name("a1_qualification.yaml")


class Exp01LaunchConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(DEFAULT_CONFIG)
        self.request = build_run_instances_request(self.config, "test-run")

    def test_uses_immutable_ecr_digest_reference(self):
        self.assertEqual(
            image_reference(self.config),
            "037678282394.dkr.ecr.us-west-2.amazonaws.com/"
            "multi-gpu-training-pytorch@"
            "sha256:e17de82324539ff25707ebe267dede8e70c558005c9e9f0f0c6e3dbd7f9f9d8f",
        )

    def test_confirmation_phrase_names_profile_and_lifetime(self):
        self.assertEqual(
            confirmation_phrase(self.config),
            "launch EXP-01 AWS-A2 terminate-after-90m",
        )
        self.assertEqual(
            confirmation_phrase(self.config, hold_open_on_exit=True),
            "launch EXP-01 AWS-A2 stop-after-90m",
        )

    def test_run_instances_request_has_expected_safety_controls(self):
        self.assertEqual(self.request["ImageId"], "ami-04b4c34375925db5f")
        self.assertEqual(self.request["InstanceType"], "g7e.12xlarge")
        self.assertEqual(
            self.request["InstanceInitiatedShutdownBehavior"], "terminate"
        )
        self.assertEqual(
            self.request["MetadataOptions"]["HttpTokens"], "required"
        )
        self.assertEqual(
            self.request["IamInstanceProfile"]["Name"],
            "FinetuningGpuInstanceRole",
        )
        self.assertEqual(
            self.request["SecurityGroupIds"],
            ["sg-0797f3b8520d4efa9"],
        )
        self.assertNotIn("NetworkInterfaces", self.request)
        self.assertTrue(
            self.request["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"]
        )

    def test_cache_volume_is_declared_but_not_root_block_device(self):
        self.assertIn("vol-0746f5d3a6d2cd859", cache_volume_summary(self.config))
        self.assertEqual(self.config["cache_volume"]["size_gib"], 300)
        self.assertEqual(self.config["cache_volume"]["availability_zone"], "us-west-2d")
        self.assertEqual(self.config["cache_volume"]["docker_data_root"], "/mnt/aws-cache/docker")
        self.assertEqual(len(self.request["BlockDeviceMappings"]), 1)

    def test_run_id_and_container_name_are_predictable(self):
        self.assertEqual(validate_run_id("20260714T041113Z"), "20260714T041113Z")
        self.assertEqual(container_name("test-run"), "exp01-test-run")
        with self.assertRaisesRegex(Exception, "unsupported characters"):
            validate_run_id("bad/run")

    def test_user_data_stages_out_and_shuts_down(self):
        user_data = base64.b64decode(self.request["UserData"]).decode("utf-8")
        self.assertIn("aws s3 sync", user_data)
        self.assertIn("docker pull", user_data)
        self.assertIn("collect_exp01.sh", user_data)
        self.assertIn("configure_cache_volume", user_data)
        self.assertIn("CACHE_MOUNT_POINT=\"/mnt/aws-cache\"", user_data)
        self.assertIn("data[\"data-root\"] = os.environ[\"CACHE_DOCKER_DATA_ROOT\"]", user_data)
        self.assertIn("--name \"${CONTAINER_NAME}\"", user_data)
        self.assertIn("shutdown -h now", user_data)
        self.assertIn("exp01-user-data.log", user_data)
        self.assertIn("MAX_LIFETIME_SECONDS=\"5400\"", user_data)

    def test_hold_open_changes_user_data_without_changing_safety_limit(self):
        request = build_run_instances_request(
            self.config,
            "test-run",
            hold_open_on_exit=True,
        )
        user_data = base64.b64decode(request["UserData"]).decode("utf-8")
        self.assertEqual(request["InstanceInitiatedShutdownBehavior"], "stop")
        self.assertIn("HOLD_OPEN_ON_EXIT=\"1\"", user_data)
        self.assertIn("MAX_LIFETIME_SECONDS=\"5400\"", user_data)
        self.assertIn("POST_RUN_INSPECTION_SECONDS=\"900\"", user_data)
        self.assertIn("exp01-safety-shutdown.timer", user_data)
        self.assertIn("OnBootSec=${MAX_LIFETIME_SECONDS}s", user_data)
        self.assertIn("Holding host open for ${POST_RUN_INSPECTION_MINUTES} minutes", user_data)
        self.assertIn("Post-run inspection window finished; stopping instance", user_data)

    def test_summary_exposes_confirmation_but_not_user_data_body(self):
        summary = request_summary(self.config, self.request, "test-run")
        self.assertEqual(summary["run_id"], "test-run")
        self.assertEqual(summary["shutdown_behavior"], "terminate")
        self.assertEqual(summary["post_run_inspection_minutes"], 0)
        self.assertEqual(summary["container_name"], "exp01-test-run")
        self.assertIn("/mnt/aws-cache/docker", summary["cache_volume"])
        self.assertFalse(summary["hold_open_on_exit"])
        self.assertIn("user_data_sha256", summary)
        self.assertNotIn("UserData", summary)

    def test_hold_open_summary_exposes_stop_policy(self):
        request = build_run_instances_request(
            self.config,
            "test-run",
            hold_open_on_exit=True,
        )
        summary = request_summary(
            self.config,
            request,
            "test-run",
            hold_open_on_exit=True,
        )
        self.assertEqual(summary["shutdown_behavior"], "stop")
        self.assertEqual(summary["post_run_inspection_minutes"], 15)
        self.assertEqual(
            summary["confirmation_phrase"],
            "launch EXP-01 AWS-A2 stop-after-90m",
        )

    def test_a1_config_uses_one_gpu_smoke_contract(self):
        config = load_config(A1_CONFIG)
        request = build_run_instances_request(config, "test-run")
        user_data = base64.b64decode(request["UserData"]).decode("utf-8")
        summary = request_summary(config, request, "test-run")

        self.assertEqual(request["InstanceType"], "g7e.2xlarge")
        self.assertEqual(config["compute"]["profile"], "AWS-A1")
        self.assertEqual(config["compute"]["expected_physical_gpus"], 1)
        self.assertEqual(config["safety"]["required_tags"]["Experiment"], "QUAL-A1")
        self.assertEqual(summary["container_name"], "qual-a1-test-run")
        self.assertEqual(summary["cuda_visible_devices"], "0")
        self.assertIn("all_reduce_perf", summary["container_command"])
        self.assertIn("CONTAINER_COMMAND=(", user_data)
        self.assertIn('CUDA_VISIBLE_DEVICES_VALUE="0"', user_data)
        self.assertIn("expected exactly 1 CUDA device", user_data)
        self.assertIn("artifacts/QUAL-A1", config["artifacts"]["durable_uri"])


if __name__ == "__main__":
    unittest.main()
