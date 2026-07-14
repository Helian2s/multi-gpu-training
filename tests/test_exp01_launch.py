from __future__ import annotations

import base64
import unittest

from infra.aws.exp01_launch import (
    DEFAULT_CONFIG,
    build_run_instances_request,
    container_name,
    confirmation_phrase,
    image_reference,
    load_config,
    request_summary,
    validate_run_id,
)


class Exp01LaunchConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(DEFAULT_CONFIG)
        self.request = build_run_instances_request(self.config, "test-run")

    def test_uses_immutable_ecr_digest_reference(self):
        self.assertEqual(
            image_reference(self.config),
            "037678282394.dkr.ecr.us-west-2.amazonaws.com/"
            "multi-gpu-training-pytorch@"
            "sha256:c36c871dcd7e1894f6666c81280e8416c556b44d50e9b4ff5247756472dff59c",
        )

    def test_confirmation_phrase_names_profile_and_lifetime(self):
        self.assertEqual(
            confirmation_phrase(self.config),
            "launch EXP-01 AWS-G7E-2 terminate-after-90m",
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
            self.request["NetworkInterfaces"][0]["Groups"],
            ["sg-0797f3b8520d4efa9"],
        )
        self.assertTrue(
            self.request["NetworkInterfaces"][0]["AssociatePublicIpAddress"]
        )
        self.assertTrue(
            self.request["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"]
        )

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
        self.assertIn("HOLD_OPEN_ON_EXIT=\"1\"", user_data)
        self.assertIn("MAX_LIFETIME_SECONDS=\"5400\"", user_data)

    def test_summary_exposes_confirmation_but_not_user_data_body(self):
        summary = request_summary(self.config, self.request, "test-run")
        self.assertEqual(summary["run_id"], "test-run")
        self.assertEqual(summary["shutdown_behavior"], "terminate")
        self.assertEqual(summary["container_name"], "exp01-test-run")
        self.assertFalse(summary["hold_open_on_exit"])
        self.assertIn("user_data_sha256", summary)
        self.assertNotIn("UserData", summary)


if __name__ == "__main__":
    unittest.main()
