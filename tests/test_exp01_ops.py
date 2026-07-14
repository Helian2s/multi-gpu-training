from __future__ import annotations

import unittest

from infra.aws.exp01_ops import container_name, instance_run_id, tags_to_dict


class Exp01OpsTest(unittest.TestCase):
    def test_tag_helpers_extract_run_id(self):
        instance = {
            "Tags": [
                {"Key": "Project", "Value": "NCP-GENL"},
                {"Key": "RunId", "Value": "20260714T041113Z"},
            ]
        }

        self.assertEqual(tags_to_dict(instance["Tags"])["Project"], "NCP-GENL")
        self.assertEqual(instance_run_id(instance), "20260714T041113Z")
        self.assertEqual(instance_run_id(instance, "override"), "override")

    def test_container_name_matches_launcher_contract(self):
        self.assertEqual(container_name("20260714T041113Z"), "exp01-20260714T041113Z")
        self.assertEqual(
            container_name("20260714T041113Z", {"container": {"name_prefix": "qual-a1"}}),
            "qual-a1-20260714T041113Z",
        )


if __name__ == "__main__":
    unittest.main()
