#!/usr/bin/env python3
"""Read-only AWS preflight for EXP-01.

This script intentionally performs no launch, stop, terminate, delete, or
publish operation. It checks whether the requested AWS profile, quota, EC2
instance type, ECR repository/image, EC2 pull role, S3 artifact bucket, and
budget envelope are coherent before a later explicit launch step is added.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPOSITORY_ROOT / "infra" / "aws" / "exp01_qualification.yaml"
G_VT_QUOTA_CODE = "L-DB2E81BA"
EC2_SERVICE_CODE = "ec2"
PRICING_REGION = "us-east-1"
PRICE_LOCATION = "US West (Oregon)"


@dataclass
class Check:
    name: str
    status: str
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise SystemExit(f"configuration is not a mapping: {path}")
    return data


def run_aws(
    profile: str,
    region: str,
    *args: str,
    allow_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = ["aws", *args, "--profile", profile, "--region", region, "--output", "json"]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0 and not allow_failure:
        raise SystemExit(
            f"command failed: {' '.join(command)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return result


def json_aws(profile: str, region: str, *args: str, allow_failure: bool = False) -> Any:
    result = run_aws(profile, region, *args, allow_failure=allow_failure)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout or "{}")


def ok(name: str, detail: str) -> Check:
    return Check(name, "ok", detail)


def warn(name: str, detail: str) -> Check:
    return Check(name, "warning", detail)


def fail(name: str, detail: str) -> Check:
    return Check(name, "fail", detail)


def first_price_usd(profile: str, instance_type: str) -> float:
    result = json_aws(
        profile,
        PRICING_REGION,
        "pricing",
        "get-products",
        "--service-code",
        "AmazonEC2",
        "--filters",
        f"Type=TERM_MATCH,Field=location,Value={PRICE_LOCATION}",
        f"Type=TERM_MATCH,Field=instanceType,Value={instance_type}",
        "Type=TERM_MATCH,Field=operatingSystem,Value=Linux",
        "Type=TERM_MATCH,Field=tenancy,Value=Shared",
        "Type=TERM_MATCH,Field=preInstalledSw,Value=NA",
        "Type=TERM_MATCH,Field=capacitystatus,Value=Used",
    )
    raw = result["PriceList"][0]
    product = json.loads(raw) if isinstance(raw, str) else raw
    for term in product["terms"]["OnDemand"].values():
        for dimension in term["priceDimensions"].values():
            if dimension.get("unit") == "Hrs":
                return float(dimension["pricePerUnit"]["USD"])
    raise SystemExit(f"no hourly on-demand price found for {instance_type}")


def active_g_vt_vcpus(profile: str, region: str) -> int:
    data = json_aws(
        profile,
        region,
        "ec2",
        "describe-instances",
        "--filters",
        "Name=instance-state-name,Values=pending,running,stopping,stopped",
    )
    instance_types = sorted(
        {
            instance["InstanceType"]
            for reservation in data.get("Reservations", [])
            for instance in reservation.get("Instances", [])
            if instance.get("InstanceType", "").startswith(("g", "vt"))
        }
    )
    if not instance_types:
        return 0
    type_data = json_aws(
        profile,
        region,
        "ec2",
        "describe-instance-types",
        "--instance-types",
        *instance_types,
    )
    vcpus_by_type = {
        item["InstanceType"]: item["VCpuInfo"]["DefaultVCpus"]
        for item in type_data.get("InstanceTypes", [])
    }
    return sum(
        vcpus_by_type.get(instance["InstanceType"], 0)
        for reservation in data.get("Reservations", [])
        for instance in reservation.get("Instances", [])
        if instance.get("InstanceType", "").startswith(("g", "vt"))
    )


def check_identity(config: dict[str, Any]) -> Check:
    aws_cfg = config["aws"]
    data = json_aws(aws_cfg["profile"], aws_cfg["region"], "sts", "get-caller-identity")
    account = data.get("Account")
    if account != aws_cfg["account_id"]:
        return fail("aws identity", f"expected account {aws_cfg['account_id']}, got {account}")
    return ok("aws identity", f"account {account}, arn {data.get('Arn')}")


def check_quota(config: dict[str, Any]) -> Check:
    aws_cfg = config["aws"]
    compute = config["compute"]
    quota = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "service-quotas",
        "get-service-quota",
        "--service-code",
        EC2_SERVICE_CODE,
        "--quota-code",
        G_VT_QUOTA_CODE,
    )["Quota"]["Value"]
    active = active_g_vt_vcpus(aws_cfg["profile"], aws_cfg["region"])
    required = compute["expected_vcpus"]
    if active + required > quota:
        return fail("g/vt quota", f"active {active} + required {required} exceeds quota {quota}")
    return ok("g/vt quota", f"quota {quota:.0f} vCPUs; active {active}; required {required}")


def check_instance_type(config: dict[str, Any]) -> list[Check]:
    aws_cfg = config["aws"]
    compute = config["compute"]
    checks: list[Check] = []
    data = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "ec2",
        "describe-instance-types",
        "--instance-types",
        compute["instance_type"],
    )
    item = data["InstanceTypes"][0]
    gpu = item["GpuInfo"]["Gpus"][0]
    vcpus = item["VCpuInfo"]["DefaultVCpus"]
    gpu_count = gpu["Count"]
    gpu_name = gpu["Name"]
    if vcpus == compute["expected_vcpus"] and gpu_count == compute["expected_physical_gpus"]:
        checks.append(ok("instance shape", f"{compute['instance_type']}: {vcpus} vCPUs, {gpu_count} x {gpu_name}"))
    else:
        checks.append(fail("instance shape", f"unexpected shape: {vcpus} vCPUs, {gpu_count} x {gpu_name}"))

    offerings = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "ec2",
        "describe-instance-type-offerings",
        "--location-type",
        "availability-zone",
        "--filters",
        f"Name=instance-type,Values={compute['instance_type']}",
    )
    azs = sorted(item["Location"] for item in offerings.get("InstanceTypeOfferings", []))
    if azs:
        checks.append(ok("instance offerings", f"{compute['instance_type']} offered in {', '.join(azs)}"))
    else:
        checks.append(fail("instance offerings", f"{compute['instance_type']} has no AZ offerings"))
    return checks


def check_price(config: dict[str, Any]) -> Check:
    aws_cfg = config["aws"]
    compute = config["compute"]
    budget = config["budget"]
    price = first_price_usd(aws_cfg["profile"], compute["instance_type"])
    projected = price * float(budget["maximum_instance_hours"])
    limit = float(budget["maximum_provider_cost_usd"])
    if projected > limit:
        return fail("budget", f"{projected:.2f} USD projected exceeds {limit:.2f} USD limit")
    return ok("budget", f"{price:.5f} USD/hr x {budget['maximum_instance_hours']} hr = {projected:.2f} USD <= {limit:.2f} USD")


def check_host_ami(config: dict[str, Any]) -> Check:
    aws_cfg = config["aws"]
    host = config["host"]
    image_data = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "ec2",
        "describe-images",
        "--image-ids",
        host["ami_id"],
        allow_failure=True,
    )
    if image_data is None or not image_data.get("Images"):
        return fail("host ami", f"{host['ami_id']} was not found")
    image = image_data["Images"][0]
    description = image.get("Description", "")
    name = image.get("Name", "")
    if image.get("State") != "available":
        return fail("host ami", f"{host['ami_id']} state={image.get('State')}")
    if image.get("Architecture") != "x86_64":
        return fail("host ami", f"{host['ami_id']} architecture={image.get('Architecture')}")
    if image.get("RootDeviceName") != host["root_device_name"]:
        return fail("host ami", f"{host['ami_id']} root device={image.get('RootDeviceName')}")
    if "G7e" not in description:
        return fail("host ami", f"{host['ami_id']} description does not list G7e support")
    if name != host["ami_name"]:
        return warn("host ami", f"{host['ami_id']} name changed from {host['ami_name']!r} to {name!r}")
    return ok("host ami", f"{host['ami_id']} {name}")


def check_network(config: dict[str, Any]) -> list[Check]:
    aws_cfg = config["aws"]
    network = config["network"]
    checks: list[Check] = []

    subnet_data = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "ec2",
        "describe-subnets",
        "--subnet-ids",
        *network["subnet_ids"],
    )
    subnets = subnet_data.get("Subnets", [])
    subnet_ids = {subnet["SubnetId"] for subnet in subnets}
    missing_subnets = sorted(set(network["subnet_ids"]) - subnet_ids)
    if missing_subnets:
        checks.append(fail("network subnets", f"missing {missing_subnets}"))
    else:
        azs = sorted(subnet["AvailabilityZone"] for subnet in subnets)
        public_flags = {subnet["SubnetId"]: subnet.get("MapPublicIpOnLaunch") for subnet in subnets}
        wrong_vpc = sorted(
            subnet["SubnetId"]
            for subnet in subnets
            if subnet.get("VpcId") != network["vpc_id"]
        )
        if wrong_vpc:
            checks.append(fail("network subnets", f"subnets outside {network['vpc_id']}: {wrong_vpc}"))
        elif not all(public_flags.values()):
            checks.append(warn("network subnets", f"not all subnets map public IP on launch: {public_flags}"))
        else:
            checks.append(ok("network subnets", f"{len(subnets)} default public subnets in {', '.join(azs)}"))

    sg_data = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "ec2",
        "describe-security-groups",
        "--group-ids",
        *network["security_group_ids"],
    )
    security_groups = sg_data.get("SecurityGroups", [])
    ingress = [
        permission
        for group in security_groups
        for permission in group.get("IpPermissions", [])
    ]
    wrong_sg_vpc = sorted(
        group["GroupId"]
        for group in security_groups
        if group.get("VpcId") != network["vpc_id"]
    )
    if wrong_sg_vpc:
        checks.append(fail("network security groups", f"security groups outside {network['vpc_id']}: {wrong_sg_vpc}"))
    elif ingress:
        checks.append(fail("network security groups", "expected no ingress rules"))
    else:
        checks.append(ok("network security groups", f"{', '.join(network['security_group_ids'])} has no ingress"))

    if network.get("ssh_key_name") is None:
        checks.append(ok("ssh access", "no SSH key configured; SSM-only access"))
    else:
        checks.append(warn("ssh access", f"SSH key configured: {network['ssh_key_name']}"))
    return checks


def check_ecr(config: dict[str, Any]) -> list[Check]:
    aws_cfg = config["aws"]
    image = config["image"]
    checks: list[Check] = []
    repo = image["repository"]
    data = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "ecr",
        "describe-repositories",
        "--repository-names",
        repo,
    )
    uri = data["repositories"][0]["repositoryUri"]
    checks.append(ok("ecr repository", uri))

    tag = image.get("tag")
    digest = image.get("digest")
    if not tag and not digest:
        checks.append(warn("ecr image", "no EXP-01 image tag/digest recorded yet"))
        return checks
    args = ["ecr", "describe-images", "--repository-name", repo, "--image-ids"]
    if digest:
        args.append(f"imageDigest={digest}")
    else:
        args.append(f"imageTag={tag}")
    image_data = json_aws(aws_cfg["profile"], aws_cfg["region"], *args, allow_failure=True)
    if image_data is None:
        checks.append(fail("ecr image", f"image not found for tag={tag!r} digest={digest!r}"))
    else:
        detail = image_data["imageDetails"][0]
        checks.append(ok("ecr image", f"{detail['imageDigest']} tags={detail.get('imageTags', [])}"))
    return checks


def check_instance_role(config: dict[str, Any]) -> list[Check]:
    aws_cfg = config["aws"]
    identity = config["identity"]
    repo_arn = (
        f"arn:aws:ecr:{aws_cfg['region']}:{aws_cfg['account_id']}:"
        f"repository/{config['image']['repository']}"
    )
    checks: list[Check] = []
    profile = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "iam",
        "get-instance-profile",
        "--instance-profile-name",
        identity["instance_profile"],
    )
    roles = [role["RoleName"] for role in profile["InstanceProfile"].get("Roles", [])]
    if identity["role_name"] not in roles:
        checks.append(fail("instance profile", f"{identity['role_name']} not in {roles}"))
        return checks
    checks.append(ok("instance profile", f"{identity['instance_profile']} contains {identity['role_name']}"))

    simulation = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "iam",
        "simulate-principal-policy",
        "--policy-source-arn",
        f"arn:aws:iam::{aws_cfg['account_id']}:role/{identity['role_name']}",
        "--action-names",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
        "ecr:PutImage",
        "--resource-arns",
        repo_arn,
    )
    decisions = {item["EvalActionName"]: item["EvalDecision"] for item in simulation["EvaluationResults"]}
    if decisions.get("ecr:BatchGetImage") == "allowed" and decisions.get("ecr:GetDownloadUrlForLayer") == "allowed":
        checks.append(ok("ecr pull policy", "BatchGetImage and GetDownloadUrlForLayer allowed"))
    else:
        checks.append(fail("ecr pull policy", json.dumps(decisions, sort_keys=True)))

    token_simulation = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "iam",
        "simulate-principal-policy",
        "--policy-source-arn",
        f"arn:aws:iam::{aws_cfg['account_id']}:role/{identity['role_name']}",
        "--action-names",
        "ecr:GetAuthorizationToken",
        "--resource-arns",
        "*",
    )
    token_decision = token_simulation["EvaluationResults"][0]["EvalDecision"]
    if token_decision == "allowed":
        checks.append(ok("ecr auth token", "GetAuthorizationToken allowed for Docker login"))
    else:
        checks.append(fail("ecr auth token", f"GetAuthorizationToken decision={token_decision}"))

    if decisions.get("ecr:PutImage") == "implicitDeny":
        checks.append(ok("ecr push denied", "instance role cannot PutImage"))
    else:
        checks.append(fail("ecr push denied", f"unexpected PutImage decision {decisions.get('ecr:PutImage')}"))
    return checks


def check_artifacts(config: dict[str, Any]) -> Check:
    aws_cfg = config["aws"]
    parsed = urlparse(config["artifacts"]["durable_uri"])
    if parsed.scheme != "s3" or not parsed.netloc:
        return fail("artifact uri", f"invalid S3 URI: {config['artifacts']['durable_uri']}")
    result = run_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "s3api",
        "head-bucket",
        "--bucket",
        parsed.netloc,
        allow_failure=True,
    )
    if result.returncode != 0:
        return fail("artifact bucket", result.stderr.strip())
    versioning = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "s3api",
        "get-bucket-versioning",
        "--bucket",
        parsed.netloc,
    )
    return ok("artifact bucket", f"s3://{parsed.netloc}; versioning={versioning.get('Status', 'Disabled')}")


def check_artifact_role_access(config: dict[str, Any]) -> list[Check]:
    aws_cfg = config["aws"]
    identity = config["identity"]
    parsed = urlparse(config["artifacts"]["durable_uri"])
    prefix = parsed.path.lstrip("/")
    object_arn = f"arn:aws:s3:::{parsed.netloc}/{prefix.rstrip('/')}/preflight.txt"
    bucket_arn = f"arn:aws:s3:::{parsed.netloc}"
    role_arn = f"arn:aws:iam::{aws_cfg['account_id']}:role/{identity['role_name']}"

    checks: list[Check] = []
    list_result = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "iam",
        "simulate-principal-policy",
        "--policy-source-arn",
        role_arn,
        "--action-names",
        "s3:ListBucket",
        "--resource-arns",
        bucket_arn,
        "--context-entries",
        f"ContextKeyName=s3:prefix,ContextKeyType=string,ContextKeyValues={prefix}",
    )
    list_decision = list_result["EvaluationResults"][0]["EvalDecision"]
    if list_decision == "allowed":
        checks.append(ok("artifact list policy", f"{prefix} is listable by {identity['role_name']}"))
    else:
        checks.append(fail("artifact list policy", f"s3:ListBucket decision={list_decision} for {prefix}"))

    write_result = json_aws(
        aws_cfg["profile"],
        aws_cfg["region"],
        "iam",
        "simulate-principal-policy",
        "--policy-source-arn",
        role_arn,
        "--action-names",
        "s3:GetObject",
        "s3:PutObject",
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts",
        "--resource-arns",
        object_arn,
    )
    decisions = {
        item["EvalActionName"]: item["EvalDecision"]
        for item in write_result["EvaluationResults"]
    }
    if all(decision == "allowed" for decision in decisions.values()):
        checks.append(ok("artifact write policy", f"{object_arn} read/write actions allowed"))
    else:
        checks.append(fail("artifact write policy", json.dumps(decisions, sort_keys=True)))
    return checks


def check_launch_safety(config: dict[str, Any]) -> list[Check]:
    safety = config["safety"]
    checks: list[Check] = []
    if safety.get("instance_initiated_shutdown_behavior") == "terminate":
        checks.append(ok("shutdown behavior", "instance-initiated shutdown terminates the instance"))
    else:
        checks.append(fail("shutdown behavior", "expected instance_initiated_shutdown_behavior=terminate"))

    metadata = safety.get("metadata_options", {})
    if metadata.get("http_tokens") == "required" and metadata.get("http_endpoint") == "enabled":
        checks.append(ok("metadata options", "IMDSv2 is required"))
    else:
        checks.append(fail("metadata options", f"unexpected metadata options: {metadata}"))

    if safety.get("launch_enabled"):
        checks.append(warn("launch mode", "launch_enabled=true; wrapper still requires explicit confirmation"))
    else:
        checks.append(ok("launch mode", "launch disabled; wrapper dry-run only until explicitly enabled"))
    return checks


def render(checks: list[Check], as_json: bool) -> None:
    if as_json:
        print(json.dumps([check.__dict__ for check in checks], indent=2, sort_keys=True))
        return
    width = max(len(check.name) for check in checks)
    for check in checks:
        print(f"{check.status.upper():7} {check.name:<{width}}  {check.detail}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    checks: list[Check] = [
        check_identity(config),
        check_quota(config),
        *check_instance_type(config),
        check_price(config),
        check_host_ami(config),
        *check_network(config),
        *check_ecr(config),
        *check_instance_role(config),
        check_artifacts(config),
        *check_artifact_role_access(config),
        *check_launch_safety(config),
    ]
    render(checks, args.json)
    has_failures = any(check.status == "fail" for check in checks)
    has_warnings = any(check.status == "warning" for check in checks)
    if has_failures or (args.strict and has_warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
