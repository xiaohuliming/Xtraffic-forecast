from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from .manifest import load_table4_manifest


_PLAN_KEYS = {
    "schema_version",
    "plan_id",
    "protocol_id",
    "matrix_id",
    "manifest_sha256",
    "gpu_sku",
    "execution_layout",
    "region_host_assignment",
    "hosts",
    "shards",
}
_HOST_KEYS = {"host_id", "gpu_count", "gpu_sku"}
_SHARD_KEYS = {
    "shard_id",
    "host_id",
    "gpu_index",
    "results_root",
    "task_ids",
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda item: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON constant is forbidden: {item}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError("Distributed plan must contain a JSON object")
    return value


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def load_distributed_plan(
    plan_path: Path | str,
    *,
    manifest_path: Path | str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    plan_source = Path(plan_path)
    plan = _read_json(plan_source)
    if set(plan) != _PLAN_KEYS:
        raise ValueError("Distributed plan schema mismatch")
    if plan["schema_version"] != 1:
        raise ValueError("Distributed plan schema_version must be 1")
    for field in (
        "plan_id",
        "protocol_id",
        "matrix_id",
        "manifest_sha256",
        "gpu_sku",
        "execution_layout",
    ):
        _nonempty_string(plan[field], field=field)
    if plan["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError("Distributed plan manifest SHA-256 mismatch")

    manifest = load_table4_manifest(
        manifest_path,
        expected_sha256=expected_manifest_sha256,
    )
    if plan["protocol_id"] != manifest["protocol_id"]:
        raise ValueError("Distributed plan protocol_id mismatch")
    if plan["matrix_id"] != manifest["matrix_id"]:
        raise ValueError("Distributed plan matrix_id mismatch")
    if plan["execution_layout"] != "dual_host_gpu_sku_homogeneous":
        raise ValueError("Distributed plan execution_layout mismatch")

    hosts = plan["hosts"]
    if not isinstance(hosts, list) or len(hosts) != 2:
        raise ValueError("Distributed plan must declare exactly two hosts")
    host_by_id: dict[str, dict[str, Any]] = {}
    for host in hosts:
        if not isinstance(host, dict) or set(host) != _HOST_KEYS:
            raise ValueError("Distributed plan host schema mismatch")
        host_id = _nonempty_string(host["host_id"], field="host_id")
        if host_id in host_by_id:
            raise ValueError("Distributed plan host_id values must be unique")
        if type(host["gpu_count"]) is not int or host["gpu_count"] <= 0:
            raise ValueError("Distributed plan gpu_count must be positive")
        if host["gpu_sku"] != plan["gpu_sku"]:
            raise ValueError("Distributed plan hosts must use the declared GPU SKU")
        host_by_id[host_id] = host

    assignment = plan["region_host_assignment"]
    if not isinstance(assignment, dict) or set(assignment) != set(host_by_id):
        raise ValueError("Distributed plan region_host_assignment mismatch")
    region_to_host: dict[str, str] = {}
    for host_id, regions in assignment.items():
        if not isinstance(regions, list) or not regions:
            raise ValueError("Each host must receive at least one region")
        for region in regions:
            _nonempty_string(region, field="region")
            if region in region_to_host:
                raise ValueError("Each region must be assigned exactly once")
            region_to_host[region] = host_id
    expected_regions = {task["region"] for task in manifest["tasks"]}
    if set(region_to_host) != expected_regions:
        raise ValueError("Distributed plan must assign every manifest region")

    shards = plan["shards"]
    if not isinstance(shards, list):
        raise ValueError("Distributed plan shards must be a list")
    if len(shards) != sum(host["gpu_count"] for host in hosts):
        raise ValueError("Distributed plan must declare one shard per GPU")
    shard_ids: set[str] = set()
    gpu_slots: set[tuple[str, int]] = set()
    assigned_task_ids: list[str] = []
    task_host: dict[str, str] = {}
    manifest_order = {
        task["task_id"]: index for index, task in enumerate(manifest["tasks"])
    }
    for shard in shards:
        if not isinstance(shard, dict) or set(shard) != _SHARD_KEYS:
            raise ValueError("Distributed plan shard schema mismatch")
        shard_id = _nonempty_string(shard["shard_id"], field="shard_id")
        host_id = _nonempty_string(shard["host_id"], field="host_id")
        if shard_id in shard_ids:
            raise ValueError("Distributed plan shard_id values must be unique")
        shard_ids.add(shard_id)
        if host_id not in host_by_id:
            raise ValueError("Distributed plan shard references an unknown host")
        gpu_index = shard["gpu_index"]
        if (
            type(gpu_index) is not int
            or gpu_index < 0
            or gpu_index >= host_by_id[host_id]["gpu_count"]
        ):
            raise ValueError("Distributed plan shard gpu_index is invalid")
        slot = (host_id, gpu_index)
        if slot in gpu_slots:
            raise ValueError("Distributed plan GPU slots must be unique")
        gpu_slots.add(slot)
        results_root = _nonempty_string(
            shard["results_root"],
            field="results_root",
        )
        pure_root = PurePosixPath(results_root)
        if (
            pure_root.is_absolute()
            or ".." in pure_root.parts
            or str(pure_root) != results_root
            or pure_root.parts[:1] != ("shards",)
        ):
            raise ValueError("Distributed plan results_root is unsafe")
        task_ids = shard["task_ids"]
        if not isinstance(task_ids, list) or not task_ids:
            raise ValueError("Distributed plan shards must not be empty")
        if any(not isinstance(task_id, str) for task_id in task_ids):
            raise ValueError("Distributed plan task IDs must be strings")
        if task_ids != sorted(task_ids, key=manifest_order.get):
            raise ValueError("Shard task IDs must follow canonical manifest order")
        assigned_task_ids.extend(task_ids)
        for task_id in task_ids:
            task_host[task_id] = host_id

    expected_task_ids = [task["task_id"] for task in manifest["tasks"]]
    if (
        len(assigned_task_ids) != len(set(assigned_task_ids))
        or set(assigned_task_ids) != set(expected_task_ids)
    ):
        raise ValueError("Distributed plan must assign every task exactly once")
    for task in manifest["tasks"]:
        if task_host[task["task_id"]] != region_to_host[task["region"]]:
            raise ValueError("Every region must remain on its assigned host")
    return plan


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    arguments = parser.parse_args()
    plan = load_distributed_plan(
        arguments.plan,
        manifest_path=arguments.manifest,
        expected_manifest_sha256=arguments.manifest_sha256,
    )
    print(
        json.dumps(
            {
                "plan_id": plan["plan_id"],
                "plan_sha256": sha256_file(arguments.plan),
                "host_count": len(plan["hosts"]),
                "shard_count": len(plan["shards"]),
                "task_count": sum(
                    len(shard["task_ids"]) for shard in plan["shards"]
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
