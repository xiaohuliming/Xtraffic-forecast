from __future__ import annotations

from pathlib import Path

from scripts.reproduction.igstgnn_table4.distributed_host_runner import (
    build_host_launch_specs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "matrix.public_material_v2.json"
)
PLAN = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "distributed_4090d_9gpu_v1.json"
)
MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357e"
    "b5320393417a83d344adb05151023a78"
)


def test_host_launch_specs_bind_one_canonical_shard_to_each_visible_gpu(tmp_path):
    specs = build_host_launch_specs(
        plan_path=PLAN,
        manifest_path=MANIFEST,
        expected_manifest_sha256=MANIFEST_SHA256,
        host_id="table4_4090d5",
        splits_root=tmp_path / "splits",
        results_base=tmp_path / "results",
        python_executable=Path("/locked/env/bin/python"),
    )

    assert len(specs) == 5
    assert {spec["cuda_visible_devices"] for spec in specs} == {
        "0",
        "1",
        "2",
        "3",
        "4",
    }
    assert all(spec["command"][0] == "/locked/env/bin/python" for spec in specs)
    assert all(
        spec["command"][spec["command"].index("--device") + 1] == "cuda:0"
        for spec in specs
    )
    task_ids = [
        task_id
        for spec in specs
        for task_id in spec["task_ids"]
    ]
    assert len(task_ids) == 13
    assert len(task_ids) == len(set(task_ids))
    assert all(
        Path(spec["results_root"]).is_relative_to(tmp_path / "results")
        for spec in specs
    )
