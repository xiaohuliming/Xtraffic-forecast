from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from scripts.remote.distributed import protocol_v3_distributed as distributed
from scripts.remote.windows import protocol_v3_runner as runner


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT
    / "scripts"
    / "remote"
    / "windows"
    / "protocol_v3_manifest.json"
)
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "remote"
    / "distributed"
    / "protocol_v3_composite_postflight.py"
)
CODE_SHA = "3a4952837aaf6fbda1403d50b610b1bf40c6098a"
CACHE_BUNDLE_ID = "bundle-composite-test"
CACHE_BUNDLE_DIGEST = "b" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            _sha256(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_result(
    root: Path,
    task: runner.Task,
    manifest: runner.Manifest,
    run_inputs: runner.ProtocolV3Inputs,
) -> str:
    invocation = runner.build_production_invocation(
        task,
        manifest,
        run_inputs,
        root / ".config_probe" / task.task_id,
    )
    config_hash = runner.derive_config_hash(
        task,
        manifest,
        run_inputs,
        invocation_config=invocation.config,
    )
    result_dir = root / task.output_dir
    result_dir.mkdir(parents=True)

    prediction = np.zeros((2, 1, 2), dtype=np.float32)
    actual = np.ones((2, 1, 2), dtype=np.float32)
    np.savez_compressed(
        result_dir / "test_predictions.npz",
        region_code=np.full(
            (2,),
            {"Alameda": 0, "ContraCosta": 1, "Orange": 2}[task.region],
            dtype=np.int64,
        ),
        sample_start=np.array([100, 200], dtype=np.int64),
        region_node_idx=np.array([10, 11], dtype=np.int64),
        pred_raw_flow=prediction,
        actual_future_flow=actual,
        y_mask_flow=np.ones_like(prediction, dtype=np.uint8),
        affected_mask=np.array([[1, 0], [0, 1]], dtype=np.uint8),
    )
    summary = {
        "task_id": task.task_id,
        "region": task.region,
        "model": task.model,
        "variant": task.variant,
        "seed": task.seed,
        "protocol_version": manifest.protocol_version,
        "sampling_protocol": task.sampling_protocol,
        "cache_schema_version": manifest.cache_schema_version,
        "cache_bundle_id": run_inputs.cache_bundle_id,
        "cache_bundle_digest": run_inputs.cache_bundle_digest,
        "code_sha": run_inputs.code_sha,
        "config_hash": config_hash,
        "valid_count": 4,
        "metrics": {
            "all": 1.0,
            "affected": 1.0,
            "unaffected": 1.0,
        },
        "private_summary_marker": f"do-not-copy-{task.task_id}",
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for relative in invocation.required_artifacts:
        path = result_dir / relative
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact:{task.task_id}:{relative}\n".encode())
    return config_hash


def _write_shard_spec(
    tmp_path: Path,
    manifest: runner.Manifest,
    *,
    host: str,
    task_ids: list[str],
) -> tuple[Path, Path]:
    results_root = tmp_path / host / "results"
    run_inputs = runner.ProtocolV3Inputs(
        results_root=results_root,
        cache_dir=tmp_path / host / "formal_cache",
        graph_dir=tmp_path / host / "formal_graphs",
        cache_bundle_id=CACHE_BUNDLE_ID,
        cache_bundle_digest=CACHE_BUNDLE_DIGEST,
        code_sha=CODE_SHA,
        python_executable=sys.executable,
    )
    tasks = {task.task_id: task for task in manifest.tasks}
    config_hashes = {
        task_id: _write_result(
            results_root,
            tasks[task_id],
            manifest,
            run_inputs,
        )
        for task_id in task_ids
    }
    payload = {
        "spec_version": 1,
        "host": host,
        "results_root": str(results_root),
        "task_ids": task_ids,
        "run_inputs": {
            "cache_dir": str(run_inputs.cache_dir),
            "graph_dir": str(run_inputs.graph_dir),
            "cache_bundle_id": run_inputs.cache_bundle_id,
            "cache_bundle_digest": run_inputs.cache_bundle_digest,
            "code_sha": run_inputs.code_sha,
            "python_executable": run_inputs.python_executable,
        },
        "config_hashes": config_hashes,
        "runtime_provenance": {
            "python": "3.11.9",
            "torch": "2.7.1",
            "torch_cuda": "12.8",
            "numpy": "2.1.3",
            "h5py": "3.12.1",
            "pandas": "2.2.3",
            "gpu": f"NVIDIA GeForce {host}",
        },
    }
    spec_path = tmp_path / f"{host}.json"
    spec_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return spec_path, results_root


def _two_shards(
    tmp_path: Path,
) -> tuple[runner.Manifest, Path, Path, Path, Path]:
    manifest = runner.load_manifest(
        MANIFEST_PATH,
        require_production_matrix=True,
    )
    first_ids = [
        task.task_id
        for task in manifest.tasks
        if distributed.task_number(task)
        in distributed.SHARD_SPECS["5080"].task_numbers
    ]
    second_ids = [
        task.task_id
        for task in manifest.tasks
        if distributed.task_number(task)
        in distributed.SHARD_SPECS["5090"].task_numbers
    ]
    first_spec, first_root = _write_shard_spec(
        tmp_path,
        manifest,
        host="RTX5080",
        task_ids=first_ids,
    )
    second_spec, second_root = _write_shard_spec(
        tmp_path,
        manifest,
        host="RTX5090",
        task_ids=second_ids,
    )
    return manifest, first_spec, first_root, second_spec, second_root


def _write_source_host_report(
    spec_path: Path,
    results_root: Path,
    *,
    shard: str,
    windows_paths: bool,
) -> Path:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    runtime = {
        "hostname": f"gpu-{shard}",
        "platform": "Windows-11" if windows_paths else "Linux-6.8",
        "python": "3.11.9",
        "python_executable": (
            r"C:\miniconda3\python.exe"
            if windows_paths
            else "/root/miniconda3/bin/python"
        ),
        "requested_python_executable": (
            r"C:\miniconda3\python.exe"
            if windows_paths
            else "/root/miniconda3/bin/python"
        ),
        "torch": "2.7.1",
        "torch_cuda": "12.8",
        "numpy": "2.1.3",
        "h5py": "3.12.1",
        "pandas": "2.2.3",
        "gpu": [f"NVIDIA GeForce RTX {shard}, GPU-uuid, 32768, 576.80"],
    }
    if windows_paths:
        source_results_root = rf"C:\xtraffic\results\shard_{shard}"
        source_cache_dir = r"C:\xtraffic\formal_cache"
        source_graph_dir = r"C:\xtraffic\formal_graphs"
    else:
        source_results_root = f"/root/xtraffic/results/shard_{shard}"
        source_cache_dir = "/root/xtraffic/formal_cache"
        source_graph_dir = "/root/xtraffic/formal_graphs"
    run_inputs = {
        "results_root": source_results_root,
        "cache_dir": source_cache_dir,
        "graph_dir": source_graph_dir,
        "cache_bundle_id": spec["run_inputs"]["cache_bundle_id"],
        "cache_bundle_digest": spec["run_inputs"][
            "cache_bundle_digest"
        ],
        "code_sha": spec["run_inputs"]["code_sha"],
        "python_executable": runtime["requested_python_executable"],
    }
    manifest = runner.load_manifest(
        MANIFEST_PATH,
        require_production_matrix=True,
    )
    tasks_by_id = {task.task_id: task for task in manifest.tasks}
    source_inputs = runner.ProtocolV3Inputs(
        results_root=results_root,
        cache_dir=Path(source_cache_dir),
        graph_dir=Path(source_graph_dir),
        cache_bundle_id=run_inputs["cache_bundle_id"],
        cache_bundle_digest=run_inputs["cache_bundle_digest"],
        code_sha=run_inputs["code_sha"],
        python_executable=run_inputs["python_executable"],
    )
    task_reports = []
    for task_id in spec["task_ids"]:
        task = tasks_by_id[task_id]
        invocation = runner.build_production_invocation(
            task,
            manifest,
            source_inputs,
            results_root / ".source_probe" / task_id,
        )
        invocation_config = dict(invocation.config)
        invocation_config["cache_dir"] = source_cache_dir
        invocation_config["graph_dir"] = source_graph_dir
        config_hash = runner.derive_config_hash(
            task,
            manifest,
            source_inputs,
            invocation_config=invocation_config,
        )
        summary_path = results_root / task.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["config_hash"] = config_hash
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        task_reports.append(
            {
                "task_id": task_id,
                "config_hash": config_hash,
                "invocation_config": invocation_config,
            }
        )
    files = [
        {
            "path": relative,
            "size_bytes": size,
            "sha256": digest,
        }
        for relative, (size, _, digest) in _tree_snapshot(
            results_root
        ).items()
    ]
    payload = {
        "status": "verified",
        "shard": shard,
        "results_root": source_results_root,
        "validated_tasks": len(spec["task_ids"]),
        "protocol_version": "protocol_v3_noleak_dedup",
        "cache_schema_version": 3,
        "code_sha": run_inputs["code_sha"],
        "cache_bundle_id": run_inputs["cache_bundle_id"],
        "cache_bundle_digest": run_inputs["cache_bundle_digest"],
        "manifest_raw_sha256": _sha256(MANIFEST_PATH),
        "manifest_canonical_json_sha256": hashlib.sha256(
            json.dumps(
                json.loads(MANIFEST_PATH.read_text(encoding="utf-8")),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "run_inputs": run_inputs,
        "runtime": runtime,
        "tasks": task_reports,
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files_digest": hashlib.sha256(
            json.dumps(
                files,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "export_paths": [item["path"] for item in files],
        "files": files,
    }
    path = spec_path.with_name(f"shard_{shard}_postflight.json")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _two_source_reports(
    tmp_path: Path,
) -> tuple[runner.Manifest, Path, Path, Path, Path]:
    (
        manifest,
        first_spec,
        first_root,
        second_spec,
        second_root,
    ) = _two_shards(tmp_path)
    first_report = _write_source_host_report(
        first_spec,
        first_root,
        shard="5080",
        windows_paths=True,
    )
    second_report = _write_source_host_report(
        second_spec,
        second_root,
        shard="5090",
        windows_paths=False,
    )
    return manifest, first_report, first_root, second_report, second_root


def _refresh_report_file_record(
    report_path: Path,
    results_root: Path,
    relative: str,
) -> None:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    path = results_root / Path(*PurePosixPath(relative).parts)
    for record in payload["files"]:
        if record["path"] == relative:
            record["size_bytes"] = path.stat().st_size
            record["sha256"] = _sha256(path)
            break
    else:
        raise AssertionError(f"missing report file record: {relative}")
    payload["total_bytes"] = sum(
        record["size_bytes"] for record in payload["files"]
    )
    payload["files_digest"] = hashlib.sha256(
        json.dumps(
            payload["files"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_builds_read_only_index_for_two_valid_disjoint_shards(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        manifest,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    before = {
        "RTX5080": _tree_snapshot(first_root),
        "RTX5090": _tree_snapshot(second_root),
    }
    output = tmp_path / "composite_index.json"

    assert protocol_v3_composite_postflight.main(
        [
            "--manifest",
            str(MANIFEST_PATH),
            "--shard-report",
            str(first_report),
            "--shard-report",
            str(second_report),
            "--results-root",
            str(first_root),
            "--results-root",
            str(second_root),
            "--output",
            str(output),
        ]
    ) == 0

    assert before == {
        "RTX5080": _tree_snapshot(first_root),
        "RTX5090": _tree_snapshot(second_root),
    }
    index = json.loads(output.read_text(encoding="utf-8"))
    assert index["status"] == "verified"
    assert index["protocol_version"] == "protocol_v3_noleak_dedup"
    assert index["task_count"] == 45
    assert index["shard_count"] == 2
    assert [task["task_id"] for task in index["tasks"]] == [
        task.task_id for task in manifest.tasks
    ]
    assert {task["host"] for task in index["tasks"]} == {
        "gpu-5080",
        "gpu-5090",
    }
    first = index["tasks"][0]
    assert first["runtime_provenance"]["gpu"] == [
        "NVIDIA GeForce RTX 5080, GPU-uuid, 32768, 576.80"
    ]
    assert first["summary"]["path"] == (
        f"tasks/{first['task_id']}/summary.json"
    )
    assert len(first["summary"]["sha256"]) == 64
    assert first["artifacts"]["summary.json"]["sha256"] == (
        first["summary"]["sha256"]
    )
    assert "private_summary_marker" not in output.read_text(encoding="utf-8")


def test_builds_index_from_real_crlf_source_manifest_identity(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        manifest,
        first_spec,
        first_root,
        second_spec,
        second_root,
    ) = _two_shards(tmp_path)
    first_report = _write_source_host_report(
        first_spec,
        first_root,
        shard="5080",
        windows_paths=True,
    )
    second_report = _write_source_host_report(
        second_spec,
        second_root,
        shard="5090",
        windows_paths=False,
    )
    manifest_bytes = MANIFEST_PATH.read_bytes()
    assert b"\n" in manifest_bytes
    assert b"\r\n" not in manifest_bytes
    crlf_manifest = tmp_path / "protocol_v3_manifest_crlf.json"
    crlf_manifest.write_bytes(manifest_bytes.replace(b"\n", b"\r\n"))
    crlf_bytes = crlf_manifest.read_bytes()
    assert b"\r\n" in crlf_bytes
    assert b"\n" not in crlf_bytes.replace(b"\r\n", b"")
    crlf_payload = json.loads(crlf_manifest.read_text(encoding="utf-8"))
    crlf_canonical_sha256 = hashlib.sha256(
        json.dumps(
            crlf_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    first_payload = json.loads(first_report.read_text(encoding="utf-8"))
    first_payload["manifest_raw_sha256"] = _sha256(crlf_manifest)
    first_payload[
        "manifest_canonical_json_sha256"
    ] = crlf_canonical_sha256
    first_report.write_text(
        json.dumps(first_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    second_payload = json.loads(second_report.read_text(encoding="utf-8"))
    assert first_payload["manifest_raw_sha256"] != _sha256(MANIFEST_PATH)
    assert crlf_canonical_sha256 == second_payload[
        "manifest_canonical_json_sha256"
    ]
    output = tmp_path / "report_backed_index.json"

    payload = protocol_v3_composite_postflight.build_composite_index(
        manifest_path=MANIFEST_PATH,
        shard_report_paths=[first_report, second_report],
        results_roots=[first_root, second_root],
        output_path=output,
    )

    assert payload["status"] == "verified"
    assert payload["task_count"] == 45
    assert [task["task_id"] for task in payload["tasks"]] == [
        task.task_id for task in manifest.tasks
    ]
    first = payload["tasks"][0]
    assert first["shard"] == "5080"
    assert first["host"] == "gpu-5080"
    assert first["source_run_inputs"]["cache_dir"].startswith("C:\\")
    assert len(first["source_postflight_sha256"]) == 64


def test_rejects_overlapping_task_assignments_without_writing_index(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        manifest,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    first_payload = json.loads(first_report.read_text(encoding="utf-8"))
    second_payload = json.loads(second_report.read_text(encoding="utf-8"))
    second_payload["tasks"].append(first_payload["tasks"][0])
    second_payload["validated_tasks"] += 1
    second_report.write_text(
        json.dumps(second_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    before = {
        "RTX5080": _tree_snapshot(first_root),
        "RTX5090": _tree_snapshot(second_root),
    }
    output = tmp_path / "must_not_exist.json"

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="disjoint union",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=output,
        )

    assert not output.exists()
    assert before == {
        "RTX5080": _tree_snapshot(first_root),
        "RTX5090": _tree_snapshot(second_root),
    }


def test_rejects_incomplete_task_union(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        manifest,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    second_payload = json.loads(second_report.read_text(encoding="utf-8"))
    omitted = manifest.tasks[-1].task_id
    second_payload["tasks"] = [
        item
        for item in second_payload["tasks"]
        if item["task_id"] != omitted
    ]
    second_payload["validated_tasks"] -= 1
    second_report.write_text(
        json.dumps(second_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="missing=.*045_fullwindow",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=tmp_path / "must_not_exist.json",
        )


def test_rejects_disjoint_complete_but_swapped_fixed_assignments(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        _,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    first_payload = json.loads(first_report.read_text(encoding="utf-8"))
    second_payload = json.loads(second_report.read_text(encoding="utf-8"))
    first_payload["tasks"][0], second_payload["tasks"][0] = (
        second_payload["tasks"][0],
        first_payload["tasks"][0],
    )
    first_report.write_text(
        json.dumps(first_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    second_report.write_text(
        json.dumps(second_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "must_not_exist.json"

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="shard 5080 differs from its fixed task assignment",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=output,
        )

    assert not output.exists()


def test_rejects_noncontract_shard_names(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        _,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    first_payload = json.loads(first_report.read_text(encoding="utf-8"))
    first_payload["shard"] = "RTX5080"
    first_report.write_text(
        json.dumps(first_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="shard names must equal the fixed distributed contract",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=tmp_path / "must_not_exist.json",
        )


def test_rejects_synchronized_summary_and_config_proof_tampering(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        manifest,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    first_payload = json.loads(first_report.read_text(encoding="utf-8"))
    task = manifest.tasks[0]
    task_proof = first_payload["tasks"][0]
    task_proof["invocation_config"]["epochs"] = 31
    raw_inputs = first_payload["run_inputs"]
    proof_inputs = runner.ProtocolV3Inputs(
        results_root=first_root,
        cache_dir=Path(raw_inputs["cache_dir"]),
        graph_dir=Path(raw_inputs["graph_dir"]),
        cache_bundle_id=raw_inputs["cache_bundle_id"],
        cache_bundle_digest=raw_inputs["cache_bundle_digest"],
        code_sha=raw_inputs["code_sha"],
        python_executable=raw_inputs["python_executable"],
    )
    tampered_hash = runner.derive_config_hash(
        task,
        manifest,
        proof_inputs,
        invocation_config=task_proof["invocation_config"],
    )
    task_proof["config_hash"] = tampered_hash
    first_report.write_text(
        json.dumps(first_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path = first_root / task.output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["config_hash"] = tampered_hash
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_report_file_record(
        first_report,
        first_root,
        (task.output_dir / "summary.json").as_posix(),
    )

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="invocation_config mismatch",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=tmp_path / "must_not_exist.json",
        )


def test_rejects_symlinked_artifact_even_when_runner_can_read_it(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        manifest,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    task = manifest.tasks[0]
    result_dir = first_root / task.output_dir
    checkpoint = result_dir / "ckpt_best.pt"
    checkpoint.unlink()
    checkpoint.symlink_to(result_dir / "summary.json")

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="symlink",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=tmp_path / "must_not_exist.json",
        )


def test_rejects_mixed_training_code_identities(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        _,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    second_payload = json.loads(second_report.read_text(encoding="utf-8"))
    second_payload["run_inputs"]["code_sha"] = "a" * 40
    second_payload["code_sha"] = "a" * 40
    second_report.write_text(
        json.dumps(second_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="code_sha mismatch",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=tmp_path / "must_not_exist.json",
        )


def test_rejects_incomplete_runtime_provenance(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        _,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    first_payload = json.loads(first_report.read_text(encoding="utf-8"))
    del first_payload["runtime"]["gpu"]
    first_report.write_text(
        json.dumps(first_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="runtime is missing fields.*gpu",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=tmp_path / "must_not_exist.json",
        )


def test_rejects_unassigned_task_directory_in_a_shard(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        _,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    (first_root / "tasks" / "foreign_task").mkdir()

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="task directories mismatch.*foreign_task",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=tmp_path / "must_not_exist.json",
        )


def test_rejects_duplicate_keys_in_summary(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        manifest,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    task = manifest.tasks[0]
    summary_path = first_root / task.output_dir / "summary.json"
    text = summary_path.read_text(encoding="utf-8").rstrip()
    summary_path.write_text(
        text[:-1] + f',\n  "task_id": "{task.task_id}"\n}}\n',
        encoding="utf-8",
    )
    _refresh_report_file_record(
        first_report,
        first_root,
        (task.output_dir / "summary.json").as_posix(),
    )

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="duplicate JSON key.*task_id",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=tmp_path / "must_not_exist.json",
        )


def test_never_overwrites_existing_index(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        _,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    output = tmp_path / "existing.json"
    output.write_text("preserve-me\n", encoding="utf-8")

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="will not be overwritten",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=output,
        )

    assert output.read_text(encoding="utf-8") == "preserve-me\n"


def test_rejects_output_inside_result_root_without_creating_parent(tmp_path):
    from scripts.remote.distributed import protocol_v3_composite_postflight

    (
        _,
        first_report,
        first_root,
        second_report,
        second_root,
    ) = _two_source_reports(tmp_path)
    forbidden_parent = first_root / "composite_control"
    output = forbidden_parent / "index.json"

    with pytest.raises(
        protocol_v3_composite_postflight.CompositeValidationError,
        match="outside both result roots",
    ):
        protocol_v3_composite_postflight.build_composite_index(
            manifest_path=MANIFEST_PATH,
            shard_report_paths=[first_report, second_report],
            results_roots=[first_root, second_root],
            output_path=output,
        )

    assert not forbidden_parent.exists()


def test_cli_is_directly_executable_outside_repository(tmp_path):
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--shard-report" in completed.stdout


def test_standalone_copy_uses_configured_repository_root(tmp_path):
    shallow_script = tmp_path / "protocol_v3_composite_postflight.py"
    shutil.copy2(SCRIPT_PATH, shallow_script)

    completed = subprocess.run(
        [sys.executable, str(shallow_script), "--help"],
        cwd=tmp_path,
        env={
            **os.environ,
            "XTRAFFIC_REPO_ROOT": str(REPO_ROOT),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--shard-report" in completed.stdout
