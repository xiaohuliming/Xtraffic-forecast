import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4 import sandboxed_split
from scripts.reproduction.igstgnn_table4.deep_audit import (
    sequence_content_digest,
)
from scripts.reproduction.igstgnn_table4.sandboxed_split import (
    run_sandboxed_official_split,
)


def _sample(index):
    ticks = np.arange(48 + index, 48 + index + 24, dtype=np.float32)
    x_data = np.empty((12, 3, 3), dtype=np.float32)
    x_data[..., 0] = index
    x_data[..., 1] = (ticks[:12] % 288)[:, None] / 288
    x_data[..., 2] = 4 / 7
    y_data = np.empty((12, 3, 3), dtype=np.float32)
    y_data[..., 0] = index + 1
    y_data[..., 1] = (ticks[12:] % 288)[:, None] / 288
    y_data[..., 2] = 4 / 7
    return {
        "x_data": x_data,
        "y_data": y_data,
        "incident_features": {
            "Incident Time": float(index),
            "Description": 0,
            "Type": 0,
            "Holiday": 0,
        },
        "incident_position": 11,
        "incident_distances": np.ones((3, 3), dtype=np.float64),
        "durations": np.ones(12, dtype=np.float64),
    }


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_valid_support_files(region_directory, *, node_count):
    np.save(
        region_directory / "adj_matrix.npy",
        np.zeros((node_count, node_count), dtype=np.float64),
    )
    np.savez(
        region_directory / "incident_stats.npz",
        mean=np.float32(1.0),
        std=np.float32(2.0),
        normalized=np.array(True),
    )
    sensor_rows = [
        "order,Sensor Type,Surface,Roadway Use,Road Width,Design Speed Limit"
    ]
    sensor_rows.extend(
        f"{index},loops,Concrete,No Special Features,80,70"
        for index in range(node_count)
    )
    (region_directory / "sensors.csv").write_text(
        "\n".join(sensor_rows) + "\n",
        encoding="utf-8",
    )
    (region_directory / "desc_mapping.json").write_text(
        json.dumps({"description": 0}),
        encoding="utf-8",
    )
    (region_directory / "type_mapping.json").write_text(
        json.dumps({"type": 0}),
        encoding="utf-8",
    )
    return {
        name: (region_directory / name, _sha256(region_directory / name))
        for name in (
            "adj_matrix.npy",
            "incident_stats.npz",
            "sensors.csv",
            "desc_mapping.json",
            "type_mapping.json",
        )
    }


def _write_locked_release_manifest(
    path,
    *,
    region,
    region_directory,
    sample_count,
    node_count,
):
    files = {
        item.name: {
            "bytes": item.stat().st_size,
            "sha256": _sha256(item),
        }
        for item in region_directory.iterdir()
        if item.is_file()
    }
    manifest = {
        "status": "safe_structure_verified",
        "protocol_id": "igstgnn_table4_official_v1",
        "regions": {
            region: {
                "sample_count": sample_count,
                "node_count": node_count,
                "files": files,
                "safe_structure": {
                    "mappings": {
                        "description_count": 1,
                        "type_count": 1,
                    }
                },
            }
        },
    }
    path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return _sha256(path)


def test_sandboxed_split_atomically_prepares_the_official_order(tmp_path):
    samples = np.asarray([_sample(index) for index in range(20)], dtype=object)
    raw_release_root = tmp_path / "raw"
    prepared_root = tmp_path / "prepared"
    region_directory = raw_release_root / "Synthetic"
    region_directory.mkdir(parents=True)
    prepared_root.mkdir()
    incident_file = region_directory / "incident_all.npy"
    np.save(incident_file, samples)
    expected_sha256 = _sha256(incident_file)
    output_directory = prepared_root / "Synthetic"
    support_files = _write_valid_support_files(
        region_directory,
        node_count=3,
    )
    release_manifest_file = tmp_path / "release_manifest.safe_structure.json"
    release_manifest_sha256 = _write_locked_release_manifest(
        release_manifest_file,
        region="Synthetic",
        region_directory=region_directory,
        sample_count=20,
        node_count=3,
    )

    manifest = run_sandboxed_official_split(
        incident_file=incident_file,
        expected_sha256=expected_sha256,
        expected_content_digest=sequence_content_digest(samples),
        output_directory=output_directory,
        region="Synthetic",
        node_count=3,
        expected_sample_count=20,
        description_count=1,
        type_count=1,
        timeout_seconds=30,
        support_files=support_files,
        raw_release_root=raw_release_root,
        release_manifest_file=release_manifest_file,
        expected_release_manifest_sha256=release_manifest_sha256,
    )

    assert manifest["status"] == "official_splits_prepared"
    assert manifest["schema_version"] == 2
    assert manifest["record_structure_status"] == ("records_structurally_verified")
    assert manifest["split_integrity_status"] == ("no_cross_split_duplicates_detected")
    assert manifest["validity_label"] == ("official_protocol_without_detected_overlap")
    assert manifest["support_structure_status"] == "safe_structure_verified"
    assert manifest["release_manifest_binding"] == {
        "path": str(release_manifest_file),
        "sha256": release_manifest_sha256,
        "region": "Synthetic",
    }
    assert manifest["split_semantics"] == {
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "ordered": True,
        "shuffled": False,
        "renormalized": False,
        "deduplicated": False,
    }
    assert manifest["counts"] == {"train": 14, "val": 3, "test": 3}
    assert manifest["recombined_matches_input"] is True
    assert manifest["duplicate_model_consumed_record_count"] == 0
    assert manifest["cross_split_model_consumed_record_intersections"] == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    assert len(manifest["model_consumed_digest"]) == 64
    assert manifest["independent_postflight"]["status"] == (
        "official_splits_independently_reverified"
    )
    assert manifest["independent_postflight"][
        "recombined_content_digest"
    ] == sequence_content_digest(samples)
    assert (
        manifest["independent_postflight"]["model_consumed_digest"]
        == manifest["model_consumed_digest"]
    )
    assert (
        manifest["independent_postflight"]["model_consumed_record_multiplicity"]
        == manifest["model_consumed_record_multiplicity"]
    )
    negative_controls = manifest["security_boundary"]["negative_controls"]
    assert set(negative_controls) == {"writer", "postflight"}
    assert negative_controls["writer"]["status"] == ("sandbox_boundary_verified")
    assert negative_controls["writer"]["profile"] == "writer"
    assert negative_controls["postflight"]["status"] == ("sandbox_boundary_verified")
    assert negative_controls["postflight"]["profile"] == "postflight"
    assert (
        negative_controls["writer"]["sandbox_profile_sha256"]
        == (manifest["security_boundary"]["split_profile_sha256"])
    )
    assert (
        negative_controls["postflight"]["sandbox_profile_sha256"]
        == (manifest["security_boundary"]["postflight_profile_sha256"])
    )
    assert output_directory.is_dir()
    expected_artifacts = {
        "incident_train.npy",
        "incident_val.npy",
        "incident_test.npy",
        "split_manifest.json",
        "adj_matrix.npy",
        "incident_stats.npz",
        "sensors.csv",
        "desc_mapping.json",
        "type_mapping.json",
    }
    assert set(manifest["artifact_allowlist"]) == expected_artifacts
    assert {path.name for path in output_directory.iterdir()} == expected_artifacts
    assert set(manifest["support_files"]) == set(support_files)


def test_sandboxed_split_refuses_to_write_inside_the_raw_release(tmp_path):
    raw_release_root = tmp_path / "raw"
    raw_release_root.mkdir()
    samples = np.asarray([_sample(index) for index in range(20)], dtype=object)
    incident_file = raw_release_root / "incident_all.npy"
    np.save(incident_file, samples)

    with np.testing.assert_raises_regex(
        ValueError,
        "raw release",
    ):
        run_sandboxed_official_split(
            incident_file=incident_file,
            expected_sha256=hashlib.sha256(incident_file.read_bytes()).hexdigest(),
            expected_content_digest=sequence_content_digest(samples),
            output_directory=raw_release_root / "Synthetic",
            raw_release_root=raw_release_root,
            region="Synthetic",
            node_count=3,
            expected_sample_count=20,
            description_count=1,
            type_count=1,
            timeout_seconds=30,
            support_files={},
            release_manifest_file=tmp_path / "unused.json",
            expected_release_manifest_sha256="0" * 64,
        )


def test_sandboxed_split_rejects_support_from_another_directory(tmp_path):
    samples = np.asarray([_sample(index) for index in range(20)], dtype=object)
    raw_release_root = tmp_path / "raw"
    region_directory = raw_release_root / "Synthetic"
    region_directory.mkdir(parents=True)
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    incident_file = region_directory / "incident_all.npy"
    np.save(incident_file, samples)
    support_files = _write_valid_support_files(
        region_directory,
        node_count=3,
    )
    release_manifest_file = tmp_path / "release_manifest.safe_structure.json"
    release_manifest_sha256 = _write_locked_release_manifest(
        release_manifest_file,
        region="Synthetic",
        region_directory=region_directory,
        sample_count=20,
        node_count=3,
    )
    other_directory = raw_release_root / "Other"
    other_directory.mkdir()
    external_adjacency = other_directory / "adj_matrix.npy"
    external_adjacency.write_bytes((region_directory / "adj_matrix.npy").read_bytes())
    support_files["adj_matrix.npy"] = (
        external_adjacency,
        _sha256(external_adjacency),
    )

    with pytest.raises(ValueError, match="fixed same-region file"):
        run_sandboxed_official_split(
            incident_file=incident_file,
            expected_sha256=_sha256(incident_file),
            expected_content_digest=sequence_content_digest(samples),
            output_directory=prepared_root / "Synthetic",
            raw_release_root=raw_release_root,
            region="Synthetic",
            node_count=3,
            expected_sample_count=20,
            description_count=1,
            type_count=1,
            timeout_seconds=30,
            support_files=support_files,
            release_manifest_file=release_manifest_file,
            expected_release_manifest_sha256=release_manifest_sha256,
        )


def test_sandboxed_split_bounds_every_parent_side_json_read(monkeypatch, tmp_path):
    samples = np.asarray([_sample(index) for index in range(20)], dtype=object)
    raw_release_root = tmp_path / "raw"
    region_directory = raw_release_root / "Synthetic"
    region_directory.mkdir(parents=True)
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    incident_file = region_directory / "incident_all.npy"
    np.save(incident_file, samples)
    support_files = _write_valid_support_files(
        region_directory,
        node_count=3,
    )
    release_manifest_file = tmp_path / "release_manifest.safe_structure.json"
    release_manifest_sha256 = _write_locked_release_manifest(
        release_manifest_file,
        region="Synthetic",
        region_directory=region_directory,
        sample_count=20,
        node_count=3,
    )
    labels = []
    real_read_bounded_json = sandboxed_split._read_bounded_json

    def observe_bounded_read(path, *, max_bytes, label, expected_sha256=None):
        labels.append(label)
        return real_read_bounded_json(
            path,
            max_bytes=max_bytes,
            label=label,
            expected_sha256=expected_sha256,
        )

    monkeypatch.setattr(
        sandboxed_split,
        "_read_bounded_json",
        observe_bounded_read,
    )

    run_sandboxed_official_split(
        incident_file=incident_file,
        expected_sha256=_sha256(incident_file),
        expected_content_digest=sequence_content_digest(samples),
        output_directory=prepared_root / "Synthetic",
        raw_release_root=raw_release_root,
        region="Synthetic",
        node_count=3,
        expected_sample_count=20,
        description_count=1,
        type_count=1,
        timeout_seconds=30,
        support_files=support_files,
        release_manifest_file=release_manifest_file,
        expected_release_manifest_sha256=release_manifest_sha256,
    )

    assert labels == [
        "Locked release manifest",
        "writer sandbox probe",
        "postflight sandbox probe",
        "Split worker manifest",
        "Independent split postflight",
    ]


def test_atomic_directory_publish_never_replaces_a_racing_destination(tmp_path):
    staging = tmp_path / ".bundle.stage"
    staging.mkdir()
    (staging / "payload").write_text("prepared", encoding="utf-8")
    destination = tmp_path / "bundle"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        sandboxed_split._atomic_publish_directory_no_clobber(
            staging,
            destination,
        )

    assert (staging / "payload").read_text(encoding="utf-8") == "prepared"
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_postflight_report_checks_output_identity_before_truncating(
    monkeypatch,
    tmp_path,
):
    from scripts.reproduction.igstgnn_table4 import split_postflight_worker

    output = tmp_path / "postflight.json"
    output.touch(mode=0o600)
    replacement_content = b"must-not-be-truncated"
    real_open = split_postflight_worker.os.open
    replaced = False

    def replace_before_open(path, flags, *args):
        nonlocal replaced
        if Path(path) == output and flags & os.O_WRONLY and not replaced:
            replaced = True
            output.unlink()
            output.write_bytes(replacement_content)
            os.chmod(output, 0o600)
        return real_open(path, flags, *args)

    monkeypatch.setattr(split_postflight_worker.os, "open", replace_before_open)

    with pytest.raises(RuntimeError, match="identity changed"):
        split_postflight_worker._write_precreated_report(
            output,
            {"status": "test"},
        )

    assert output.read_bytes() == replacement_content


def test_sandboxed_split_never_publishes_an_unsafe_support_bundle(tmp_path):
    samples = np.asarray([_sample(index) for index in range(20)], dtype=object)
    raw_release_root = tmp_path / "raw"
    region_directory = raw_release_root / "Synthetic"
    region_directory.mkdir(parents=True)
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    incident_file = region_directory / "incident_all.npy"
    np.save(incident_file, samples)
    support_files = _write_valid_support_files(
        region_directory,
        node_count=3,
    )
    unsafe_adjacency = np.zeros((3, 3), dtype=np.float64)
    unsafe_adjacency[0, 1] = -1
    np.save(region_directory / "adj_matrix.npy", unsafe_adjacency)
    support_files["adj_matrix.npy"] = (
        region_directory / "adj_matrix.npy",
        _sha256(region_directory / "adj_matrix.npy"),
    )
    release_manifest_file = tmp_path / "release_manifest.safe_structure.json"
    release_manifest_sha256 = _write_locked_release_manifest(
        release_manifest_file,
        region="Synthetic",
        region_directory=region_directory,
        sample_count=20,
        node_count=3,
    )
    output_directory = prepared_root / "Synthetic"

    with pytest.raises(ValueError, match="negative values"):
        run_sandboxed_official_split(
            incident_file=incident_file,
            expected_sha256=_sha256(incident_file),
            expected_content_digest=sequence_content_digest(samples),
            output_directory=output_directory,
            raw_release_root=raw_release_root,
            region="Synthetic",
            node_count=3,
            expected_sample_count=20,
            description_count=1,
            type_count=1,
            timeout_seconds=30,
            support_files=support_files,
            release_manifest_file=release_manifest_file,
            expected_release_manifest_sha256=release_manifest_sha256,
        )

    assert not output_directory.exists()


def test_official_overlap_is_explicitly_labeled_after_postflight(tmp_path):
    records = [_sample(index) for index in range(20)]
    records[14] = records[0]
    samples = np.asarray(records, dtype=object)
    raw_release_root = tmp_path / "raw"
    region_directory = raw_release_root / "Synthetic"
    region_directory.mkdir(parents=True)
    prepared_root = tmp_path / "prepared"
    prepared_root.mkdir()
    incident_file = region_directory / "incident_all.npy"
    np.save(incident_file, samples)
    support_files = _write_valid_support_files(
        region_directory,
        node_count=3,
    )
    release_manifest_file = tmp_path / "release_manifest.safe_structure.json"
    release_manifest_sha256 = _write_locked_release_manifest(
        release_manifest_file,
        region="Synthetic",
        region_directory=region_directory,
        sample_count=20,
        node_count=3,
    )

    manifest = run_sandboxed_official_split(
        incident_file=incident_file,
        expected_sha256=_sha256(incident_file),
        expected_content_digest=sequence_content_digest(samples),
        output_directory=prepared_root / "Synthetic",
        raw_release_root=raw_release_root,
        region="Synthetic",
        node_count=3,
        expected_sample_count=20,
        description_count=1,
        type_count=1,
        timeout_seconds=30,
        support_files=support_files,
        release_manifest_file=release_manifest_file,
        expected_release_manifest_sha256=release_manifest_sha256,
    )

    assert manifest["split_integrity_status"] == ("cross_split_duplicates_detected")
    assert manifest["validity_label"] == ("official_protocol_with_known_overlap")


def test_final_manifest_checks_identity_before_rewriting(monkeypatch, tmp_path):
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text('{"status": "worker"}\n', encoding="utf-8")
    os.chmod(manifest_path, 0o600)
    replacement_content = b"must-not-be-truncated"
    real_open = sandboxed_split.os.open
    replaced = False

    def replace_before_open(path, flags, *args):
        nonlocal replaced
        if Path(path) == manifest_path and flags & os.O_WRONLY and not replaced:
            replaced = True
            manifest_path.unlink()
            manifest_path.write_bytes(replacement_content)
            os.chmod(manifest_path, 0o600)
        return real_open(path, flags, *args)

    monkeypatch.setattr(sandboxed_split.os, "open", replace_before_open)

    with pytest.raises(RuntimeError, match="identity changed"):
        sandboxed_split._rewrite_owned_json_file(
            manifest_path,
            {"status": "prepared"},
        )

    assert manifest_path.read_bytes() == replacement_content
