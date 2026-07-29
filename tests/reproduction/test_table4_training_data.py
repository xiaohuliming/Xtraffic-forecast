import hashlib
import json
import os

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4.training_data import (
    load_verified_split_bundle,
)


RETAIN_LOADER_CONTRACT = {
    "training_shuffle": "global_numpy_permutation",
    "partial_batch": "retain",
}
DROP_LOADER_CONTRACT = {
    "training_shuffle": "global_numpy_permutation",
    "partial_batch": "drop",
}


_SUPPORT_FILES = (
    "adj_matrix.npy",
    "incident_stats.npz",
    "sensors.csv",
    "desc_mapping.json",
    "type_mapping.json",
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(index, *, node_count=3):
    ticks = np.arange(48 + index, 48 + index + 24, dtype=np.float32)
    x_data = np.empty((12, node_count, 3), dtype=np.float32)
    x_data[..., 0] = index
    x_data[..., 1] = (ticks[:12] % 288)[:, None] / 288
    x_data[..., 2] = 4 / 7
    y_data = np.empty((12, node_count, 3), dtype=np.float32)
    y_data[..., 0] = index + 100
    y_data[..., 1] = (ticks[12:] % 288)[:, None] / 288
    y_data[..., 2] = 4 / 7
    return {
        "x_data": x_data,
        "y_data": y_data,
        "incident_features": {
            "Incident Time": float(index) / 100,
            "Description": index % 2,
            "Type": index % 3,
            "Holiday": index % 2,
        },
        "incident_position": 11,
        "incident_distances": np.full(
            (node_count, 3),
            index + 0.25,
            dtype=np.float64,
        ),
        "durations": np.full(12, index + 0.5, dtype=np.float64),
    }


def _write_verified_bundle(tmp_path):
    bundle_dir = tmp_path / "Synthetic"
    bundle_dir.mkdir()
    samples = [_sample(index) for index in range(20)]
    samples[17] = _sample(0)
    split_samples = {
        "train": samples[:14],
        "val": samples[14:17],
        "test": samples[17:],
    }
    split_files = {}
    for split, records in split_samples.items():
        path = bundle_dir / f"incident_{split}.npy"
        np.save(path, np.asarray(records, dtype=object))
        split_files[split] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }

    np.save(
        bundle_dir / "adj_matrix.npy",
        np.zeros((3, 3), dtype=np.float64),
    )
    np.savez(
        bundle_dir / "incident_stats.npz",
        mean=np.float32(1.0),
        std=np.float32(2.0),
        normalized=np.array(True),
    )
    (bundle_dir / "sensors.csv").write_text(
        "order,Sensor Type,Surface,Roadway Use,Road Width,"
        "Design Speed Limit\n"
        "0,loops,Concrete,None,80,70\n"
        "1,loops,Concrete,None,80,70\n"
        "2,loops,Concrete,None,80,70\n",
        encoding="utf-8",
    )
    (bundle_dir / "desc_mapping.json").write_text(
        json.dumps({"a": 0, "b": 1}),
        encoding="utf-8",
    )
    (bundle_dir / "type_mapping.json").write_text(
        json.dumps({"a": 0, "b": 1, "c": 2}),
        encoding="utf-8",
    )
    support_files = {
        name: {
            "path": str(bundle_dir / name),
            "bytes": (bundle_dir / name).stat().st_size,
            "sha256": _sha256(bundle_dir / name),
        }
        for name in _SUPPORT_FILES
    }
    overlap = {"train_val": 0, "train_test": 1, "val_test": 0}
    manifest = {
        "schema_version": 2,
        "protocol_id": "igstgnn_table4_official_v1",
        "status": "official_splits_prepared",
        "region": "Synthetic",
        "record_structure_status": "records_structurally_verified",
        "split_integrity_status": "cross_split_duplicates_detected",
        "validity_label": "official_protocol_with_known_overlap",
        "support_structure_status": "safe_structure_verified",
        "artifact_allowlist": sorted(
            {
                "incident_train.npy",
                "incident_val.npy",
                "incident_test.npy",
                "split_manifest.json",
                *_SUPPORT_FILES,
            }
        ),
        "recombined_matches_input": True,
        "counts": {"train": 14, "val": 3, "test": 3},
        "files": split_files,
        "support_files": support_files,
        "split_semantics": {
            "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "ordered": True,
            "shuffled": False,
            "renormalized": False,
            "deduplicated": False,
        },
        "cross_split_full_record_intersections": overlap,
        "cross_split_xy_window_intersections": overlap,
        "cross_split_model_consumed_record_intersections": overlap,
        "duplicate_model_consumed_record_count": 1,
        "model_consumed_record_multiplicity": {"1": 18, "2": 1},
        "model_consumed_digest": "d" * 64,
        "independent_postflight": {
            "status": "official_splits_independently_reverified",
            "records_revalidated": 20,
            "counts": {"train": 14, "val": 3, "test": 3},
            "cross_split_full_record_intersections": overlap,
            "cross_split_xy_window_intersections": overlap,
            "cross_split_model_consumed_record_intersections": overlap,
            "duplicate_model_consumed_record_count": 1,
            "model_consumed_record_multiplicity": {"1": 18, "2": 1},
            "model_consumed_digest": "d" * 64,
        },
        "safe_structure": {
            "adjacency": {
                "dtype": "float64",
                "finite": True,
                "nonnegative": True,
                "diagonal_zero": True,
            },
            "incident_stats": {
                "mean": 1.0,
                "std": 2.0,
                "normalized": True,
            },
            "mappings": {
                "description_count": 2,
                "type_count": 3,
            },
            "sensors": {"row_count": 3, "order_is_unique": True},
        },
        "release_manifest_binding": {
            "path": "/locked/release_manifest.safe_structure.json",
            "sha256": "a" * 64,
            "region": "Synthetic",
        },
        "security_boundary": {
            "negative_controls": {
                profile: {
                    "status": "sandbox_boundary_verified",
                    "profile": profile,
                    "allowed_input_read": True,
                    "credential_read_denied": True,
                    "neighbor_read_denied": True,
                    "network_denied": True,
                    "fork_denied": True,
                    "spawn_denied": True,
                    "sandbox_profile_sha256": "b" * 64,
                    "probe_worker_sha256": "c" * 64,
                }
                for profile in ("writer", "postflight")
            },
        },
    }
    manifest_path = bundle_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle_dir, _sha256(manifest_path)


def test_verified_bundle_retain_contract_matches_igstgnn_tail_semantics(
    tmp_path,
):
    bundle_dir, manifest_sha256 = _write_verified_bundle(tmp_path)

    bundle = load_verified_split_bundle(
        bundle_dir,
        expected_manifest_sha256=manifest_sha256,
        batch_size=5,
        input_dim=3,
        include_incident=True,
        sensor_info=None,
        seed=2025,
        loader_contract=RETAIN_LOADER_CONTRACT,
    )
    batches = list(bundle.train_loader.get_iterator())

    assert bundle.train_loader.size == 14
    assert bundle.train_loader.num_batch == 3
    assert [batch["x_data"].shape[0] for batch in batches] == [5, 5, 4]
    assert batches[0]["x_data"].shape == (5, 12, 3, 3)
    assert batches[0]["y_data"].shape == (5, 12, 3, 1)
    assert batches[0]["x_data"].dtype == np.float32
    assert batches[0]["y_data"].dtype == np.float32
    assert [
        int(value) for batch in batches for value in batch["x_data"][:, 0, 0, 0]
    ] == list(range(14))
    assert np.all(batches[0]["y_data"][0, ..., 0] == 100)
    assert batches[0]["incident_features"].shape == (5, 4)
    assert batches[0]["incident_position"].shape == (5,)
    assert batches[0]["incident_distances"].shape == (5, 3, 3)
    assert batches[0]["durations"].shape == (5, 12)


def test_verified_bundle_can_retain_verified_target_time_features(tmp_path):
    bundle_dir, manifest_sha256 = _write_verified_bundle(tmp_path)

    bundle = load_verified_split_bundle(
        bundle_dir,
        expected_manifest_sha256=manifest_sha256,
        batch_size=5,
        input_dim=3,
        include_incident=False,
        sensor_info=None,
        seed=42,
        loader_contract=RETAIN_LOADER_CONTRACT,
        target_dim=3,
    )
    first = next(bundle.train_loader.get_iterator())

    assert first["y_data"].shape == (5, 12, 3, 3)
    np.testing.assert_allclose(
        first["y_data"][0, :, 0, 1],
        np.arange(60, 72, dtype=np.float32) / 288,
    )
    np.testing.assert_allclose(first["y_data"][0, :, 0, 2], 4 / 7)


def test_verified_bundle_drop_contract_matches_largest_and_bist_semantics(
    tmp_path,
):
    bundle_dir, manifest_sha256 = _write_verified_bundle(tmp_path)

    bundle = load_verified_split_bundle(
        bundle_dir,
        expected_manifest_sha256=manifest_sha256,
        batch_size=5,
        input_dim=3,
        include_incident=False,
        sensor_info=None,
        seed=2023,
        loader_contract=DROP_LOADER_CONTRACT,
    )
    batches = list(bundle.train_loader.get_iterator())

    assert bundle.train_loader.size == 14
    assert bundle.train_loader.num_batch == 2
    assert [batch["x_data"].shape[0] for batch in batches] == [5, 5]
    assert np.concatenate(
        [batch["sample_indices"] for batch in batches]
    ).tolist() == list(range(10))


def test_verified_bundle_loads_through_the_windows_compatibility_boundary(
    tmp_path,
    monkeypatch,
):
    bundle_dir, manifest_sha256 = _write_verified_bundle(tmp_path)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    bundle = load_verified_split_bundle(
        bundle_dir,
        expected_manifest_sha256=manifest_sha256,
        batch_size=5,
        input_dim=3,
        include_incident=True,
        sensor_info=None,
        seed=2025,
        loader_contract=RETAIN_LOADER_CONTRACT,
    )

    assert bundle.train_loader.size == 14
    assert bundle.val_loader.size == 3
    assert bundle.test_loader.size == 3


def test_verified_bundle_slices_flow_only_for_one_channel_baselines(
    tmp_path,
):
    bundle_dir, manifest_sha256 = _write_verified_bundle(tmp_path)

    bundle = load_verified_split_bundle(
        bundle_dir,
        expected_manifest_sha256=manifest_sha256,
        batch_size=5,
        input_dim=1,
        include_incident=False,
        sensor_info=None,
        seed=2025,
        loader_contract=RETAIN_LOADER_CONTRACT,
    )
    batch = next(bundle.train_loader.get_iterator())

    assert batch["x_data"].shape == (5, 12, 3, 1)
    assert set(batch) == {"x_data", "y_data", "sample_indices"}


def test_training_shuffle_matches_upstream_global_cumulative_permutation(
    tmp_path,
):
    bundle_dir, manifest_sha256 = _write_verified_bundle(tmp_path)
    arguments = {
        "expected_manifest_sha256": manifest_sha256,
        "batch_size": 4,
        "input_dim": 3,
        "include_incident": True,
        "sensor_info": None,
        "seed": 2025,
        "loader_contract": RETAIN_LOADER_CONTRACT,
    }
    np.random.seed(2025)
    expected_first = np.arange(14)[np.random.permutation(14)]
    expected_second = expected_first[np.random.permutation(14)]

    np.random.seed(2025)
    bundle = load_verified_split_bundle(bundle_dir, **arguments)
    bundle.train_loader.shuffle()
    first_order = np.concatenate(
        [batch["sample_indices"] for batch in bundle.train_loader.get_iterator()]
    )
    bundle.train_loader.shuffle()
    second_order = np.concatenate(
        [batch["sample_indices"] for batch in bundle.train_loader.get_iterator()]
    )

    assert np.array_equal(first_order, expected_first)
    assert np.array_equal(second_order, expected_second)
    validation_first = np.concatenate(
        [batch["x_data"][:, 0, 0, 0] for batch in bundle.val_loader.get_iterator()]
    )
    validation_second = np.concatenate(
        [batch["x_data"][:, 0, 0, 0] for batch in bundle.val_loader.get_iterator()]
    )
    assert validation_first.tolist() == [14, 15, 16]
    assert np.array_equal(validation_first, validation_second)
    with pytest.raises(RuntimeError, match="fixed"):
        bundle.val_loader.shuffle()


@pytest.mark.parametrize("mutation", ["extra", "symlink", "hardlink"])
def test_verified_bundle_rejects_unlisted_and_linked_files(
    tmp_path,
    mutation,
):
    bundle_dir, manifest_sha256 = _write_verified_bundle(tmp_path)
    if mutation == "extra":
        (bundle_dir / "unlisted.txt").write_text("unexpected", encoding="utf-8")
    else:
        target = bundle_dir / "sensors.csv"
        external = tmp_path / "external-sensors.csv"
        external.write_bytes(target.read_bytes())
        target.unlink()
        if mutation == "symlink":
            target.symlink_to(external)
        else:
            os.link(external, target)

    with pytest.raises(ValueError, match="extra|safely|hardlinked"):
        load_verified_split_bundle(
            bundle_dir,
            expected_manifest_sha256=manifest_sha256,
            batch_size=4,
            input_dim=3,
            include_incident=True,
            sensor_info=None,
            seed=2025,
            loader_contract=RETAIN_LOADER_CONTRACT,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("manifest_hash", "split_manifest.json SHA-256"),
        ("status", "status"),
        ("validity", "validity_label"),
        ("split_size", "size"),
        ("split_hash", "SHA-256"),
    ],
)
def test_verified_bundle_enforces_manifest_and_split_identity(
    tmp_path,
    mutation,
    message,
):
    bundle_dir, manifest_sha256 = _write_verified_bundle(tmp_path)
    manifest_path = bundle_dir / "split_manifest.json"
    if mutation == "manifest_hash":
        manifest_sha256 = "b" * 64
    elif mutation in {"status", "validity", "split_size"}:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "status":
            manifest["status"] = "pending"
        elif mutation == "validity":
            manifest["validity_label"] = "noleak_verified"
        else:
            manifest["files"]["train"]["bytes"] += 1
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_sha256 = _sha256(manifest_path)
    else:
        split_path = bundle_dir / "incident_train.npy"
        with split_path.open("r+b") as handle:
            handle.seek(-1, os.SEEK_END)
            final_byte = handle.read(1)
            handle.seek(-1, os.SEEK_END)
            handle.write(bytes([final_byte[0] ^ 1]))

    with pytest.raises(ValueError, match=message):
        load_verified_split_bundle(
            bundle_dir,
            expected_manifest_sha256=manifest_sha256,
            batch_size=4,
            input_dim=3,
            include_incident=True,
            sensor_info=None,
            seed=2025,
            loader_contract=RETAIN_LOADER_CONTRACT,
        )


def test_verified_bundle_revalidates_record_time_channels(tmp_path):
    bundle_dir, _ = _write_verified_bundle(tmp_path)
    train_records = [_sample(index) for index in range(14)]
    train_records[4]["x_data"][..., 1] = 1.5
    train_path = bundle_dir / "incident_train.npy"
    np.save(train_path, np.asarray(train_records, dtype=object))
    manifest_path = bundle_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["train"].update(
        {
            "bytes": train_path.stat().st_size,
            "sha256": _sha256(train_path),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="time-of-day"):
        load_verified_split_bundle(
            bundle_dir,
            expected_manifest_sha256=_sha256(manifest_path),
            batch_size=4,
            input_dim=3,
            include_incident=True,
            sensor_info=None,
            seed=2025,
            loader_contract=RETAIN_LOADER_CONTRACT,
        )


def test_verified_bundle_rejects_a_manifest_allowlist_mismatch(tmp_path):
    bundle_dir, _ = _write_verified_bundle(tmp_path)
    manifest_path = bundle_dir / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_allowlist"].append("unapproved.bin")
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact_allowlist"):
        load_verified_split_bundle(
            bundle_dir,
            expected_manifest_sha256=_sha256(manifest_path),
            batch_size=4,
            input_dim=3,
            include_incident=True,
            sensor_info=None,
            seed=2025,
            loader_contract=RETAIN_LOADER_CONTRACT,
        )
