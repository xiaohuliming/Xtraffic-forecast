from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.reproduction.igstgnn_table4 import task_runner as runner_module
from scripts.reproduction.igstgnn_table4.manifest import (
    build_table4_manifest,
)
from scripts.reproduction.igstgnn_table4.profiles import load_profile_set
from scripts.reproduction.igstgnn_table4.task_runner import (
    load_table4_task_spec,
    run_table4_task,
)
from scripts.reproduction.igstgnn_table4.runtime_data import TupleLoaderView
from scripts.reproduction.igstgnn_table4.upstream_lock import (
    load_upstream_lock,
    upstream_lock_digest,
)


PROFILE_PATH = runner_module.PROFILE_PATH


def _write_manifest(
    tmp_path: Path,
    *,
    config_hashes: dict[str, str] | None = None,
    source_lock_sha256: str | None = None,
) -> tuple[Path, str]:
    profile_set = load_profile_set(PROFILE_PATH)
    profile_ids = {
        method: profile_set["profiles"][method]["profile_id"]
        for method in profile_set["profiles"]
    }
    task_seeds_by_region = {
        region: {
            method: (
                424242 + region_index
                if method == "bist"
                else profile_set["profiles"][method]["effective"]["seed"]
            )
            for method in profile_set["profiles"]
        }
        for region_index, region in enumerate(("Alameda", "Contra_Costa", "Orange"))
    }
    manifest = build_table4_manifest(
        release_digest="a" * 64,
        source_lock_digest=(
            source_lock_sha256 or upstream_lock_digest(load_upstream_lock())
        ),
        profile_set_id=profile_set["profile_set_id"],
        profile_set_sha256=profile_set["profile_set_sha256"],
        config_hashes=config_hashes or profile_set["config_hashes"],
        profile_ids=profile_ids,
        task_seeds_by_region=task_seeds_by_region,
        split_bundle_manifest_sha256={
            "Alameda": "b" * 64,
            "Contra_Costa": "c" * 64,
            "Orange": "d" * 64,
        },
    )
    path = tmp_path / "matrix_manifest.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_task_spec_binds_the_canonical_manifest_profiles_and_lock(
    tmp_path,
):
    manifest_path, manifest_sha256 = _write_manifest(tmp_path)

    spec = load_table4_task_spec(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        task_id="001_alameda_hl",
    )

    assert spec.task["method"] == "hl"
    assert spec.task["region"] == "Alameda"
    assert spec.profile["profile_id"] == "hl_public_material_v2"
    assert spec.manifest["manifest_sha256"] == manifest_sha256

    with pytest.raises(ValueError, match="Task ID is not in"):
        load_table4_task_spec(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            task_id="999_missing",
        )


def test_load_task_spec_rejects_a_manifest_not_bound_to_current_profiles(
    tmp_path,
):
    profile_set = load_profile_set(PROFILE_PATH)
    wrong = dict(profile_set["config_hashes"])
    wrong["hl"] = "e" * 64
    manifest_path, manifest_sha256 = _write_manifest(
        tmp_path,
        config_hashes=wrong,
    )

    with pytest.raises(ValueError, match="profile config"):
        load_table4_task_spec(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            task_id="001_alameda_hl",
        )


def test_cli_requires_the_out_of_band_manifest_sha256(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        runner_module.main(
            [
                "--manifest",
                str(tmp_path / "matrix_manifest.json"),
                "--task-id",
                "001_alameda_hl",
                "--splits-root",
                str(tmp_path / "splits"),
                "--results-root",
                str(tmp_path / "results"),
                "--device",
                "cpu",
            ]
        )

    assert error.value.code == 2
    assert "--manifest-sha256" in capsys.readouterr().err


def test_runner_refuses_to_overwrite_before_loading_training_data(
    tmp_path,
):
    manifest_path, manifest_sha256 = _write_manifest(tmp_path)
    results_root = tmp_path / "results"
    target = results_root / "tasks" / "001_alameda_hl"
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        run_table4_task(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            task_id="001_alameda_hl",
            splits_root=tmp_path / "must-not-be-read",
            results_root=results_root,
            device="cpu",
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_runner_rejects_a_symlinked_tasks_parent_before_training(
    tmp_path,
):
    manifest_path, manifest_sha256 = _write_manifest(tmp_path)
    results_root = tmp_path / "results"
    outside = tmp_path / "outside"
    results_root.mkdir()
    outside.mkdir()
    (results_root / "tasks").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="tasks directory"):
        run_table4_task(
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            task_id="001_alameda_hl",
            splits_root=tmp_path / "must-not-be-read",
            results_root=results_root,
            device="cpu",
        )

    assert list(outside.iterdir()) == []


class _FakeLoader:
    size = 2
    num_batch = 1
    current_ind = 0

    def shuffle(self):
        return None

    def get_iterator(self):
        return iter(())


class _FakeBundle:
    node_count = 2
    manifest_sha256 = "b" * 64

    def __init__(self):
        self.train_loader = _FakeLoader()
        self.val_loader = _FakeLoader()
        self.test_loader = _FakeLoader()

    def __getitem__(self, key):
        return {
            "train_loader": self.train_loader,
            "val_loader": self.val_loader,
            "test_loader": self.test_loader,
        }[key]


def test_runtime_support_files_are_bound_during_their_own_read(
    tmp_path,
    monkeypatch,
):
    calls = {}
    bindings = {
        "sensors.csv": {
            "expected_bytes": 11,
            "expected_sha256": "1" * 64,
        },
        "incident_stats.npz": {
            "expected_bytes": 22,
            "expected_sha256": "2" * 64,
        },
        "adj_matrix.npy": {
            "expected_bytes": 33,
            "expected_sha256": "3" * 64,
        },
    }
    bundle_dir = tmp_path / "Alameda"
    monkeypatch.setattr(
        runner_module,
        "_read_bound_split_node_count",
        lambda **kwargs: (2, bundle_dir, bindings),
    )

    def sensor_loader(path, **kwargs):
        calls["sensor"] = kwargs
        return (
            {"sensor_type": np.zeros(2)},
            SimpleNamespace(
                sensor_type_size=1,
                surface_size=1,
                roadway_use_size=1,
            ),
        )

    def stats_loader(path, **kwargs):
        calls["stats"] = kwargs
        return {
            "mean": np.float32(1),
            "std": np.float32(2),
            "normalized": True,
        }

    def adjacency_loader(path, **kwargs):
        calls["adjacency"] = kwargs
        return np.eye(2, dtype=np.float32)

    monkeypatch.setattr(
        runner_module,
        "load_official_sensor_info",
        sensor_loader,
    )
    monkeypatch.setattr(
        runner_module,
        "load_official_incident_stats",
        stats_loader,
    )
    monkeypatch.setattr(
        runner_module,
        "load_official_adjacency",
        adjacency_loader,
    )
    monkeypatch.setattr(
        runner_module,
        "load_verified_split_bundle",
        lambda *args, **kwargs: _FakeBundle(),
    )
    profile = load_profile_set(PROFILE_PATH)["profiles"]["hl"]["effective"]

    runner_module._load_verified_runtime_bundle(
        splits_root=tmp_path,
        region="Alameda",
        expected_manifest_sha256="b" * 64,
        profile_effective=profile,
        task_seed=2023,
        include_incident=False,
    )

    assert calls["sensor"] == {
        "node_count": 2,
        **bindings["sensors.csv"],
    }
    assert calls["stats"] == bindings["incident_stats.npz"]
    assert calls["adjacency"] == {
        "node_count": 2,
        **bindings["adj_matrix.npy"],
    }


class _OneBatchLoader(_FakeLoader):
    def get_iterator(self):
        x = np.arange(2 * 12 * 2, dtype=np.float32).reshape(
            2,
            12,
            2,
            1,
        )
        y = np.full((2, 12, 2, 1), 10, dtype=np.float32)
        return iter(
            [
                {
                    "x_data": x,
                    "y_data": y,
                    "sample_indices": np.asarray([0, 1], dtype=np.int64),
                }
            ]
        )


class _HLBundle(_FakeBundle):
    def __init__(self):
        self.train_loader = _OneBatchLoader()
        self.val_loader = _OneBatchLoader()
        self.test_loader = _OneBatchLoader()


class _RepeatLast(torch.nn.Module):
    horizon = 12

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, value, label=None):
        return value[:, -1:, :, :1].expand(-1, 12, -1, -1) * self.weight


class _HLTestEngine:
    def __init__(self, dataloader):
        self.model = _RepeatLast()
        self._dataloader = dataloader
        self.evaluate_calls = []

    @staticmethod
    def _to_tensor(values):
        return [torch.as_tensor(value, dtype=torch.float32) for value in values]

    @staticmethod
    def _to_device(values):
        return values

    @staticmethod
    def _inverse_transform(values):
        return values

    def evaluate(self, mode):
        self.evaluate_calls.append(mode)


@pytest.mark.parametrize(
    ("task_id", "expected_factory", "incident", "expected_seed"),
    [
        ("001_alameda_hl", "largest", False, 2023),
        ("012_alameda_bist", "bist", False, 424242),
        ("013_alameda_igstgnn", "igstgnn", True, 2025),
    ],
)
def test_runner_routes_each_upstream_family_with_the_expected_loader_contract(
    tmp_path,
    monkeypatch,
    task_id,
    expected_factory,
    incident,
    expected_seed,
):
    manifest_path, manifest_sha256 = _write_manifest(tmp_path)
    calls: dict[str, object] = {}
    bundle = _FakeBundle()

    monkeypatch.setattr(
        runner_module,
        "_load_verified_runtime_bundle",
        lambda **kwargs: (
            calls.update(runtime=kwargs)
            or SimpleNamespace(
                bundle=bundle,
                bundle_dir=tmp_path / "splits" / "Alameda",
                adjacency=np.eye(2, dtype=np.float32),
                scaler=object(),
                sensor_dimensions=SimpleNamespace(
                    sensor_type_size=1,
                    surface_size=1,
                    roadway_use_size=1,
                ),
                incident_stats={
                    "mean": np.float32(1),
                    "std": np.float32(2),
                    "normalized": True,
                },
            )
        ),
    )

    def fake_factory(name):
        def build(**kwargs):
            calls["factory"] = name
            calls["factory_args"] = kwargs
            return SimpleNamespace(
                model=object(),
                engine=object(),
                provenance={"factory": name},
            )

        return build

    monkeypatch.setattr(
        runner_module,
        "build_largest_factory",
        fake_factory("largest"),
    )
    monkeypatch.setattr(
        runner_module,
        "build_bist_factory",
        fake_factory("bist"),
    )
    monkeypatch.setattr(
        runner_module,
        "build_igstgnn_factory",
        fake_factory("igstgnn"),
    )
    monkeypatch.setattr(
        runner_module,
        "run_upstream_training",
        lambda **kwargs: calls.update(training=kwargs),
    )
    monkeypatch.setattr(
        runner_module,
        "capture_test_predictions",
        lambda **kwargs: (
            np.ones((1, 12, 2), dtype=np.float32),
            np.full((1, 12, 2), 2, dtype=np.float32),
        ),
    )

    def save_checkpoint(**kwargs):
        target = Path(kwargs["target"])
        target.write_bytes(b"checkpoint\n")
        return target

    monkeypatch.setattr(
        runner_module,
        "save_model_state_checkpoint",
        save_checkpoint,
    )
    monkeypatch.setattr(
        runner_module,
        "write_completed_task_artifacts",
        lambda **kwargs: (calls.update(publish=kwargs) or Path(kwargs["target_dir"])),
    )

    result = run_table4_task(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        task_id=task_id,
        splits_root=tmp_path / "splits",
        results_root=tmp_path / "results",
        device="cpu",
    )

    assert calls["factory"] == expected_factory
    assert calls["runtime"]["include_incident"] is incident
    assert calls["runtime"]["task_seed"] == expected_seed
    assert calls["runtime"]["profile_effective"]["loader_contract"][
        "partial_batch"
    ] == ("retain" if incident else "drop")
    factory_dataloader = calls["factory_args"]["dataloader"]
    if incident:
        assert factory_dataloader is bundle
    else:
        assert isinstance(factory_dataloader["train_loader"], TupleLoaderView)
    if expected_factory == "bist":
        assert calls["factory_args"]["manifest_frozen_seed"] == expected_seed
    assert calls["training"]["method"] == task_id.rsplit("_", 1)[-1]
    assert result == (tmp_path / "results" / "tasks" / task_id)


def test_hl_runner_chain_publishes_checkpoint_predictions_log_and_provenance(
    tmp_path,
    monkeypatch,
):
    manifest_path, manifest_sha256 = _write_manifest(tmp_path)
    bundle = _HLBundle()
    engine_holder = {}

    monkeypatch.setattr(
        runner_module,
        "_load_verified_runtime_bundle",
        lambda **kwargs: SimpleNamespace(
            bundle=bundle,
            bundle_dir=tmp_path / "splits" / "Alameda",
            adjacency=np.eye(2, dtype=np.float32),
            scaler=object(),
            sensor_dimensions=SimpleNamespace(
                sensor_type_size=1,
                surface_size=1,
                roadway_use_size=1,
            ),
            incident_stats={
                "mean": np.float32(1),
                "std": np.float32(2),
                "normalized": True,
            },
        ),
    )

    def build_hl(**kwargs):
        engine = _HLTestEngine(kwargs["dataloader"])
        engine_holder["engine"] = engine
        return SimpleNamespace(
            model=engine.model,
            engine=engine,
            provenance={"source_id": "largest_b7c6ff5"},
        )

    monkeypatch.setattr(
        runner_module,
        "build_largest_factory",
        build_hl,
    )

    target = run_table4_task(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        task_id="001_alameda_hl",
        splits_root=tmp_path / "splits",
        results_root=tmp_path / "results",
        device="cpu",
    )

    assert engine_holder["engine"].evaluate_calls == ["test"]
    assert sorted(path.name for path in target.iterdir()) == [
        "checkpoint.pt",
        "predictions.npz",
        "provenance.json",
        "summary.json",
        "train.log",
    ]
    summary = json.loads((target / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["prediction_shape"] == [2, 12, 2]
    assert provenance["matrix_manifest_sha256"] == manifest_sha256
    assert provenance["factory"]["source_id"] == "largest_b7c6ff5"
    assert provenance["loader"]["contract"]["partial_batch"] == "drop"
    assert provenance["loader"]["batch_size"] == 64
    assert provenance["loader"]["splits"]["test"] == {
        "available_samples": 2,
        "num_batches": 1,
        "consumed_samples": 2,
        "dropped_tail_samples": 0,
    }
    assert provenance["compatibility_patches"] == {
        "declared": [
            "incident_object_split_adapter_v1",
            "verified_split_loader_source_semantics_v2",
            "isolated_artifact_paths_v1",
            "windows_secure_io_compatibility_v1",
        ],
        "applied_by_data_preparation": [],
        "applied_by_loader": [
            "incident_object_split_adapter_v1",
            "verified_split_loader_source_semantics_v2",
        ],
        "applied_by_factory": [],
        "applied_by_result_pipeline": [
            "isolated_artifact_paths_v1",
        ],
        "applied_by_runtime_io": [],
        "not_applicable": [
            "windows_secure_io_compatibility_v1",
        ],
        "unresolved": [],
    }
    assert len(provenance["training_code_sha256"]) == 64
    assert "Captured prediction shape" in (target / "train.log").read_text(
        encoding="utf-8"
    )
    checkpoint = torch.load(
        target / "checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(checkpoint) == {"weight"}
