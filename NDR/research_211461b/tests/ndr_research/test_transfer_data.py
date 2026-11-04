from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from ndr_research.transfer_data import (
    FIVE_MINUTES_NS,
    GRAPH_KEYS,
    PROTOCOL,
    REGIONS,
    FrozenBundleContract,
    TestEvaluationGate as EvaluationGate,
    TransferBatchLoader,
    TransferPreprocessing,
    TransferRegionReader,
    _json_hash,
    fit_historical_calibration,
    fit_source_only_pool,
    load_transfer_sources,
)


def freeze_fixture(data: Path, graph: Path) -> FrozenBundleContract:
    reports, counts = {}, {}
    bundle_id = "synthetic-whitelist-fixture"
    for region in REGIONS:
        artifacts = {
            "samples": data / f"{region}_samples.h5",
            "traffic": data / f"{region}_traffic.h5",
            "relation": data / f"{region}_event_rel_feat.npz",
            "graph": graph / f"{GRAPH_KEYS[region]}_sparse_adj.npz",
        }
        hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in artifacts.items()}
        with h5py.File(artifacts["samples"], "r") as handle:
            origins = handle["sample_start"][...]
            split = handle["split"][...]
        counts[region] = {name: int((split == code).sum()) for name, code in (("train", 0), ("val", 1), ("test", 2))}
        counts[region]["total"] = int(origins.size)
        provenance = {
            "bundle_id": bundle_id,
            "traffic_cache_id": f"traffic-{region}",
            "samples_cache_id": f"samples-{region}",
            "relation_cache_id": f"relation-{region}",
            "artifact_sha256": hashes,
        }
        digest = _json_hash({
            "protocol_version": PROTOCOL, "cache_schema_version": 3,
            "bundle_id": bundle_id,
            "traffic_cache_id": provenance["traffic_cache_id"],
            "samples_cache_id": provenance["samples_cache_id"],
            "relation_cache_id": provenance["relation_cache_id"],
            "artifact_sha256": hashes,
        })
        reports[region] = {
            "cache_bundle_digest": digest, "cache_provenance": provenance,
            "sample_count": origins.size, "unique_forecast_origins": origins.size,
            "split_counts": {key: counts[region][key] for key in ("train", "val", "test")},
        }
    digest = _json_hash({
        "protocol_version": PROTOCOL, "cache_schema_version": 3,
        "bundle_id": bundle_id,
        "regions": {region: reports[region]["cache_bundle_digest"] for region in sorted(reports)},
    })
    manifest = {
        "status": "verified", "protocol_version": PROTOCOL, "cache_schema_version": 3,
        "bundle_id": bundle_id, "cache_bundle_digest": digest, "regions": reports,
    }
    (data / "cache_bundle_manifest.json").write_text(json.dumps(manifest, default=int), encoding="utf-8")
    return FrozenBundleContract(bundle_id=bundle_id, bundle_digest=digest, counts=counts)


def make_fixture(tmp_path: Path, *, levels=None, node_counts=None):
    data, graph = tmp_path / "data", tmp_path / "graph"
    data.mkdir()
    graph.mkdir()
    levels = levels or {"Alameda": 10.0, "ContraCosta": 30.0, "Orange": 90000.0}
    node_counts = node_counts or {"Alameda": 3, "ContraCosta": 5, "Orange": 4}
    times = np.datetime64("2024-01-01T00:00", "ns").astype(np.int64) + np.arange(100, dtype=np.int64) * FIVE_MINUTES_NS
    origins = np.array([11, 12, 13, 60, 61, 80, 81], dtype=np.int64)
    split = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int8)
    for region in REGIONS:
        nodes = node_counts[region]
        attributes = {
            "bundle_id": "synthetic-whitelist-fixture", "cache_schema_version": 3,
            "protocol_version": PROTOCOL, "cache_mode": "formal", "git_dirty": False,
            "N": nodes, "T": 100, "T_h": 12, "T_p": 12, "train_end": 60, "val_end": 80,
            "traffic_cache_id": f"traffic-{region}", "samples_cache_id": f"samples-{region}",
        }
        with h5py.File(data / f"{region}_samples.h5", "w") as handle:
            handle.attrs.update(attributes)
            handle.attrs["cache_role"] = "samples"
            handle.create_dataset("sample_start", data=origins)
            handle.create_dataset("split", data=split)
            handle.create_dataset("incident_feat", data=np.full((7, 2, 4), np.nan))
            handle.create_dataset("analysis_labels/affected_mask", data=np.ones((7, nodes), dtype=bool))
        with h5py.File(data / f"{region}_traffic.h5", "w") as handle:
            handle.attrs.update(attributes)
            handle.attrs["cache_role"] = "traffic"
            flow = np.full((100, nodes, 3), levels[region], dtype=np.float32)
            flow[:, :, 1:] = 12345
            handle.create_dataset("flow_series_imputed", data=flow)
            handle.create_dataset("flow_mask", data=np.ones_like(flow, dtype=bool))
            handle.create_dataset("times_ns", data=times)
            handle.create_dataset("region_idx", data=np.arange(nodes, dtype=np.int64) + 100)
            handle.create_dataset("baseline_median", data=np.full((7, 288, nodes), np.nan))
            handle.create_dataset("baseline_scale", data=np.full(nodes, np.nan))
            handle.create_dataset("normalizer", data=np.full(nodes, np.nan))
            handle.create_dataset("static_meta", data=np.full((nodes, 5), np.nan))
        np.savez(data / f"{region}_event_rel_feat.npz", incident=np.full((nodes, 3), np.nan))
        edges = np.stack((np.arange(nodes - 1), np.arange(1, nodes))).astype(np.int64)
        np.savez(graph / f"{GRAPH_KEYS[region]}_sparse_adj.npz", edge_index=edges, region_idx=np.arange(nodes) + 100, core_idx=np.arange(nodes))
    return data, graph, freeze_fixture(data, graph)


def _sources(data, graph, contract):
    return {name: TransferRegionReader(name, data, graph, role="source", contract=contract) for name in REGIONS[:2]}


def test_only_whitelisted_datasets_and_flow_channel_zero_are_interpreted(tmp_path, monkeypatch):
    data, graph, contract = make_fixture(tmp_path)
    accessed = []
    original = h5py.Dataset.__getitem__

    def observe(dataset, selection):
        accessed.append((dataset.name, selection))
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", observe)
    readers = _sources(data, graph, contract)
    preprocessing = fit_source_only_pool(readers)
    target = TransferRegionReader("Orange", data, graph, role="target", contract=contract)
    batch = target.get_history_batch(target.indices("val"), preprocessing, split="val")
    names = {name for name, _ in accessed}
    assert names <= {"/sample_start", "/split", "/times_ns", "/region_idx", "/flow_series_imputed", "/flow_mask"}
    assert all(selection[-1] == 0 for name, selection in accessed if name in ("/flow_series_imputed", "/flow_mask"))
    assert set(batch) == {"flow_hist", "history_mask", "x_baseline", "y_baseline", "time_feat", "node_scale", "sample_start", "edge_index"}
    assert batch["flow_hist"].shape == (2, 4, 12)
    assert batch["node_scale"].shape == (4,)
    assert not target.identity["formal_release"]
    assert target.access_audit()["forbidden_dataset_values_read"] is False
    assert not any("source_fit" in key or "historical_calibration" in key for key in target.access_audit()["counts"])


def test_source_only_template_and_scale_weight_regions_equally_and_exclude_future(tmp_path):
    data, graph, contract = make_fixture(tmp_path, node_counts={"Alameda": 1, "ContraCosta": 9, "Orange": 4})
    for region in REGIONS[:2]:
        with h5py.File(data / f"{region}_traffic.h5", "r+") as handle:
            handle["flow_series_imputed"][60:, :, 0] = 1e8
    contract = freeze_fixture(data, graph)
    readers = _sources(data, graph, contract)
    preprocessing = fit_source_only_pool(readers)
    np.testing.assert_allclose(preprocessing.template, 20.0)
    assert preprocessing.residual_scale == pytest.approx(10.0)
    assert preprocessing.identity["target_statistics_used"] is False
    assert preprocessing.identity["fit_ranges"]["Alameda"]["unique_raw_timestamps"] == 60
    assert all(event["stop"] <= 60 for reader in readers.values() for event in reader.access_audit()["recent_access"] if event["kind"] == "flow_bounds")


def test_cached_target_baseline_poison_cannot_change_source_statistics_or_target_inputs(tmp_path):
    data, graph, contract = make_fixture(tmp_path)
    initial = fit_source_only_pool(_sources(data, graph, contract))
    reader = TransferRegionReader("Orange", data, graph, role="target", contract=contract)
    first = reader.get_history_batch(reader.indices("val"), initial, split="val")
    with h5py.File(data / "Orange_traffic.h5", "r+") as handle:
        for name in ("baseline_median", "baseline_scale", "normalizer", "static_meta"):
            handle[name][...] = -1e20
    contract = freeze_fixture(data, graph)
    repeated = fit_source_only_pool(_sources(data, graph, contract))
    second_reader = TransferRegionReader("Orange", data, graph, role="target", contract=contract)
    second = second_reader.get_history_batch(second_reader.indices("val"), repeated, split="val")
    np.testing.assert_array_equal(initial.template, repeated.template)
    assert initial.residual_scale == repeated.residual_scale
    assert initial.identity["template_sha256"] == repeated.identity["template_sha256"]
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_target_training_and_ungranted_prefix_reads_are_rejected(tmp_path):
    data, graph, contract = make_fixture(tmp_path)
    preprocessing = fit_source_only_pool(_sources(data, graph, contract))
    target = TransferRegionReader("Orange", data, graph, role="target", contract=contract)
    before = target.access_audit()["counts"]
    with pytest.raises(PermissionError):
        list(target.iter_source_training())
    with pytest.raises(PermissionError):
        fit_source_only_pool([target, TransferRegionReader("Alameda", data, graph, role="source", contract=contract)])
    with pytest.raises(PermissionError):
        target.get_batch(target.indices("train"), preprocessing, split="train")
    with pytest.raises(PermissionError):
        target._read_flow(0, 60, purpose="historical_calibration")
    with pytest.raises(PermissionError):
        target._read_flow(0, 24, purpose="window:val")
    assert target.access_audit()["counts"] == before


def test_test_split_requires_explicit_checkpoint_bound_gate(tmp_path):
    data, graph, contract = make_fixture(tmp_path)
    preprocessing = fit_source_only_pool(_sources(data, graph, contract))
    target = TransferRegionReader("Orange", data, graph, role="target", contract=contract)
    with pytest.raises(PermissionError):
        target.get_batch(target.indices("val"), preprocessing, split="val")
    with pytest.raises(PermissionError):
        TransferBatchLoader(target, preprocessing, split="val", batch_size=2, seed=1)
    with pytest.raises(PermissionError):
        target.get_batch(target.indices("test"), preprocessing, split="test")
    with pytest.raises(PermissionError):
        TransferBatchLoader(target, preprocessing, split="test", batch_size=2, seed=1)
    gate = EvaluationGate("synthetic-explicit-test-only", "a" * 64)
    result = target.get_batch(target.indices("test"), preprocessing, split="test", evaluation_gate=gate)
    assert result["sample_start"].tolist() == [80, 81]
    assert target._test_grant is None
    validation = target.get_batch(target.indices("val"), preprocessing, split="val", evaluation_gate=gate)
    assert validation["target"].shape == (2, 4, 12)


def test_historical_calibration_is_explicit_prefix_bounded_and_separately_identified(tmp_path):
    data, graph, contract = make_fixture(tmp_path)
    target = TransferRegionReader("Orange", data, graph, role="target", contract=contract)
    with pytest.raises(PermissionError):
        fit_historical_calibration(target, prefix_end=61, authorization_id="fixture-calibration")
    with pytest.raises(PermissionError):
        fit_historical_calibration(target, prefix_end=40, authorization_id="")
    calibrated = fit_historical_calibration(target, prefix_end=40, authorization_id="fixture-calibration")
    assert calibrated.mode == "historical_calibrated_transfer"
    assert calibrated.template.shape == (7, 288, 4)
    assert calibrated.identity["target_statistics_used"] is True
    assert calibrated.identity["prefix_end"] == 40
    assert target._calibration_grant is None
    target.get_history_batch(target.indices("val"), calibrated, split="val")
    source = TransferRegionReader("Alameda", data, graph, role="source", contract=contract)
    with pytest.raises(ValueError):
        calibrated.validate_reader(source)


def test_masked_flow_sentinels_are_zeroed_and_observed_zero_is_retained(tmp_path):
    data, graph, contract = make_fixture(tmp_path)
    with h5py.File(data / "Orange_traffic.h5", "r+") as handle:
        handle["flow_mask"][49:73, 0, 0] = False
        handle["flow_series_imputed"][49:73, 0, 0] = np.nan
        handle["flow_series_imputed"][50, 1, 0] = 0
    contract = freeze_fixture(data, graph)
    preprocessing = fit_source_only_pool(_sources(data, graph, contract))
    target = TransferRegionReader("Orange", data, graph, role="target", contract=contract)
    result = target.get_history_batch(np.array([3]), preprocessing, split="val")
    assert np.isfinite(result["flow_hist"]).all()
    assert not result["history_mask"][0, 0].any()
    assert result["flow_hist"][0, 0].sum() == 0
    assert result["flow_hist"][0, 1, 1] == 0
    assert result["history_mask"][0, 1, 1]


def test_history_engineering_interface_never_reads_past_the_forecast_origin(tmp_path, monkeypatch):
    data, graph, contract = make_fixture(tmp_path)
    preprocessing = fit_source_only_pool(_sources(data, graph, contract))
    target = TransferRegionReader("Orange", data, graph, role="target", contract=contract)
    selections = []
    original = h5py.Dataset.__getitem__

    def record(dataset, selection):
        if dataset.name in ("/flow_series_imputed", "/flow_mask"):
            selections.append(selection)
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record)
    result = target.get_history_batch(np.array([3]), preprocessing, split="val")
    assert set(result).isdisjoint({"target", "target_mask"})
    assert all(selection[0].start == 49 and selection[0].stop == 61 for selection in selections)
    assert all(selection[-1] == 0 for selection in selections)
    assert not any("window:" in key for key in target.access_audit()["counts"])


def test_resumable_permutation_cursor_and_rng_reproduce_next_batch_and_epoch(tmp_path):
    data, graph, contract = make_fixture(tmp_path)
    readers = _sources(data, graph, contract)
    preprocessing = fit_source_only_pool(readers)
    first = TransferBatchLoader(readers["Alameda"], preprocessing, split="train", batch_size=2, seed=14)
    first.start_epoch(1)
    first.get_next()
    state = first.state_dict()
    restored = TransferBatchLoader(readers["Alameda"], preprocessing, split="train", batch_size=2, seed=999)
    restored.load_state_dict(state)
    expected, actual = first.get_next(), restored.get_next()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    assert first.get_next() is None and restored.get_next() is None
    first.start_epoch(2)
    restored.start_epoch(2)
    np.testing.assert_array_equal(first.permutation, restored.permutation)
    corrupted = copy.deepcopy(state)
    corrupted["cursor"] = 1
    with pytest.raises(ValueError):
        restored.load_state_dict(corrupted)
    corrupted = copy.deepcopy(state)
    corrupted["permutation"][1] = corrupted["permutation"][0]
    with pytest.raises(ValueError):
        restored.load_state_dict(corrupted)


def test_preprocessing_roundtrip_revalidates_array_and_identity_digests(tmp_path):
    data, graph, contract = make_fixture(tmp_path)
    preprocessing = fit_source_only_pool(_sources(data, graph, contract))
    loaded = TransferPreprocessing.from_state(preprocessing.to_state())
    np.testing.assert_array_equal(preprocessing.template, loaded.template)
    assert preprocessing.identity == loaded.identity
    tampered = preprocessing.to_state()
    tampered["template"][0, 0] += 100
    with pytest.raises(ValueError):
        TransferPreprocessing.from_state(tampered)


@pytest.mark.parametrize("mutation", ["duplicate_origin", "cross_split_target", "graph_order", "wrong_counts", "unverified_status"])
def test_formal_identity_and_split_invariants_fail_closed(tmp_path, mutation):
    data, graph, contract = make_fixture(tmp_path)
    if mutation in ("duplicate_origin", "cross_split_target"):
        with h5py.File(data / "Alameda_samples.h5", "r+") as handle:
            handle["sample_start"][1 if mutation == "duplicate_origin" else 4] = 11 if mutation == "duplicate_origin" else 75
        contract = freeze_fixture(data, graph)
    elif mutation == "graph_order":
        np.savez(graph / "alameda_sparse_adj.npz", edge_index=np.array([[0], [1]]), region_idx=np.array([102, 101, 100]))
        contract = freeze_fixture(data, graph)
    elif mutation == "wrong_counts":
        changed = copy.deepcopy(contract.counts)
        changed["Alameda"]["train"] += 1
        contract = FrozenBundleContract(contract.bundle_id, contract.bundle_digest, changed)
    else:
        manifest_path = data / "cache_bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "preliminary"
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        TransferRegionReader("Alameda", data, graph, role="source", contract=contract)


def test_bytes_tamper_rejected_before_any_dataset_values_are_read(tmp_path):
    data, graph, contract = make_fixture(tmp_path)
    with h5py.File(data / "Alameda_traffic.h5", "r+") as handle:
        handle["baseline_median"][0, 0, 0] = 999
    with pytest.raises(ValueError, match="bytes"):
        TransferRegionReader("Alameda", data, graph, contract=contract)


def test_source_loading_never_instantiates_or_reads_heldout_artifacts(tmp_path, monkeypatch):
    data, graph, contract = make_fixture(tmp_path)
    paths = []
    original = h5py.File

    def observe(path, *args, **kwargs):
        paths.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", observe)
    bundle = load_transfer_sources(["Alameda", "ContraCosta"], "Orange", data, graph, 2, 3, contract=contract)
    assert not any("Orange" in path for path in paths)
    assert bundle.identity["heldout_reader_instantiated"] is False
    assert set(bundle.readers) == {"Alameda", "ContraCosta"}
    assert not bundle.identity["target_statistics_used"]
    with pytest.raises(ValueError):
        load_transfer_sources(["Alameda", "Orange"], "Orange", data, graph, 2, 3, contract=contract)


def test_source_channel_zero_preload_matches_lazy_bytes_without_changing_statistical_identity(tmp_path):
    data, graph, contract = make_fixture(tmp_path)
    readers = _sources(data, graph, contract)
    preprocessing = fit_source_only_pool(readers)
    source = readers["Alameda"]
    before_identity = copy.deepcopy(source.identity)
    train_selection = source.indices("train")[:2]
    val_selection = source.indices("val")
    train_lazy = source.get_batch(train_selection, preprocessing, split="train")
    val_lazy = source.get_batch(val_selection, preprocessing, split="val")
    report = source.preload_source_train_val()
    assert report["bytes"] == 80 * 3 * 5
    assert report["prefix_end"] == source.val_end
    assert report["channels"] == [0] and report["role"] == "source"
    assert not source._source_flow_cache.flags.writeable
    assert not source._source_mask_cache.flags.writeable
    assert source.identity == before_identity
    preprocessing.validate_reader(source)
    for split, selection, expected in (("train", train_selection, train_lazy), ("val", val_selection, val_lazy)):
        actual = source.get_batch(selection, preprocessing, split=split)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
            assert actual[key].tobytes() == expected[key].tobytes()
    audit = source.access_audit()
    assert any(key.startswith("memory_slice:") for key in audit["counts"])
    counts = audit["counts"]
    assert source.preload_source_train_val() == report
    assert source.access_audit()["counts"] == counts


def test_target_preload_is_denied_before_h5_values_are_read(tmp_path, monkeypatch):
    data, graph, contract = make_fixture(tmp_path)
    target = TransferRegionReader("Orange", data, graph, role="target", contract=contract)
    before = target.access_audit()

    def forbidden(*args, **kwargs):
        raise AssertionError("target preloading attempted a dataset read")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbidden)
    with pytest.raises(PermissionError):
        target.preload_source_train_val()
    assert target.access_audit() == before
