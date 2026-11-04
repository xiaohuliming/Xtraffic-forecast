from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from dist_net.protocol import build_full_window_index
from ndr_research.fullwindow_data import (
    FULLWINDOW_COUNTS,
    FullWindowBatchLoader,
    FullWindowMaterialContract,
    FullWindowRegionReader,
    FullWindowStatistics,
    fit_fullwindow_source_pool,
    fit_fullwindow_statistics,
    legacy_array_digest,
)
from ndr_research.transfer_data import (
    FIVE_MINUTES_NS,
    FORMAL_BUNDLE_ID,
    GRAPH_KEYS,
    PROTOCOL,
    REGIONS,
    _json_hash,
    fit_source_only_pool,
)
from scripts.protocol_v3_event_extension_data import fit_weekly_climatology


def freeze_material(data: Path, graph: Path, steps: int) -> FullWindowMaterialContract:
    reports = {}
    for region in REGIONS:
        hashes = {"traffic": hashlib.sha256((data / f"{region}_traffic.h5").read_bytes()).hexdigest(),
                  "graph": hashlib.sha256((graph / f"{GRAPH_KEYS[region]}_sparse_adj.npz").read_bytes()).hexdigest(),
                  "samples": "a" * 64, "relation": "b" * 64}
        provenance = {"traffic_cache_id": f"traffic-{region}", "samples_cache_id": f"samples-{region}",
                      "relation_cache_id": f"relation-{region}", "artifact_sha256": hashes}
        payload = {"protocol_version": PROTOCOL, "cache_schema_version": 3,
                   "bundle_id": "synthetic-fullwindow-material", **provenance}
        # Deliberately unrelated event counts cannot become full-window counts.
        reports[region] = {"cache_bundle_digest": _json_hash(payload),
                           "cache_provenance": provenance, "sample_count": 1,
                           "split_counts": {"train": 1, "val": 0, "test": 0}}
    digest = _json_hash({"protocol_version": PROTOCOL, "cache_schema_version": 3,
                         "bundle_id": "synthetic-fullwindow-material",
                         "regions": {r: reports[r]["cache_bundle_digest"] for r in sorted(reports)}})
    (data / "cache_bundle_manifest.json").write_text(json.dumps({
        "status": "verified", "protocol_version": PROTOCOL, "cache_schema_version": 3,
        "bundle_id": "synthetic-fullwindow-material", "cache_bundle_digest": digest,
        "regions": reports,
    }))
    return FullWindowMaterialContract("synthetic-fullwindow-material", digest, total_steps=steps)


def make_material(tmp_path: Path, steps: int = 100, *, levels=None):
    data, graph = tmp_path / "data", tmp_path / "graph"
    data.mkdir()
    graph.mkdir()
    levels = levels or {"Alameda": 10, "ContraCosta": 30, "Orange": 90000}
    for region in REGIONS:
        with h5py.File(data / f"{region}_traffic.h5", "w") as handle:
            handle.attrs.update({"bundle_id": "synthetic-fullwindow-material", "cache_schema_version": 3,
                                 "protocol_version": PROTOCOL, "cache_mode": "formal", "cache_role": "traffic",
                                 "git_dirty": False, "N": 2, "T": steps,
                                 "train_end": int(steps * .7), "val_end": int(steps * .85),
                                 "traffic_cache_id": f"traffic-{region}",
                                 "imputation_method": "causal_forward_fill_zero_prefix"})
            flow = np.empty((steps, 2, 3), dtype=np.float32)
            flow[..., 0] = levels[region] + np.arange(steps)[:, None] + np.arange(2)[None, :]
            flow[..., 1:] = np.nan
            handle.create_dataset("flow_series_imputed", data=flow)
            handle.create_dataset("flow_mask", data=np.ones(flow.shape, dtype=bool))
            handle.create_dataset("times_ns", data=np.datetime64("2024-01-01", "ns").astype(np.int64) + np.arange(steps, dtype=np.int64) * FIVE_MINUTES_NS)
            handle.create_dataset("region_idx", data=np.array([17, 23], dtype=np.int64))
            for forbidden in ("baseline_median", "baseline_scale", "static_meta", "affected_mask", "incident_feat"):
                handle.create_dataset(forbidden, data=np.array([np.nan]))
        np.savez(graph / f"{GRAPH_KEYS[region]}_sparse_adj.npz", edge_index=np.array([[0], [1]], dtype=np.int64),
                 region_idx=np.array([17, 23], dtype=np.int64), affected_mask=np.array([True]), incident_feat=np.array([np.nan]))
    return data, graph, freeze_material(data, graph, steps)


def reader(data, graph, contract, region="Alameda", **kwargs):
    return FullWindowRegionReader(region, data, graph, contract=contract, **kwargs)


def test_exact_verified27_index_and_legacy_hash_without_numeric_reads(tmp_path, monkeypatch):
    data, graph, contract = make_material(tmp_path, steps=105120)
    read = []
    original = h5py.Dataset.__getitem__

    def observe(dataset, selection):
        read.append(dataset.name)
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", observe)
    r = reader(data, graph, contract)
    assert r.sampling_identity["counts"] == FULLWINDOW_COUNTS
    assert r.sampling_identity["split_origin_sha256"]["test"] == "63c98716ab1f0e1855c8c013e687be859e60a3976edbef75ba39ee19e10d9f26"
    assert read == ["/times_ns", "/region_idx"]
    for name, code, first, last in [("train", 0, 11, 73571), ("val", 1, 73583, 89339), ("test", 2, 89351, 105107)]:
        origins = r.sample_start[r.split == code]
        assert (origins[0], origins[-1]) == (first, last)
        assert len(origins) == FULLWINDOW_COUNTS[name]
    assert r.identity["material"]["authority_role"] == "raw_traffic_and_static_graph_only"
    assert r.sampling_identity["event_sample_identity_used"] is False
    assert not (data / "Alameda_samples.h5").exists()
    assert not (data / "Alameda_event_rel_feat.npz").exists()


def test_whitelist_channels_and_supervised_split_bounds(tmp_path, monkeypatch):
    data, graph, contract = make_material(tmp_path)
    read = []
    original = h5py.Dataset.__getitem__

    def observe(dataset, selection):
        if dataset.name in ("/flow_series_imputed", "/flow_mask"):
            assert selection[2] == 0
            assert selection[0].stop <= 85
        read.append((dataset.name, selection))
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", observe)
    r = reader(data, graph, contract)
    stats = fit_fullwindow_statistics(r)
    for split in ("train", "val"):
        batch = r.get_batch(r.indices(split)[[0, -1]], stats, split=split)
        assert set(batch) == {"flow_hist", "history_mask", "x_baseline", "y_baseline", "time_feat", "node_scale", "edge_index", "sample_start", "target", "target_mask"}
        assert batch["flow_hist"].shape == batch["target"].shape == (2, 2, 12)
        assert batch["time_feat"].shape == (2, 12, 2)
        assert batch["node_scale"].shape == (2,)
        assert np.isfinite(batch["flow_hist"]).all()
    assert {name for name, _ in read} <= {"/flow_series_imputed", "/flow_mask", "/times_ns", "/region_idx"}
    assert r.access_audit()["test_numeric_values_read"] is False


def test_test_and_target_numeric_access_have_no_gate_bypass(tmp_path):
    data, graph, contract = make_material(tmp_path)
    source = reader(data, graph, contract)
    stats = fit_fullwindow_statistics(source)
    target = reader(data, graph, contract, "Orange", role="target")
    before = target.access_audit()
    with pytest.raises(PermissionError):
        target.get_batch(target.indices("val"), stats, split="val")
    with pytest.raises(PermissionError):
        target.preload_source_train_val()
    with pytest.raises(PermissionError):
        list(target.iter_source_training())
    with pytest.raises(PermissionError):
        fit_fullwindow_statistics(target)
    with pytest.raises(PermissionError):
        target._read_flow(0, 24, purpose="window:train")
    with pytest.raises(PermissionError):
        source.indices("test")
    with pytest.raises(PermissionError):
        source.get_batch(np.flatnonzero(source.split == 2), stats, split="test")
    with pytest.raises(PermissionError):
        source._read_flow(70, 94, purpose="window:test")
    with pytest.raises(PermissionError):
        source._read_flow(0, 85, purpose="source_fit")
    with h5py.File(data / "Orange_traffic.h5", "r") as handle:
        with pytest.raises(PermissionError):
            target._dataset(handle, "flow_series_imputed", (slice(0, 24), slice(None), 0), "window:train")
    with h5py.File(data / "Alameda_traffic.h5", "r") as handle:
        with pytest.raises(PermissionError):
            source._dataset(handle, "flow_series_imputed", (slice(70, 94), slice(None), 0), "window:test")
        with pytest.raises(PermissionError):
            source._dataset(handle, "affected_mask", Ellipsis, "source_fit")
    assert target.access_audit() == before


def test_masks_canonical_imputation_and_observed_zero_are_distinct_policies(tmp_path):
    data, graph, contract = make_material(tmp_path)
    with h5py.File(data / "Alameda_traffic.h5", "r+") as f:
        f["flow_series_imputed"][5, 0, 0] = 14.0
        f["flow_mask"][5, 0, 0] = False
        f["flow_series_imputed"][12, 0, 0] = 0.0
        f["flow_series_imputed"][13, 1, 0] = 22.0
        f["flow_mask"][13, 1, 0] = False
    contract = freeze_material(data, graph, 100)
    canonical = reader(data, graph, contract)
    zero = reader(data, graph, contract, history_policy="observed_zero")
    a = canonical.get_batch(canonical.indices("train")[:1], fit_fullwindow_statistics(canonical), split="train")
    b = zero.get_batch(zero.indices("train")[:1], fit_fullwindow_statistics(zero), split="train")
    assert a["flow_hist"][0, 0, 5] == 14.0
    assert b["flow_hist"][0, 0, 5] == 0.0
    assert not a["history_mask"][0, 0, 5]
    assert a["target"][0, 0, 0] == 0.0 and a["target_mask"][0, 0, 0]
    assert a["target"][0, 1, 1] == 0.0 and not a["target_mask"][0, 1, 1]
    np.testing.assert_array_equal(a["history_mask"], b["history_mask"])
    assert canonical.identity["identity_digest"] != zero.identity["identity_digest"]


def test_weekly_statistics_match_current_ndr_semantics_and_ignore_future(tmp_path):
    data, graph, contract = make_material(tmp_path)
    with h5py.File(data / "Alameda_traffic.h5", "r+") as f:
        f["flow_mask"][3, 0, 0] = False
        f["flow_series_imputed"][3, 0, 0] = 999.0
        f["flow_series_imputed"][70:, :, 0] = 1e8
    contract = freeze_material(data, graph, 100)
    r = reader(data, graph, contract)
    result = fit_fullwindow_statistics(r)
    with h5py.File(data / "Alameda_traffic.h5", "r") as f:
        flow, mask = f["flow_series_imputed"][:70], f["flow_mask"][:70]

    class PlainPrefix:
        train_end, N = 70, 2
        flow_series, flow_mask = flow, mask
        dow, tod = r.dow, r.tod

    expected = fit_weekly_climatology(PlainPrefix())
    assert result.flow_mean == expected.flow_mean
    assert result.flow_std == expected.flow_std
    assert result.residual_std == expected.residual_std
    np.testing.assert_array_equal(result.baseline, expected.baseline)
    np.testing.assert_array_equal(result.residual_std_by_node, expected.residual_std_by_node)
    assert result.identity["unique_raw_timestamps"] == 70
    assert result.identity["valid_points"] == 139
    assert all(event["stop"] <= 70 for event in r.access_audit()["recent_access"] if event["kind"] == "flow_bounds")
    restored = FullWindowStatistics.from_state(result.state_dict())
    assert dict(restored.identity) == dict(result.identity)


def test_source_only_pool_binds_fullwindow_and_rejects_event_identity(tmp_path):
    data, graph, contract = make_material(tmp_path)
    sources = {region: reader(data, graph, contract, region) for region in REGIONS[:2]}
    pooled = fit_fullwindow_source_pool(sources)
    assert pooled.identity["sampling_protocol"] == "full_window"
    assert pooled.identity["target_statistics_used"] is False
    assert pooled.template.shape == (7, 288)
    assert all(row["unique_raw_timestamps"] == 70 for row in pooled.identity["fit_ranges"].values())
    source = sources["Alameda"]
    source.get_batch(source.indices("val")[:1], pooled, split="val")
    event_identity = fit_source_only_pool(sources)
    with pytest.raises(ValueError, match="event or unbound"):
        source.get_batch(source.indices("val")[:1], event_identity, split="val")
    with pytest.raises(PermissionError):
        fit_fullwindow_source_pool([source, reader(data, graph, contract, "Orange", role="target")])
    pooled.identity["sampling_identities"]["Alameda"] = "f" * 64
    with pytest.raises(ValueError, match="digest is inconsistent"):
        source.get_batch(source.indices("val")[:1], pooled, split="val")


def test_identity_and_sampling_cannot_be_relabelled(tmp_path):
    data, graph, contract = make_material(tmp_path)
    r = reader(data, graph, contract)
    identity = r.identity
    identity["sampling"]["counts"]["train"] = 32001
    assert r.sampling_identity["counts"]["train"] == 47
    with pytest.raises(AttributeError):
        r.sample_start = np.array([11])
    with pytest.raises(ValueError):
        r.sample_start.setflags(write=True)
    with pytest.raises(AttributeError):
        r.val_end = 100
    with pytest.raises(ValueError):
        reader(data, graph, contract, val_frac=.1)
    with pytest.raises(ValueError):
        FullWindowMaterialContract(bundle_id=FORMAL_BUNDLE_ID, bundle_digest="f" * 64)
    stats = fit_fullwindow_statistics(r)
    with pytest.raises(TypeError):
        stats.identity["reader_identity_digest"] = "a" * 64


def test_material_hash_and_node_order_admission(tmp_path):
    data, graph, contract = make_material(tmp_path)
    with h5py.File(data / "Alameda_traffic.h5", "r+") as f:
        f["flow_series_imputed"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="traffic bytes"):
        reader(data, graph, contract)
    contract = freeze_material(data, graph, 100)
    np.savez(graph / "alameda_sparse_adj.npz", edge_index=np.array([[0], [1]]), region_idx=np.array([23, 17]))
    contract = freeze_material(data, graph, 100)
    with pytest.raises(ValueError, match="exact traffic node order"):
        reader(data, graph, contract)


def test_preload_preserves_lazy_native_history_and_binds_changed_artifact(tmp_path):
    data, graph, contract = make_material(tmp_path)
    r = reader(data, graph, contract)
    stats = fit_fullwindow_statistics(r)
    before = r.get_batch(r.indices("val"), stats, split="val")
    receipt = r.preload_source_train_val()
    after = r.get_batch(r.indices("val"), stats, split="val")
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert receipt["prefix_end"] == 85
    with h5py.File(data / "Alameda_traffic.h5", "r+") as f:
        f["flow_series_imputed"][0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="changed after"):
        r.get_batch(r.indices("train")[:1], stats, split="train")


def test_loader_exact_resume_partial_tail_and_invalid_cursor(tmp_path):
    data, graph, contract = make_material(tmp_path)
    r = reader(data, graph, contract)
    stats = fit_fullwindow_statistics(r)
    first = FullWindowBatchLoader(r, stats, split="train", batch_size=8, seed=42)
    first.start_epoch(3)
    first.get_next()
    state = first.state_dict()
    rest = [batch["sample_start"] for batch in first.get_iterator()]
    second = FullWindowBatchLoader(r, stats, split="train", batch_size=8, seed=2)
    second.load_state_dict(state)
    repeated = [batch["sample_start"] for batch in second.get_iterator()]
    assert sum(map(len, rest)) == 39
    assert len(rest[-1]) == 7
    for a, b in zip(rest, repeated, strict=True):
        np.testing.assert_array_equal(a, b)
    first.start_epoch()
    second.start_epoch()
    np.testing.assert_array_equal(first.permutation, second.permutation)
    for alteration in [{"cursor": 7}, {"permutation": np.zeros(47, dtype=np.int64)}, {"epoch": -1}]:
        changed = copy.deepcopy(state)
        changed.update(alteration)
        with pytest.raises(ValueError):
            second.load_state_dict(changed)
    with pytest.raises(ValueError):
        FullWindowBatchLoader(r, stats, split="train", batch_size=9, seed=42).load_state_dict(state)
    with pytest.raises(PermissionError):
        FullWindowBatchLoader(r, stats, split="test", batch_size=8, seed=42)


def test_validation_loader_fixed_order_and_non_split_indices_are_rejected(tmp_path):
    data, graph, contract = make_material(tmp_path)
    r = reader(data, graph, contract)
    stats = fit_fullwindow_statistics(r)
    with pytest.raises(ValueError):
        r.get_batch(r.indices("train")[:1], stats, split="val")
    with pytest.raises(ValueError):
        r.get_batch([0, 0], stats, split="train")
    with pytest.raises(ValueError):
        r.get_batch([-1], stats, split="train")
    with pytest.raises(ValueError):
        FullWindowBatchLoader(r, stats, split="val", batch_size=2, seed=42, shuffle=True)
    loader = FullWindowBatchLoader(r, stats, split="val", batch_size=3, seed=42)
    state = loader.state_dict()
    state["permutation"] = state["permutation"][::-1]
    with pytest.raises(ValueError, match="canonical"):
        loader.load_state_dict(state)
    origins = np.concatenate([batch["sample_start"] for batch in loader.get_iterator()])
    np.testing.assert_array_equal(origins, r.sample_start[r.indices("val")])


def test_pure_index_target_disjointness_and_array_hash_framing():
    starts, splits = build_full_window_index(105120, 12, 12, 1, 73584, 89352)
    target_sets = [set((starts[splits == code, None] + np.arange(1, 13)).ravel()) for code in range(3)]
    assert all(not target_sets[a] & target_sets[b] for a, b in ((0, 1), (1, 2), (0, 2)))
    assert (73572 not in starts) and (73582 not in starts)
    assert (89340 not in starts) and (89350 not in starts)
    assert legacy_array_digest(starts[splits == 2]) == "63c98716ab1f0e1855c8c013e687be859e60a3976edbef75ba39ee19e10d9f26"


@pytest.mark.parametrize("method", ["ndr", "shared_graph", "staeformer", "bist"])
def test_synthetic_batch_enters_real_model_forward_without_targets(tmp_path, method):
    import torch

    from ndr_research.shared_graph import SharedGraphNDR
    from ndr_research.strong_references import make_strong_reference
    from scripts.train_staeformer_deseason import STAEformerDeseason

    checkout = Path(__file__).resolve().parents[2] / ".third_party/igstgnn_table4/pristine/bist_dd94adf"
    if method == "bist" and not (checkout / "src/models/bist.py").is_file():
        checkout = Path("/Users/xhlm/Desktop/Study/科研实习/.third_party/igstgnn_table4/pristine/bist_dd94adf")
        if not (checkout / "src/models/bist.py").is_file():
            pytest.skip("locked pristine BiST checkout is not installed")
    data, graph, contract = make_material(tmp_path)
    r = reader(data, graph, contract)
    stats = fit_fullwindow_statistics(r)
    batch = r.get_batch(r.indices("train")[:2], stats, split="train")
    tensors = {key: torch.from_numpy(value) for key, value in batch.items()}
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            if method == "ndr":
                model = STAEformerDeseason(2, 12, 12, hgate=True).eval()
                model.sd_res.fill_(stats.residual_std)
                model.flow_mu.fill_(stats.flow_mean)
                model.flow_sd.fill_(stats.flow_std)
                model.sd_node.copy_(torch.from_numpy(stats.residual_std_by_node.copy()))
                output = model(*(tensors[key] for key in ("flow_hist", "x_baseline", "y_baseline", "time_feat")))
            elif method == "shared_graph":
                model = SharedGraphNDR(hidden_dim=8, num_heads=1).eval()
                output = model(*(tensors[key] for key in ("flow_hist", "x_baseline", "y_baseline", "time_feat", "history_mask", "edge_index", "node_scale")))
            else:
                model = make_strong_reference(method=method, node_count=2,
                                              flow_mean=stats.flow_mean, flow_std=stats.flow_std,
                                              bist_checkout=checkout).eval()
                output = model(tensors["flow_hist"], tensors["time_feat"])
            assert output.shape == (2, 2, 12)
            assert output.device.type == "cpu" and torch.isfinite(output).all()
    finally:
        torch.set_num_threads(old_threads)
