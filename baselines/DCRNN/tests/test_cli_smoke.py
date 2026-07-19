from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

from baselines.DCRNN.train_xtraffic import main


def build_xtraffic_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    data_dir = root / "region_data"
    graph_dir = root / "region_graphs"
    archive_dir = root / "archive"
    out_dir = root / "results"
    data_dir.mkdir()
    graph_dir.mkdir()
    archive_dir.mkdir()

    nodes, steps, channels = 3, 52, 3
    history = horizon = 12
    starts = np.array([11, 12, 13, 25, 26, 38, 39], dtype=np.int64)
    splits = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int8)
    samples = starts.size

    with h5py.File(data_dir / "Alameda_samples.h5", "w") as handle:
        handle.attrs["M_max"] = 1
        handle.attrs["C_e"] = 2
        handle.attrs["N"] = nodes
        handle.attrs["T_h"] = history
        handle.attrs["T_p"] = horizon
        handle.create_dataset("sample_start", data=starts)
        handle.create_dataset("split", data=splits)
        handle.create_dataset("event_idx", data=np.arange(samples, dtype=np.int32))
        handle.create_dataset(
            "anchor_region_idx", data=np.zeros(samples, dtype=np.int32)
        )
        handle.create_dataset("sample_offset", data=np.zeros(samples, dtype=np.int16))
        handle.create_dataset(
            "incident_feat", data=np.zeros((samples, 1, 2), dtype=np.float32)
        )
        handle.create_dataset("incident_mask", data=np.zeros((samples, 1), dtype=bool))
        handle.create_dataset(
            "affected_mask", data=np.zeros((samples, nodes), dtype=bool)
        )
        handle.create_dataset(
            "n_active_incidents", data=np.zeros(samples, dtype=np.int16)
        )

    rng = np.random.default_rng(7)
    traffic = rng.uniform(1.0, 100.0, size=(steps, nodes, channels)).astype(np.float32)
    timestamps = (
        np.datetime64("2023-01-01T00:00") + np.arange(steps) * np.timedelta64(5, "m")
    ).astype("datetime64[ns]")
    with h5py.File(data_dir / "Alameda_traffic.h5", "w") as handle:
        handle.attrs["T"] = steps
        handle.create_dataset("time_enc", data=np.zeros((steps, 5), dtype=np.float32))
        handle.create_dataset(
            "static_meta", data=np.zeros((nodes, 2), dtype=np.float32)
        )
        handle.create_dataset(
            "baseline_median",
            data=np.zeros((2, 288, nodes, channels), dtype=np.float32),
        )
        handle.create_dataset(
            "baseline_scale",
            data=np.ones((2, 288, nodes, channels), dtype=np.float32),
        )
        handle.create_dataset("region_idx", data=np.arange(nodes, dtype=np.int64))
        handle.create_dataset("times_ns", data=timestamps.astype(np.int64))
        handle.create_dataset("flow_series_imputed", data=traffic)
        handle.create_dataset("flow_mask", data=np.ones_like(traffic, dtype=bool))

    edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    np.savez_compressed(
        graph_dir / "alameda_sparse_adj.npz",
        edge_index=edge_index,
        core_idx=np.arange(nodes, dtype=np.int64),
        region_idx=np.arange(nodes, dtype=np.int64),
    )
    adjacency = np.array(
        [[0.0, 1.0, 0.0], [0.5, 0.0, 1.0], [0.0, 0.5, 0.0]],
        dtype=np.float32,
    )
    np.save(archive_dir / "adj_matrix.npy", adjacency)
    return data_dir, graph_dir, archive_dir, out_dir


def test_original_dcrnn_cli_smoke(tmp_path, monkeypatch):
    data_dir, graph_dir, archive_dir, out_dir = build_xtraffic_fixture(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_xtraffic.py",
            "--region",
            "Alameda",
            "--data-dir",
            str(data_dir),
            "--graph-dir",
            str(graph_dir),
            "--archive-dir",
            str(archive_dir),
            "--out-dir",
            str(out_dir),
            "--device",
            "cpu",
            "--seed",
            "42",
            "--smoke",
        ],
    )
    main()

    run_dir = out_dir / "event/official_graph/original_config/Alameda/seed42"
    assert (run_dir / "ckpt_best.pt").exists()
    assert (run_dir / "test_predictions.npz").exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["input_mode"] == "original_flow_time_in_day"
    assert summary["n_test_samples"] == 2
