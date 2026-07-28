import os

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4 import split_artifacts
from scripts.reproduction.igstgnn_table4.restricted_npy import (
    load_restricted_object_npy,
)
from scripts.reproduction.igstgnn_table4.split_artifacts import (
    write_precreated_official_split_files,
)


def _sample(index):
    x_data = np.empty((12, 2, 3), dtype=np.float32)
    x_data[..., 0] = index
    x_data[..., 1] = 24 / 288
    x_data[..., 2] = 1 / 7
    y_data = np.empty((12, 2, 3), dtype=np.float32)
    y_data[..., 0] = index + 1
    y_data[..., 1] = 36 / 288
    y_data[..., 2] = 1 / 7
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
        "incident_distances": np.ones((2, 3), dtype=np.float64),
        "durations": np.ones(12, dtype=np.float64),
    }


def _precreated_outputs(tmp_path):
    outputs = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"incident_{split}.npy"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        outputs[split] = path
    return outputs


def test_split_artifacts_use_exact_official_order_and_round_trip(tmp_path):
    samples = np.asarray([_sample(index) for index in range(20)], dtype=object)
    outputs = _precreated_outputs(tmp_path)

    manifest = write_precreated_official_split_files(samples, outputs=outputs)

    assert manifest["status"] == "official_splits_written_and_reloaded"
    assert manifest["counts"] == {"train": 14, "val": 3, "test": 3}
    assert manifest["recombined_matches_input"] is True
    assert len(manifest["input_content_digest"]) == 64
    assert all(len(item["sha256"]) == 64 for item in manifest["files"].values())

    train = load_restricted_object_npy(outputs["train"])
    validation = load_restricted_object_npy(outputs["val"])
    test = load_restricted_object_npy(outputs["test"])
    assert [record["x_data"][0, 0, 0] for record in train] == list(range(14))
    assert [record["x_data"][0, 0, 0] for record in validation] == [14, 15, 16]
    assert [record["x_data"][0, 0, 0] for record in test] == [17, 18, 19]


def test_split_artifacts_check_output_identity_before_truncating(monkeypatch, tmp_path):
    samples = np.asarray([_sample(index) for index in range(20)], dtype=object)
    outputs = _precreated_outputs(tmp_path)
    target = outputs["train"]
    replacement_content = b"must-not-be-truncated"
    real_open = split_artifacts.os.open
    replaced = False

    def replace_before_open(path, flags, *args):
        nonlocal replaced
        if Path(path) == target and flags & os.O_WRONLY and not replaced:
            replaced = True
            target.unlink()
            target.write_bytes(replacement_content)
            os.chmod(target, 0o600)
        return real_open(path, flags, *args)

    from pathlib import Path

    monkeypatch.setattr(split_artifacts.os, "open", replace_before_open)

    with pytest.raises(RuntimeError, match="identity changed"):
        write_precreated_official_split_files(samples, outputs=outputs)

    assert target.read_bytes() == replacement_content
