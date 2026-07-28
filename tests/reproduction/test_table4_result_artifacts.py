from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.reproduction.igstgnn_table4 import result_artifacts
from scripts.reproduction.igstgnn_table4.metrics import (
    compute_official_table4_metrics,
)
from scripts.reproduction.igstgnn_table4.result_artifacts import (
    validate_completed_task_artifacts,
    write_completed_task_artifacts,
)


TASK = {
    "task_id": "001_alameda_hl",
    "method": "hl",
    "region": "Alameda",
    "seed": None,
    "source_id": "largest_b7c6ff5",
    "profile_id": "hl_public_material_v1",
    "config_sha256": "1" * 64,
    "split_bundle_manifest_sha256": "2" * 64,
    "output_dir": "tasks/001_alameda_hl",
}


def _arrays() -> tuple[np.ndarray, np.ndarray]:
    labels = np.arange(2 * 12 * 3, dtype=np.float32).reshape(2, 12, 3)
    labels[0, 0, 0] = 0
    predictions = labels + np.float32(1.25)
    predictions[0, 0, 0] = 99
    return predictions, labels


def _source_files(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = tmp_path / "source_checkpoint.pt"
    checkpoint.write_bytes(b"locked-checkpoint\n")
    train_log = tmp_path / "source_train.log"
    train_log.write_text("pilot complete\n", encoding="utf-8")
    return checkpoint, train_log


def test_write_and_validate_completed_task_artifacts(tmp_path):
    predictions, labels = _arrays()
    checkpoint, train_log = _source_files(tmp_path)
    target = tmp_path / "tasks" / TASK["task_id"]

    written = write_completed_task_artifacts(
        target_dir=target,
        task=TASK,
        predictions=predictions,
        labels=labels,
        checkpoint_source=checkpoint,
        train_log_source=train_log,
        provenance={
            "protocol_id": "igstgnn_table4_official_v1",
            "training_code_sha256": "3" * 64,
            "host": {"device": "cpu"},
        },
    )

    assert written == target
    assert sorted(path.name for path in target.iterdir()) == [
        "checkpoint.pt",
        "predictions.npz",
        "provenance.json",
        "summary.json",
        "train.log",
    ]
    report = validate_completed_task_artifacts(
        target,
        expected_task=TASK,
    )
    assert report["status"] == "completed_task_artifacts_verified"
    assert report["task_id"] == TASK["task_id"]
    assert report["metrics"] == compute_official_table4_metrics(
        predictions,
        labels,
    )
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))
    assert (
        provenance["secure_io_boundary"]["windows_compatibility_boundary"]
        == "windows_lstat_open_fstat_lstat_no_reparse_v2"
    )

    with np.load(target / "predictions.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"labels", "predictions", "valid_mask"}
        np.testing.assert_array_equal(archive["predictions"], predictions)
        np.testing.assert_array_equal(archive["labels"], labels)
        np.testing.assert_array_equal(
            archive["valid_mask"],
            labels != labels.min(),
        )


def test_writer_refuses_to_replace_an_existing_task_directory(tmp_path):
    predictions, labels = _arrays()
    checkpoint, train_log = _source_files(tmp_path)
    target = tmp_path / "tasks" / TASK["task_id"]
    target.mkdir(parents=True)
    sentinel = target / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        write_completed_task_artifacts(
            target_dir=target,
            task=TASK,
            predictions=predictions,
            labels=labels,
            checkpoint_source=checkpoint,
            train_log_source=train_log,
            provenance={
                "protocol_id": "igstgnn_table4_official_v1",
                "training_code_sha256": "3" * 64,
            },
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_result_chain_uses_the_windows_compatibility_boundary(
    tmp_path,
    monkeypatch,
):
    predictions, labels = _arrays()
    checkpoint, train_log = _source_files(tmp_path)
    target = tmp_path / "tasks" / TASK["task_id"]
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    write_completed_task_artifacts(
        target_dir=target,
        task=TASK,
        predictions=predictions,
        labels=labels,
        checkpoint_source=checkpoint,
        train_log_source=train_log,
        provenance={
            "protocol_id": "igstgnn_table4_official_v1",
            "training_code_sha256": "3" * 64,
        },
    )
    report = validate_completed_task_artifacts(
        target,
        expected_task=TASK,
    )
    provenance = json.loads((target / "provenance.json").read_text(encoding="utf-8"))

    assert report["status"] == "completed_task_artifacts_verified"
    assert provenance["secure_io_boundary"]["active_boundary"] == (
        "windows_lstat_open_fstat_lstat_no_reparse_v2"
    )


def test_result_publish_does_not_load_posix_libc_on_native_windows(
    tmp_path,
    monkeypatch,
):
    predictions, labels = _arrays()
    checkpoint, train_log = _source_files(tmp_path)
    target = tmp_path / "tasks" / TASK["task_id"]
    monkeypatch.setattr(result_artifacts.sys, "platform", "win32")

    def reject_cdll(*args, **kwargs):
        raise AssertionError("ctypes.CDLL must not run on native Windows")

    monkeypatch.setattr(result_artifacts.ctypes, "CDLL", reject_cdll)

    written = write_completed_task_artifacts(
        target_dir=target,
        task=TASK,
        predictions=predictions,
        labels=labels,
        checkpoint_source=checkpoint,
        train_log_source=train_log,
        provenance={
            "protocol_id": "igstgnn_table4_official_v1",
            "training_code_sha256": "3" * 64,
        },
    )

    assert written == target
    assert target.is_dir()


@pytest.mark.parametrize(
    "mutation",
    [
        "prediction",
        "checkpoint",
        "extra_file",
        "summary_identity",
    ],
)
def test_validator_rejects_tampered_or_unlisted_artifacts(tmp_path, mutation):
    predictions, labels = _arrays()
    checkpoint, train_log = _source_files(tmp_path)
    target = tmp_path / "tasks" / TASK["task_id"]
    write_completed_task_artifacts(
        target_dir=target,
        task=TASK,
        predictions=predictions,
        labels=labels,
        checkpoint_source=checkpoint,
        train_log_source=train_log,
        provenance={
            "protocol_id": "igstgnn_table4_official_v1",
            "training_code_sha256": "3" * 64,
        },
    )

    if mutation == "prediction":
        with (target / "predictions.npz").open("ab") as handle:
            handle.write(b"tampered")
    elif mutation == "checkpoint":
        (target / "checkpoint.pt").write_bytes(b"wrong")
    elif mutation == "extra_file":
        (target / "unlisted.txt").write_text("wrong", encoding="utf-8")
    else:
        summary_path = target / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["task"]["region"] = "Orange"
        summary_path.write_text(
            json.dumps(summary, sort_keys=True),
            encoding="utf-8",
        )

    with pytest.raises(ValueError):
        validate_completed_task_artifacts(target, expected_task=TASK)


def test_writer_rejects_nonfinite_predictions_and_mismatched_shapes(tmp_path):
    predictions, labels = _arrays()
    checkpoint, train_log = _source_files(tmp_path)

    bad = predictions.copy()
    bad[0, 1, 1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        write_completed_task_artifacts(
            target_dir=tmp_path / "bad-finite",
            task=TASK,
            predictions=bad,
            labels=labels,
            checkpoint_source=checkpoint,
            train_log_source=train_log,
            provenance={
                "protocol_id": "igstgnn_table4_official_v1",
                "training_code_sha256": "3" * 64,
            },
        )

    with pytest.raises(ValueError, match="same shape"):
        write_completed_task_artifacts(
            target_dir=tmp_path / "bad-shape",
            task=TASK,
            predictions=predictions[:, :-1],
            labels=labels,
            checkpoint_source=checkpoint,
            train_log_source=train_log,
            provenance={
                "protocol_id": "igstgnn_table4_official_v1",
                "training_code_sha256": "3" * 64,
            },
        )
