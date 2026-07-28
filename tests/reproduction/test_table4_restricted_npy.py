import hashlib
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.lib import format as npy_format

from scripts.reproduction.igstgnn_table4.restricted_npy import (
    load_restricted_object_npy,
)
from scripts.reproduction.igstgnn_table4 import (
    deep_audit_worker,
    sandbox_probe_worker,
    split_postflight_worker,
    split_worker,
)
from scripts.reproduction.igstgnn_table4.split_worker import (
    _require_precreated_manifest,
    _write_precreated_manifest,
)


class _DirectArrayReducer:
    def __reduce__(self):
        return np.ndarray, ((4,), np.dtype("float32"))


class _OversizedArrayStateReducer:
    def __reduce__(self):
        return (
            np._core.multiarray._reconstruct,
            (np.ndarray, (0,), b"b"),
            (
                1,
                (64 * 1024 * 1024 + 1,),
                np.dtype("uint8"),
                False,
                b"",
            ),
        )


class _ForgedDtypeStateReducer:
    def __reduce__(self):
        return (
            np.dtype,
            ("f4", False, True),
            (3, "<", None, None, None, -1, -1, 1),
        )


@pytest.mark.parametrize(
    "worker_module",
    [
        split_worker,
        split_postflight_worker,
        deep_audit_worker,
        sandbox_probe_worker,
    ],
)
def test_posix_workers_fail_closed_when_resource_limits_are_unavailable(
    monkeypatch,
    worker_module,
):
    monkeypatch.setattr(worker_module, "resource", None)

    with pytest.raises(RuntimeError, match="POSIX resource limits"):
        worker_module._apply_resource_limits()


def test_restricted_npy_rejects_direct_ndarray_reduce(tmp_path):
    path = tmp_path / "malicious.npy"
    np.save(path, np.asarray([_DirectArrayReducer()], dtype=object))

    with pytest.raises(
        pickle.UnpicklingError,
        match="direct ndarray construction",
    ):
        load_restricted_object_npy(path)


def test_restricted_npy_rejects_direct_ndarray_newobj(tmp_path):
    path = tmp_path / "malicious-newobj.npy"
    payload = (
        b"\x80\x04"
        b"cnumpy\nndarray\n"
        b"K\x04\x85"
        b"cnumpy\ndtype\n"
        b"\x8c\x02f4\x89\x88\x87R"
        b"\x86\x81."
    )
    with path.open("wb") as handle:
        npy_format.write_array_header_1_0(
            handle,
            {
                "descr": "|O",
                "fortran_order": False,
                "shape": (4,),
            },
        )
        handle.write(payload)

    with pytest.raises(
        pickle.UnpicklingError,
        match="direct ndarray construction",
    ):
        load_restricted_object_npy(path)


def test_restricted_npy_rejects_legacy_inst_constructor(tmp_path):
    path = tmp_path / "malicious-inst.npy"
    payload = b"((I4\ntcnumpy\ndtype\n(S'f4'\nI00\nI01\ntRinumpy\nndarray\n."
    with path.open("wb") as handle:
        npy_format.write_array_header_1_0(
            handle,
            {
                "descr": "|O",
                "fortran_order": False,
                "shape": (4,),
            },
        )
        handle.write(payload)

    with pytest.raises(
        pickle.UnpicklingError,
        match="legacy object construction",
    ):
        load_restricted_object_npy(path)


def test_restricted_npy_accepts_bounded_plain_numpy_records(tmp_path):
    path = tmp_path / "trusted.npy"
    records = np.asarray(
        [
            {
                "x": np.zeros((12, 3, 3), dtype=np.float32),
                "duration": np.ones(12, dtype=np.float64),
                "scalar": np.float32(1.0),
            }
        ],
        dtype=object,
    )
    np.save(path, records)

    loaded = load_restricted_object_npy(path)

    assert loaded.shape == (1,)
    assert loaded[0]["x"].shape == (12, 3, 3)


def test_restricted_npy_rejects_oversized_ndarray_state_before_build(tmp_path):
    path = tmp_path / "oversized.npy"
    np.save(path, np.asarray([_OversizedArrayStateReducer()], dtype=object))

    with pytest.raises(
        pickle.UnpicklingError,
        match="ndarray state exceeds",
    ):
        load_restricted_object_npy(path)


def test_restricted_npy_rejects_noncanonical_dtype_state(tmp_path):
    path = tmp_path / "forged-dtype.npy"
    np.save(path, np.asarray([_ForgedDtypeStateReducer()], dtype=object))

    with pytest.raises(
        pickle.UnpicklingError,
        match="noncanonical NumPy dtype state",
    ):
        load_restricted_object_npy(path)


def test_split_worker_rejects_a_source_that_does_not_match_expected_sha256(
    tmp_path,
):
    incident_file = tmp_path / "incident_all.npy"
    np.save(incident_file, np.asarray([], dtype=object))
    outputs = [
        tmp_path / name
        for name in (
            "train.npy",
            "val.npy",
            "test.npy",
            "manifest.json",
        )
    ]
    for output in outputs:
        output.touch(mode=0o600)
        output.chmod(0o600)
    worker = (
        Path(__file__).parents[2]
        / "scripts/reproduction/igstgnn_table4/split_worker.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--incident-file",
            str(incident_file),
            "--expected-sha256",
            "0" * 64,
            "--train-output",
            str(outputs[0]),
            "--val-output",
            str(outputs[1]),
            "--test-output",
            str(outputs[2]),
            "--manifest-output",
            str(outputs[3]),
            "--region",
            "Synthetic",
            "--node-count",
            "1",
            "--expected-sample-count",
            "0",
            "--description-count",
            "1",
            "--type-count",
            "1",
            "--expected-content-digest",
            "0" * 64,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "Incident input SHA-256 mismatch" in result.stderr


def test_split_worker_hashes_and_loads_from_the_same_open_descriptor(
    monkeypatch,
    tmp_path,
):
    from scripts.reproduction.igstgnn_table4 import split_worker

    incident_file = tmp_path / "incident_all.npy"
    replacement_file = tmp_path / "replacement.npy"
    np.save(
        incident_file,
        np.asarray(
            [{"value": np.asarray([1], dtype=np.float32)}],
            dtype=object,
        ),
    )
    np.save(
        replacement_file,
        np.asarray(
            [{"value": np.asarray([2], dtype=np.float32)}],
            dtype=object,
        ),
    )
    expected_sha256 = hashlib.sha256(incident_file.read_bytes()).hexdigest()
    real_loader = split_worker._RESTRICTED_NPY.load_restricted_object_npy_handle
    observed = {}

    def replace_path_then_load(handle, *, source_size):
        os.replace(replacement_file, incident_file)
        loaded = real_loader(handle, source_size=source_size)
        observed["value"] = float(loaded[0]["value"][0])
        return loaded

    monkeypatch.setattr(
        split_worker._RESTRICTED_NPY,
        "load_restricted_object_npy_handle",
        replace_path_then_load,
    )

    with pytest.raises(RuntimeError, match="changed while being loaded"):
        split_worker._load_verified_incident_file(
            incident_file,
            expected_sha256=expected_sha256,
        )

    assert observed["value"] == 1.0
    assert float(load_restricted_object_npy(incident_file)[0]["value"][0]) == 2.0


def test_split_worker_manifest_checks_identity_before_truncating(
    monkeypatch,
    tmp_path,
):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(b"")
    os.chmod(manifest_path, 0o600)
    expected_identity = _require_precreated_manifest(manifest_path)
    replacement_content = b"must-not-be-truncated"
    original_open = os.open
    replaced = False

    def replacing_open(path, flags, *args, **kwargs):
        nonlocal replaced
        if Path(path) == manifest_path and flags & os.O_WRONLY and not replaced:
            replaced = True
            manifest_path.unlink()
            manifest_path.write_bytes(replacement_content)
            os.chmod(manifest_path, 0o600)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", replacing_open)

    with pytest.raises(RuntimeError, match="identity changed"):
        _write_precreated_manifest(
            manifest_path,
            {"status": "worker"},
            expected_identity=expected_identity,
        )

    assert manifest_path.read_bytes() == replacement_content
