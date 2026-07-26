from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dist_net.run_integrity import (
    RunIdentity,
    RunIntegrityError,
    prepare_fresh_output_dir,
    verify_cache_bundle_manifest,
    write_summary_json,
)


def valid_environment(staging_root: Path) -> dict[str, str]:
    return {
        "XTRAFFIC_TASK_ID": "001_event_staeformer_s0a_alameda_seed42",
        "XTRAFFIC_REGION": "Alameda",
        "XTRAFFIC_MODEL": "staeformer_deseason",
        "XTRAFFIC_VARIANT": "s0a",
        "XTRAFFIC_SEED": "42",
        "XTRAFFIC_PROTOCOL_VERSION": "protocol_v3_noleak_dedup",
        "XTRAFFIC_SAMPLING_PROTOCOL": "event",
        "XTRAFFIC_CACHE_SCHEMA_VERSION": "3",
        "XTRAFFIC_CACHE_BUNDLE_ID": "bundle-v3-test",
        "XTRAFFIC_CACHE_BUNDLE_DIGEST": "b" * 64,
        "XTRAFFIC_CODE_SHA": "a" * 40,
        "XTRAFFIC_CONFIG_HASH": "c" * 64,
        "XTRAFFIC_RUN_ATTEMPT": "1",
        "XTRAFFIC_STAGING_ROOT": str(staging_root.resolve()),
    }


def test_run_identity_requires_every_environment_field(tmp_path):
    environment = valid_environment(tmp_path / "stage")
    del environment["XTRAFFIC_CONFIG_HASH"]

    with pytest.raises(RunIntegrityError, match="XTRAFFIC_CONFIG_HASH"):
        RunIdentity.from_environment(environment)


def test_run_identity_parses_and_exposes_exact_summary_metadata(tmp_path):
    environment = valid_environment(tmp_path / "stage")

    identity = RunIdentity.from_environment(environment)

    assert identity.seed == 42
    assert identity.cache_schema_version == 3
    assert identity.run_attempt == 1
    assert identity.staging_root == (tmp_path / "stage").resolve()
    assert identity.summary_metadata() == {
        "task_id": "001_event_staeformer_s0a_alameda_seed42",
        "region": "Alameda",
        "model": "staeformer_deseason",
        "variant": "s0a",
        "seed": 42,
        "protocol_version": "protocol_v3_noleak_dedup",
        "sampling_protocol": "event",
        "cache_schema_version": 3,
        "cache_bundle_id": "bundle-v3-test",
        "cache_bundle_digest": "b" * 64,
        "code_sha": "a" * 40,
        "config_hash": "c" * 64,
    }
    identity.require_expected(
        region="Alameda",
        model="staeformer_deseason",
        variant="s0a",
        seed=42,
        sampling_protocol="event",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("XTRAFFIC_TASK_ID", "../escape", "task"),
        ("XTRAFFIC_SEED", "forty-two", "SEED"),
        ("XTRAFFIC_PROTOCOL_VERSION", "protocol_v2_noleak", "protocol"),
        ("XTRAFFIC_SAMPLING_PROTOCOL", "random", "sampling"),
        ("XTRAFFIC_CACHE_SCHEMA_VERSION", "2", "schema"),
        ("XTRAFFIC_CACHE_BUNDLE_DIGEST", "b" * 63, "digest"),
        ("XTRAFFIC_CODE_SHA", "a" * 39, "code_sha"),
        ("XTRAFFIC_CONFIG_HASH", "c" * 63, "config_hash"),
        ("XTRAFFIC_RUN_ATTEMPT", "0", "attempt"),
        ("XTRAFFIC_STAGING_ROOT", "relative/stage", "staging"),
    ],
)
def test_run_identity_rejects_invalid_environment_values(
    tmp_path,
    field,
    value,
    message,
):
    environment = valid_environment(tmp_path / "stage")
    environment[field] = value

    with pytest.raises(RunIntegrityError, match=message):
        RunIdentity.from_environment(environment)


def test_require_expected_fails_on_trainer_identity_mismatch(tmp_path):
    identity = RunIdentity.from_environment(valid_environment(tmp_path / "stage"))

    with pytest.raises(RunIntegrityError, match="model"):
        identity.require_expected(model="graphwavenet")


def test_prepare_fresh_output_dir_creates_only_inside_staging(tmp_path):
    identity = RunIdentity.from_environment(valid_environment(tmp_path / "stage"))
    output = identity.staging_root / "work" / "result"

    prepared = prepare_fresh_output_dir(output, identity)

    assert prepared == output.resolve()
    assert prepared.is_dir()
    with pytest.raises(RunIntegrityError, match="outside staging"):
        prepare_fresh_output_dir(tmp_path / "elsewhere", identity)


def test_prepare_fresh_output_dir_rejects_nonempty_or_symlink(tmp_path):
    identity = RunIdentity.from_environment(valid_environment(tmp_path / "stage"))
    nonempty = identity.staging_root / "work" / "nonempty"
    nonempty.mkdir(parents=True)
    marker = nonempty / "old.txt"
    marker.write_text("old", encoding="utf-8")

    with pytest.raises(RunIntegrityError, match="nonempty"):
        prepare_fresh_output_dir(nonempty, identity)
    assert marker.read_text(encoding="utf-8") == "old"

    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = identity.staging_root / "linked"
    symlink.parent.mkdir(parents=True, exist_ok=True)
    symlink.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RunIntegrityError, match="symlink"):
        prepare_fresh_output_dir(symlink, identity)


def write_cache_bundle_manifest(
    data_dir: Path,
    identity: RunIdentity,
    **overrides,
) -> Path:
    payload = {
        "status": "verified",
        "protocol_version": identity.protocol_version,
        "cache_schema_version": identity.cache_schema_version,
        "cache_bundle_id": identity.cache_bundle_id,
        "cache_bundle_digest": identity.cache_bundle_digest,
    }
    payload.update(overrides)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "cache_bundle_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_verify_cache_bundle_manifest_requires_exact_verified_identity(tmp_path):
    identity = RunIdentity.from_environment(valid_environment(tmp_path / "stage"))
    data_dir = tmp_path / "cache"
    write_cache_bundle_manifest(data_dir, identity)

    payload = verify_cache_bundle_manifest(data_dir, identity)

    assert payload["status"] == "verified"
    assert payload["cache_bundle_id"] == identity.cache_bundle_id
    assert payload["cache_bundle_digest"] == identity.cache_bundle_digest


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"status": "building"}, "status"),
        ({"protocol_version": "protocol_v2_noleak"}, "protocol"),
        ({"cache_schema_version": 2}, "schema"),
        ({"cache_bundle_id": "other"}, "bundle_id"),
        ({"cache_bundle_digest": "d" * 64}, "digest"),
    ],
)
def test_verify_cache_bundle_manifest_rejects_identity_mismatch(
    tmp_path,
    overrides,
    message,
):
    identity = RunIdentity.from_environment(valid_environment(tmp_path / "stage"))
    data_dir = tmp_path / "cache"
    write_cache_bundle_manifest(data_dir, identity, **overrides)

    with pytest.raises(RunIntegrityError, match=message):
        verify_cache_bundle_manifest(data_dir, identity)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "malformed"),
        ('{"status":"verified","status":"verified"}', "duplicate"),
        ('{"status":"verified","extra":NaN}', "malformed"),
        ("[]", "object"),
    ],
)
def test_verify_cache_bundle_manifest_rejects_malformed_json(
    tmp_path,
    content,
    message,
):
    identity = RunIdentity.from_environment(valid_environment(tmp_path / "stage"))
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    (data_dir / "cache_bundle_manifest.json").write_text(
        content,
        encoding="utf-8",
    )

    with pytest.raises(RunIntegrityError, match=message):
        verify_cache_bundle_manifest(data_dir, identity)


def test_verify_cache_bundle_manifest_rejects_missing_file(tmp_path):
    identity = RunIdentity.from_environment(valid_environment(tmp_path / "stage"))

    with pytest.raises(RunIntegrityError, match="missing"):
        verify_cache_bundle_manifest(tmp_path / "cache", identity)


def test_write_summary_json_attaches_identity_and_valid_count_atomically(tmp_path):
    identity = RunIdentity.from_environment(valid_environment(tmp_path / "stage"))
    output = prepare_fresh_output_dir(
        identity.staging_root / "work" / "result",
        identity,
    )

    path = write_summary_json(
        output / "summary.json",
        {"all": 1.25, "affected": 1.5, "unaffected": 1.2},
        identity=identity,
        valid_count=123,
        required_metrics=("all", "affected", "unaffected"),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["task_id"] == identity.task_id
    assert payload["config_hash"] == identity.config_hash
    assert payload["valid_count"] == 123
    assert payload["all"] == 1.25
    assert not list(output.glob("*.tmp"))


def test_write_summary_json_supports_destination_near_windows_path_limit(
    tmp_path,
    monkeypatch,
):
    target_length = 239
    one_character_destination = (
        tmp_path / "s" / "work" / "result" / "summary.json"
    )
    component_length = (
        target_length - len(os.fspath(one_character_destination)) + 1
    )
    assert 1 <= component_length <= 255
    staging_root = tmp_path / ("s" * component_length)
    identity = RunIdentity.from_environment(
        valid_environment(staging_root)
    )
    output = prepare_fresh_output_dir(
        identity.staging_root / "work" / "result",
        identity,
    )
    destination = output / "summary.json"
    assert len(os.fspath(destination)) == target_length

    original_open = Path.open

    def open_with_legacy_windows_limit(path, *args, **kwargs):
        if len(os.fspath(path)) >= 260:
            raise FileNotFoundError(
                2,
                "simulated legacy Windows path limit",
                os.fspath(path),
            )
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_legacy_windows_limit)

    written = write_summary_json(
        destination,
        {"all": 1.25},
        identity=identity,
        valid_count=123,
        required_metrics=("all",),
    )

    assert json.loads(written.read_text(encoding="utf-8"))["all"] == 1.25


@pytest.mark.parametrize(
    ("summary", "valid_count", "message"),
    [
        ({"all": float("nan")}, 1, "finite"),
        ({"all": 1.0, "region": "Orange"}, 1, "region"),
        ({"affected": 1.0}, 1, "all"),
        ({"all": 1.0}, 0, "valid_count"),
        ({"all": Path("not-json")}, 1, "JSON"),
    ],
)
def test_write_summary_json_rejects_invalid_payload(
    tmp_path,
    summary,
    valid_count,
    message,
):
    identity = RunIdentity.from_environment(valid_environment(tmp_path / "stage"))
    output = prepare_fresh_output_dir(
        identity.staging_root / "work" / "result",
        identity,
    )

    with pytest.raises(RunIntegrityError, match=message):
        write_summary_json(
            output / "summary.json",
            summary,
            identity=identity,
            valid_count=valid_count,
            required_metrics=("all",),
        )
    assert not (output / "summary.json").exists()


def test_write_summary_json_never_overwrites_existing_file(tmp_path):
    identity = RunIdentity.from_environment(valid_environment(tmp_path / "stage"))
    output = prepare_fresh_output_dir(
        identity.staging_root / "work" / "result",
        identity,
    )
    path = output / "summary.json"
    path.write_text("sentinel", encoding="utf-8")

    with pytest.raises(RunIntegrityError, match="already exists"):
        write_summary_json(
            path,
            {"all": 1.0},
            identity=identity,
            valid_count=1,
            required_metrics=("all",),
        )
    assert path.read_text(encoding="utf-8") == "sentinel"
