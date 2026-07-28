from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.reproduction.igstgnn_table4.generate_matrix_manifest import (
    create_table4_matrix_manifest,
    main,
)
from scripts.reproduction.igstgnn_table4.manifest import (
    METHODS,
    REGIONS,
    load_table4_manifest,
)
from scripts.reproduction.igstgnn_table4.profiles import load_profile_set
from scripts.reproduction.igstgnn_table4.upstream_lock import (
    load_upstream_lock,
    upstream_lock_digest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    PROJECT_ROOT
    / "config"
    / "reproduction"
    / "igstgnn_table4"
    / "profiles.public_material_v2.json"
)
UPSTREAM_LOCK_PATH = (
    PROJECT_ROOT / "scripts" / "reproduction" / "igstgnn_table4" / "upstreams.lock.json"
)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_release_and_splits(tmp_path: Path) -> tuple[Path, Path]:
    release_path = tmp_path / "release_manifest.safe_structure.json"
    split_root = tmp_path / "official_splits"
    release_regions = {}
    for index, region in enumerate(REGIONS, start=1):
        counts = {
            "train": 10 * index,
            "val": 2 * index,
            "test": 3 * index,
        }
        release_regions[region] = {
            "files": {"incident_all.npy": {"sha256": format(index, "x") * 64}},
            "sample_count": sum(counts.values()),
            "split_counts": counts,
        }
    release = {
        "hash_authority": "locally_observed_not_publisher_signed",
        "protocol_id": "igstgnn_table4_official_v1",
        "regions": release_regions,
        "release_digest": "a" * 64,
        "release_id": "test_release",
        "status": "safe_structure_verified",
    }
    release_path.write_bytes(_json_bytes(release))
    release_sha256 = hashlib.sha256(release_path.read_bytes()).hexdigest()

    for region in REGIONS:
        directory = split_root / region
        directory.mkdir(parents=True)
        split = {
            "schema_version": 2,
            "status": "official_splits_prepared",
            "protocol_id": "igstgnn_table4_official_v1",
            "region": region,
            "validity_label": "official_protocol_with_known_overlap",
            "record_structure_status": "records_structurally_verified",
            "support_structure_status": "safe_structure_verified",
            "counts": release_regions[region]["split_counts"],
            "source_incident_sha256": release_regions[region]["files"][
                "incident_all.npy"
            ]["sha256"],
            "release_manifest_binding": {
                "path": str(release_path),
                "region": region,
                "sha256": release_sha256,
            },
            "independent_postflight": {
                "status": "official_splits_independently_reverified"
            },
        }
        (directory / "split_manifest.json").write_bytes(_json_bytes(split))
    return release_path, split_root


def test_generator_creates_a_bound_v2_matrix_and_freezes_bist_per_region(
    tmp_path,
):
    release_path, split_root = _write_release_and_splits(tmp_path)
    output = tmp_path / "matrix_manifest.json"
    profile_set = load_profile_set(PROFILE_PATH)
    upstream_lock = load_upstream_lock(UPSTREAM_LOCK_PATH)
    torch.manual_seed(731)
    expected_bist_seeds = {
        region: int(torch.randint(999999, (1,)).item()) for region in REGIONS
    }
    expected_next_draw = int(torch.randint(999999, (1,)).item())
    torch.manual_seed(731)

    result = create_table4_matrix_manifest(
        profile_path=PROFILE_PATH,
        upstream_lock_path=UPSTREAM_LOCK_PATH,
        release_manifest_path=release_path,
        split_root=split_root,
        output_path=output,
    )

    payload = output.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    manifest = load_table4_manifest(
        output,
        expected_sha256=observed_sha256,
    )
    assert result == {
        "output_path": str(output.resolve()),
        "manifest_sha256": observed_sha256,
        "bist_seed_by_region": expected_bist_seeds,
    }
    assert manifest["profile_set_id"] == ("igstgnn_table4_public_material_profiles_v2")
    assert manifest["profile_set_sha256"] == profile_set["profile_set_sha256"]
    assert manifest["source_lock_digest"] == upstream_lock_digest(upstream_lock)
    assert manifest["release_digest"] == "a" * 64
    assert manifest["bist_seed_freezes"]["realized_seed_by_region"] == (
        expected_bist_seeds
    )
    assert len(manifest["tasks"]) == 39
    assert {
        task["region"]: task["seed"]
        for task in manifest["tasks"]
        if task["method"] == "bist"
    } == expected_bist_seeds
    assert {
        task["seed"] for task in manifest["tasks"] if task["method"] in METHODS[:11]
    } == {2023}
    assert {
        task["seed"] for task in manifest["tasks"] if task["method"] == "igstgnn"
    } == {2025}
    assert int(torch.randint(999999, (1,)).item()) == expected_next_draw
    assert manifest["split_bundle_manifest_sha256"] == {
        region: hashlib.sha256(
            (split_root / region / "split_manifest.json").read_bytes()
        ).hexdigest()
        for region in REGIONS
    }
    assert payload == _json_bytes(json.loads(payload))


def test_generator_refuses_an_existing_output_before_reading_inputs(
    tmp_path,
):
    output = tmp_path / "matrix_manifest.json"
    output.write_bytes(b"do-not-replace")

    with pytest.raises(FileExistsError, match="already exists"):
        create_table4_matrix_manifest(
            profile_path=tmp_path / "must-not-be-read-profile.json",
            upstream_lock_path=tmp_path / "must-not-be-read-lock.json",
            release_manifest_path=tmp_path / "must-not-be-read-release.json",
            split_root=tmp_path / "must-not-be-read-splits",
            output_path=output,
        )

    assert output.read_bytes() == b"do-not-replace"


def test_cli_prints_only_the_written_manifest_sha256(tmp_path, capsys):
    release_path, split_root = _write_release_and_splits(tmp_path)
    output = tmp_path / "matrix_manifest.json"

    exit_code = main(
        [
            "--profile-set",
            str(PROFILE_PATH),
            "--upstream-lock",
            str(UPSTREAM_LOCK_PATH),
            "--release-manifest",
            str(release_path),
            "--split-root",
            str(split_root),
            "--output",
            str(output),
        ]
    )

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert printed == hashlib.sha256(output.read_bytes()).hexdigest() + "\n"
