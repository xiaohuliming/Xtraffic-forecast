import os
from pathlib import Path

import pytest

from scripts.reproduction.igstgnn_table4.upstream_lock import (
    load_upstream_lock,
    upstream_lock_digest,
    verify_pristine_upstream_checkout,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRISTINE_ROOT = PROJECT_ROOT / ".third_party" / "igstgnn_table4" / "pristine"


def test_upstream_lock_pins_all_three_source_trees_and_archive_hashes():
    lock = load_upstream_lock()
    sources = {source["source_id"]: source for source in lock["sources"]}

    assert sources["igstgnn_64c6e24"]["commit"] == (
        "64c6e243466f791f1b75e50421d63e39577ec9fa"
    )
    assert sources["largest_b7c6ff5"]["commit"] == (
        "b7c6ff5f7f0c10051ed10a319b2cfe58021312f9"
    )
    assert sources["bist_dd94adf"]["commit"] == (
        "dd94adf7721fcbb9e3feb5d1b44040305199a4cc"
    )
    assert all(len(source["tree"]) == 40 for source in sources.values())
    assert all(len(source["archive_sha256"]) == 64 for source in sources.values())
    assert sources["bist_dd94adf"]["license_spdx"] is None
    assert sources["bist_dd94adf"]["vendoring_allowed"] is False


def test_upstream_lock_has_a_canonical_content_digest():
    lock = load_upstream_lock()

    assert upstream_lock_digest(lock) == (
        "71ffa5966dcefe2c15d1d9ffa7d6ada926d37f8f7feaef98f0bc8d02913b04de"
    )


@pytest.mark.parametrize(
    ("source_id", "directory"),
    [
        ("igstgnn_64c6e24", "igstgnn_64c6e24"),
        ("largest_b7c6ff5", "largest_b7c6ff5"),
        ("bist_dd94adf", "bist_dd94adf"),
    ],
)
def test_pristine_upstream_checkouts_match_the_locked_commit_and_tree(
    source_id,
    directory,
):
    verification = verify_pristine_upstream_checkout(
        source_id=source_id,
        checkout=PRISTINE_ROOT / directory,
    )

    assert verification["status"] == "pristine_upstream_verified"
    assert verification["source_id"] == source_id
    assert len(verification["commit"]) == 40
    assert len(verification["tree"]) == 40
    assert verification["git_dirty"] is False


def test_pristine_verifier_rejects_a_checkout_for_the_wrong_source():
    with pytest.raises(ValueError, match="HEAD mismatch"):
        verify_pristine_upstream_checkout(
            source_id="igstgnn_64c6e24",
            checkout=PRISTINE_ROOT / "largest_b7c6ff5",
        )


def test_pristine_verifier_uses_the_windows_directory_boundary_without_getuid(
    monkeypatch,
):
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    def unavailable_getuid():
        raise AssertionError("Windows compatibility path called os.getuid")

    monkeypatch.setattr(os, "getuid", unavailable_getuid, raising=False)

    verification = verify_pristine_upstream_checkout(
        source_id="largest_b7c6ff5",
        checkout=PRISTINE_ROOT / "largest_b7c6ff5",
    )

    assert verification["status"] == "pristine_upstream_verified"
    assert verification["secure_io_boundary"]["active_boundary"] == (
        "windows_lstat_open_fstat_lstat_no_reparse_v1"
    )
