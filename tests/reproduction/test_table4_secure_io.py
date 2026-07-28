from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from scripts.reproduction.igstgnn_table4 import secure_io


def test_windows_compatibility_boundary_reads_a_regular_file_without_nofollow(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"verified payload")
    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)

    payload = secure_io.read_verified_regular_file(
        source,
        purpose="test payload",
        max_bytes=1024,
        require_nonempty=True,
    )

    assert payload == b"verified payload"
    assert (
        secure_io.secure_io_provenance()["active_boundary"]
        == secure_io.WINDOWS_COMPATIBILITY_BOUNDARY
    )


def test_windows_boundary_tolerates_path_and_handle_ctime_precision_difference(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"verified payload")
    real_lstat = secure_io.os.lstat

    def path_stat_with_different_ctime(path):
        value = real_lstat(path)
        attributes = {
            name: getattr(value, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_uid",
                "st_gid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        attributes["st_ctime_ns"] += 1
        attributes["st_file_attributes"] = getattr(
            value,
            "st_file_attributes",
            0,
        )
        return SimpleNamespace(**attributes)

    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(secure_io.os, "lstat", path_stat_with_different_ctime)

    assert (
        secure_io.read_verified_regular_file(
            source,
            purpose="test payload",
            max_bytes=1024,
            require_nonempty=True,
        )
        == b"verified payload"
    )


def test_windows_compatibility_boundary_sets_binary_and_noninherit_flags(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"verified payload")
    binary_flag = 1 << 28
    noninherit_flag = 1 << 29
    observed_flags = []
    real_open = secure_io.os.open

    def observing_open(path, flags):
        observed_flags.append(flags)
        return real_open(path, flags & ~(binary_flag | noninherit_flag))

    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(secure_io.os, "O_BINARY", binary_flag, raising=False)
    monkeypatch.setattr(
        secure_io.os,
        "O_NOINHERIT",
        noninherit_flag,
        raising=False,
    )
    monkeypatch.setattr(secure_io.os, "open", observing_open)

    assert (
        secure_io.read_verified_regular_file(
            source,
            purpose="test payload",
            max_bytes=1024,
            require_nonempty=True,
        )
        == b"verified payload"
    )
    assert observed_flags == [secure_io.os.O_RDONLY | binary_flag | noninherit_flag]


@pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW"),
    reason="POSIX O_NOFOLLOW is unavailable",
)
def test_posix_boundary_keeps_nofollow_symlink_rejection(tmp_path):
    target = tmp_path / "target.bin"
    target.write_bytes(b"outside")
    source = tmp_path / "payload.bin"
    source.symlink_to(target)

    with pytest.raises(ValueError, match="opened safely"):
        secure_io.read_verified_regular_file(
            source,
            purpose="test payload",
            max_bytes=1024,
            require_nonempty=True,
        )


@pytest.mark.skipif(
    not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "getuid"),
    reason="POSIX ownership APIs are unavailable",
)
def test_posix_directory_boundary_keeps_owner_validation(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "checkout"
    source.mkdir()
    different_uid = os.getuid() + 1
    monkeypatch.setattr(
        secure_io.os,
        "getuid",
        lambda: different_uid,
    )

    with pytest.raises(ValueError, match="owned"):
        secure_io.verify_secure_directory(
            source,
            purpose="test checkout",
        )


def test_windows_compatibility_boundary_rejects_a_symbolic_link(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "target.bin"
    target.write_bytes(b"outside")
    source = tmp_path / "payload.bin"
    source.symlink_to(target)
    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(ValueError, match="regular file"):
        secure_io.read_verified_regular_file(
            source,
            purpose="test payload",
            max_bytes=1024,
            require_nonempty=True,
        )


def test_windows_compatibility_boundary_rejects_a_hardlink(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "target.bin"
    target.write_bytes(b"outside")
    source = tmp_path / "payload.bin"
    os.link(target, source)
    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(ValueError, match="non-hardlinked"):
        secure_io.read_verified_regular_file(
            source,
            purpose="test payload",
            max_bytes=1024,
            require_nonempty=True,
        )


def test_windows_compatibility_file_boundary_rejects_a_directory(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "payload.bin"
    source.mkdir()
    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(ValueError, match="regular file"):
        secure_io.read_verified_regular_file(
            source,
            purpose="test payload",
            max_bytes=1024,
            require_nonempty=True,
        )


def test_windows_compatibility_boundary_rejects_a_file_swapped_during_open(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"expected")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"malicious")
    displaced = tmp_path / "displaced.bin"
    real_open = secure_io.os.open

    def swapping_open(path, flags):
        source.rename(displaced)
        replacement.rename(source)
        return real_open(path, flags)

    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(secure_io.os, "open", swapping_open)

    with pytest.raises(RuntimeError, match="Windows compatibility"):
        secure_io.read_verified_regular_file(
            source,
            purpose="test payload",
            max_bytes=1024,
            require_nonempty=True,
        )


def test_windows_compatibility_boundary_rejects_a_file_swapped_after_read(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "payload.bin"
    source.write_bytes(b"expected")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"malicious")
    displaced = tmp_path / "displaced.bin"
    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(RuntimeError, match="changed while being read|path changed"):
        with secure_io.open_verified_regular_file(
            source,
            purpose="test payload",
            max_bytes=1024,
            require_nonempty=True,
        ) as opened:
            assert opened.handle.read() == b"expected"
            source.rename(displaced)
            replacement.rename(source)


def test_windows_compatibility_directory_check_does_not_require_getuid(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "checkout"
    source.mkdir()
    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)

    def unavailable_getuid():
        raise AssertionError("Windows compatibility path called os.getuid")

    monkeypatch.setattr(
        secure_io.os,
        "getuid",
        unavailable_getuid,
        raising=False,
    )

    resolved = secure_io.verify_secure_directory(
        source,
        purpose="test checkout",
    )

    assert resolved == source.resolve()


def test_windows_compatibility_directory_check_rejects_reparse_points(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "checkout"
    source.mkdir()
    real_lstat = secure_io.os.lstat
    real_value = real_lstat(source)
    attributes = {
        name: getattr(real_value, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_uid",
            "st_gid",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
    }
    attributes["st_file_attributes"] = getattr(
        secure_io.stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0x400,
    )

    def reparse_lstat(path):
        if os.fspath(path) == os.fspath(source):
            return SimpleNamespace(**attributes)
        return real_lstat(path)

    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(secure_io.os, "lstat", reparse_lstat)

    with pytest.raises(ValueError, match="reparse"):
        secure_io.verify_secure_directory(
            source,
            purpose="test checkout",
        )


def test_windows_compatibility_directory_check_rejects_a_swapped_directory(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "checkout"
    source.mkdir()
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    displaced = tmp_path / "displaced"
    real_resolve = secure_io.Path.resolve

    def swapping_resolve(path, *, strict=False):
        if path == source:
            source.rename(displaced)
            replacement.rename(source)
        return real_resolve(path, strict=strict)

    monkeypatch.delattr(secure_io.os, "O_NOFOLLOW", raising=False)
    monkeypatch.setattr(secure_io.Path, "resolve", swapping_resolve)

    with pytest.raises(RuntimeError, match="changed while being verified"):
        secure_io.verify_secure_directory(
            source,
            purpose="test checkout",
        )
