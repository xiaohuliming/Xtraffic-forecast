from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


POSIX_SECURE_IO_BOUNDARY = "posix_o_nofollow_single_fd_v1"
WINDOWS_COMPATIBILITY_BOUNDARY = "windows_lstat_open_fstat_lstat_no_reparse_v1"


@dataclass(frozen=True)
class VerifiedRegularFile:
    handle: BinaryIO
    stat_result: os.stat_result
    security_boundary: str


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        getattr(value, "st_uid", 0),
        getattr(value, "st_gid", 0),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        getattr(value, "st_file_attributes", 0),
    )


def _is_reparse_point(value: os.stat_result) -> bool:
    flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(value, "st_file_attributes", 0) & flag)


def _validate_regular(
    value: os.stat_result,
    *,
    path: Path,
    purpose: str,
    require_nonempty: bool,
    max_bytes: int | None,
) -> None:
    if (
        not stat.S_ISREG(value.st_mode)
        or _is_reparse_point(value)
        or value.st_nlink != 1
    ):
        raise ValueError(f"{purpose} must be a non-hardlinked regular file: {path}")
    if require_nonempty and value.st_size <= 0:
        raise ValueError(f"{purpose} must be non-empty: {path}")
    if max_bytes is not None and value.st_size > max_bytes:
        raise ValueError(f"{purpose} exceeded its byte limit: {path}")


def _fallback_open_flags() -> int:
    if os.name == "nt" and (
        not hasattr(os, "O_BINARY") or not hasattr(os, "O_NOINHERIT")
    ):
        raise RuntimeError(
            "Windows compatibility boundary requires O_BINARY and O_NOINHERIT"
        )
    return os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)


@contextmanager
def open_verified_regular_file(
    path: Path | str,
    *,
    purpose: str,
    require_nonempty: bool = False,
    max_bytes: int | None = None,
) -> Iterator[VerifiedRegularFile]:
    source = Path(path)
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")

    use_posix_boundary = hasattr(os, "O_NOFOLLOW")
    path_before: os.stat_result | None = None
    if use_posix_boundary:
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        boundary = POSIX_SECURE_IO_BOUNDARY
    else:
        # Windows compatibility boundary. The path is checked on both sides
        # of open because Windows does not expose O_NOFOLLOW.
        path_before = os.lstat(source)
        _validate_regular(
            path_before,
            path=source,
            purpose=purpose,
            require_nonempty=require_nonempty,
            max_bytes=max_bytes,
        )
        flags = _fallback_open_flags()
        boundary = WINDOWS_COMPATIBILITY_BOUNDARY

    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"{purpose} could not be opened safely: {source}") from error
    try:
        handle_before = os.fstat(descriptor)
        _validate_regular(
            handle_before,
            path=source,
            purpose=purpose,
            require_nonempty=require_nonempty,
            max_bytes=max_bytes,
        )
        if path_before is not None:
            path_after_open = os.lstat(source)
            _validate_regular(
                path_after_open,
                path=source,
                purpose=purpose,
                require_nonempty=require_nonempty,
                max_bytes=max_bytes,
            )
            if not (
                _identity(path_before)
                == _identity(handle_before)
                == _identity(path_after_open)
            ):
                raise RuntimeError(
                    f"{purpose} changed across the Windows compatibility "
                    f"boundary: {source}"
                )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            try:
                yield VerifiedRegularFile(
                    handle=handle,
                    stat_result=handle_before,
                    security_boundary=boundary,
                )
            finally:
                handle_after = os.fstat(descriptor)
                if _identity(handle_after) != _identity(handle_before):
                    raise RuntimeError(f"{purpose} changed while being read: {source}")
                if path_before is not None:
                    path_after_read = os.lstat(source)
                    _validate_regular(
                        path_after_read,
                        path=source,
                        purpose=purpose,
                        require_nonempty=require_nonempty,
                        max_bytes=max_bytes,
                    )
                    if _identity(path_after_read) != _identity(handle_before):
                        raise RuntimeError(
                            f"{purpose} path changed while being read "
                            "through the Windows compatibility boundary: "
                            f"{source}"
                        )
    finally:
        os.close(descriptor)


def read_verified_regular_file(
    path: Path | str,
    *,
    purpose: str,
    max_bytes: int,
    require_nonempty: bool = False,
) -> bytes:
    with open_verified_regular_file(
        path,
        purpose=purpose,
        max_bytes=max_bytes,
        require_nonempty=require_nonempty,
    ) as opened:
        payload = opened.handle.read(max_bytes + 1)
        if len(payload) != opened.stat_result.st_size or len(payload) > max_bytes:
            raise ValueError(f"{purpose} could not be read completely: {path}")
        return payload


def verify_secure_directory(
    path: Path | str,
    *,
    purpose: str,
) -> Path:
    source = Path(path)
    before = os.lstat(source)
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or _is_reparse_point(before)
    ):
        raise ValueError(
            f"{purpose} must be a non-symlink, non-reparse directory: {source}"
        )
    if hasattr(os, "O_NOFOLLOW"):
        getuid = getattr(os, "getuid", None)
        if getuid is None:
            raise RuntimeError("POSIX secure directory verification requires os.getuid")
        if getattr(before, "st_uid", None) != getuid():
            raise ValueError(f"{purpose} must be owned by the current user")

    resolved = source.resolve(strict=True)
    after = os.lstat(source)
    if _identity(after) != _identity(before):
        raise RuntimeError(f"{purpose} changed while being verified: {source}")
    return resolved


def secure_io_provenance() -> dict[str, str]:
    return {
        "active_boundary": (
            POSIX_SECURE_IO_BOUNDARY
            if hasattr(os, "O_NOFOLLOW")
            else WINDOWS_COMPATIBILITY_BOUNDARY
        ),
        "posix_boundary": POSIX_SECURE_IO_BOUNDARY,
        "windows_compatibility_boundary": WINDOWS_COMPATIBILITY_BOUNDARY,
    }
