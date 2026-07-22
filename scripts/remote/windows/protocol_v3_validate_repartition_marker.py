#!/usr/bin/env python3
"""Validate the relay attestation for the 5080 repartition handoff."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Sequence


EXPECTED_STRINGS = {
    "status": "ready",
    "plan": "27_18",
    "boundary_task": "039_fullwindow_staeformer_base_alameda_seed42",
    "forbidden_task": "040_fullwindow_graphwavenet_base_contracosta_seed42",
}
BOUNDARY_DIGEST_FIELDS = (
    "boundary_status_sha256",
    "boundary_summary_sha256",
    "boundary_state_sha256",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class MarkerError(ValueError):
    """Raised when the boundary marker does not satisfy the handoff contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise MarkerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise MarkerError(f"non-finite JSON value: {value}")


def load_marker(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except MarkerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MarkerError(f"cannot read strict marker JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MarkerError("marker must be a JSON object")
    return payload


def validate_marker(payload: dict[str, object]) -> None:
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise MarkerError("schema_version must be integer 1")
    for field, expected in EXPECTED_STRINGS.items():
        actual = payload.get(field)
        if type(actual) is not str or actual != expected:
            raise MarkerError(f"{field} does not match the repartition contract")
    for field in BOUNDARY_DIGEST_FIELDS:
        digest = payload.get(field)
        if type(digest) is not str or SHA256_RE.fullmatch(digest) is None:
            raise MarkerError(f"{field} must be a lowercase SHA-256 digest")
    for field in ("old_runner_pid", "old_finalizer_pid"):
        pid = payload.get(field)
        if type(pid) is not int or pid <= 0:
            raise MarkerError(f"{field} must be a positive integer")
    generated_at = payload.get("generated_at")
    if type(generated_at) is not str or not generated_at.strip():
        raise MarkerError("generated_at must be a nonempty string")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_marker(load_marker(args.marker))
    except MarkerError as exc:
        print(f"BOUNDARY_MARKER_INVALID: {exc}", file=sys.stderr)
        return 2
    print("BOUNDARY_MARKER_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
