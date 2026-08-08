#!/usr/bin/env python3
"""Read-only composite verification for the published nine-task v3 extension."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from scripts.remote.linux.protocol_v3_event_full33_composite import (
    Full33CompositeError,
    build_extension9_composite_index,
)
from scripts.remote.windows import protocol_v3_runner as runner


EXPECTED_STATUS = "verified_event_extensions_single_host_9"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-matrix-id", required=True)
    parser.add_argument("--expected-host-id", required=True)
    parser.add_argument("--expected-sources", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = build_extension9_composite_index(
            published_root=args.published_root,
            output_path=args.output,
            expected_matrix_id=args.expected_matrix_id,
            expected_host_id=args.expected_host_id,
            expected_sources=args.expected_sources,
        )
    except (Full33CompositeError, OSError, runner.ManifestError) as exc:
        print(f"EVENT_EXTENSION_COMPOSITE_FAILED {exc}", file=sys.stderr)
        return 1
    print(
        "EVENT_EXTENSION_COMPOSITE_OK "
        f"tasks={payload['task_count']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
