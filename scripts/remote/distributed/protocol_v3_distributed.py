#!/usr/bin/env python3
"""Run one audited shard of the protocol-v3 canonical 45-task matrix."""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Sequence


def _discover_repo_root(
    *,
    script_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    environment = os.environ if environ is None else environ
    configured = environment.get("XTRAFFIC_REPO_ROOT")
    runner_relative = Path(
        "scripts/remote/windows/protocol_v3_runner.py"
    )
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not (candidate / runner_relative).is_file():
            raise RuntimeError(
                "XTRAFFIC_REPO_ROOT does not contain "
                f"{runner_relative.as_posix()}: {candidate}"
            )
        return candidate

    source = (script_path or Path(__file__)).resolve()
    for candidate in source.parents:
        if (candidate / runner_relative).is_file():
            return candidate
    raise RuntimeError(
        "cannot locate the XTraffic repository from "
        f"{source}; set XTRAFFIC_REPO_ROOT to the repository root"
    )


SCRIPT_REPO_ROOT = _discover_repo_root()
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from scripts.remote.windows import protocol_v3_runner as runner  # noqa: E402


class ShardValidationError(ValueError):
    """Raised when a distributed task plan violates the fixed audit contract."""


@dataclass(frozen=True)
class ShardSpec:
    name: str
    task_numbers: frozenset[int]
    expected_skip_numbers: frozenset[int]
    default_results_root: Path


SHARD_SPECS: Mapping[str, ShardSpec] = {
    "5080": ShardSpec(
        name="5080",
        task_numbers=frozenset(
            {
                *range(1, 5),
                *range(7, 11),
                *range(13, 17),
                *range(19, 31),
                *range(40, 43),
            }
        ),
        expected_skip_numbers=frozenset(
            {
                *range(1, 5),
                *range(7, 11),
                *range(13, 17),
                *range(19, 31),
            }
        ),
        default_results_root=Path(
            "outputs/protocol_v3_noleak_dedup/"
            "remote_runs/minimal_45_5080"
        ),
    ),
    "5090": ShardSpec(
        name="5090",
        task_numbers=frozenset(
            {
                *range(5, 7),
                *range(11, 13),
                *range(17, 19),
                *range(31, 40),
                *range(43, 46),
            }
        ),
        expected_skip_numbers=frozenset(
            {
                *range(5, 7),
                *range(11, 13),
                *range(17, 19),
                *range(31, 40),
            }
        ),
        default_results_root=Path(
            "outputs/protocol_v3_noleak_dedup/"
            "remote_runs/minimal_45_5090"
        ),
    ),
}


DIRECT_COMPARISON_GROUPS: tuple[frozenset[int], ...] = tuple(
    frozenset(group)
    for group in (
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
        (17, 18),
        (19, 20),
        (21, 22),
        (23, 24),
        (25, 26),
        (27, 28),
        (29, 30),
        (31, 32),
        (33, 34),
        (35, 36),
        (37, 38, 39),
        (40, 41, 42),
        (43, 44, 45),
    )
)


def task_number(task: runner.Task) -> int:
    match = re.match(r"^(\d{3})_", task.task_id)
    if match is None:
        raise ShardValidationError(
            f"task id lacks a three-digit canonical prefix: {task.task_id}"
        )
    return int(match.group(1))


def _task_number_map(
    manifest: runner.Manifest,
) -> dict[int, runner.Task]:
    numbered: dict[int, runner.Task] = {}
    for task in manifest.tasks:
        number = task_number(task)
        if number in numbered:
            raise ShardValidationError(
                f"duplicate canonical task number: {number:03d}"
            )
        numbered[number] = task
    expected = set(range(1, 46))
    if set(numbered) != expected:
        raise ShardValidationError(
            "canonical task numbers must equal 001-045; "
            f"missing={sorted(expected - set(numbered))}, "
            f"extra={sorted(set(numbered) - expected)}"
        )
    return numbered


def _validate_manifest_identity(
    canonical: runner.Manifest,
    shard: runner.Manifest,
    shard_name: str,
) -> None:
    identity_fields = (
        "manifest_version",
        "protocol_version",
        "cache_schema_version",
        "default_results_root",
        "default_cache_dir",
        "default_graph_dir",
        "training",
        "source_path",
        "digest",
    )
    changed = [
        field
        for field in identity_fields
        if getattr(shard, field) != getattr(canonical, field)
    ]
    if changed:
        raise ShardValidationError(
            f"shard {shard_name} changed canonical manifest identity: "
            + ", ".join(changed)
        )


def validate_shard_manifests(
    canonical: runner.Manifest,
    shards: Mapping[str, runner.Manifest],
) -> None:
    expected_names = set(SHARD_SPECS)
    if set(shards) != expected_names:
        raise ShardValidationError(
            "shard names must equal 5080 and 5090"
        )
    canonical_by_number = _task_number_map(canonical)
    canonical_tasks = set(canonical.tasks)
    numbers: dict[str, set[int]] = {}
    for name, shard in shards.items():
        _validate_manifest_identity(canonical, shard, name)
        if any(task not in canonical_tasks for task in shard.tasks):
            raise ShardValidationError(
                f"shard {name} contains a non-canonical task"
            )
        shard_numbers = [task_number(task) for task in shard.tasks]
        if len(shard_numbers) != len(set(shard_numbers)):
            raise ShardValidationError(
                f"shard {name} contains duplicate tasks"
            )
        numbers[name] = set(shard_numbers)

    if not numbers["5080"].isdisjoint(numbers["5090"]):
        overlap = sorted(numbers["5080"] & numbers["5090"])
        raise ShardValidationError(
            f"shards overlap on tasks: {overlap}"
        )
    union = numbers["5080"] | numbers["5090"]
    if union != set(canonical_by_number):
        raise ShardValidationError(
            "shards do not cover canonical tasks exactly; "
            f"missing={sorted(set(canonical_by_number) - union)}, "
            f"extra={sorted(union - set(canonical_by_number))}"
        )

    for comparison in DIRECT_COMPARISON_GROUPS:
        owners = {
            name
            for name, shard_numbers in numbers.items()
            if comparison.intersection(shard_numbers)
        }
        if len(owners) != 1:
            raise ShardValidationError(
                "direct comparison group is split across hosts: "
                f"{sorted(comparison)}"
            )

    for name, spec in SHARD_SPECS.items():
        if numbers[name] != set(spec.task_numbers):
            raise ShardValidationError(
                f"shard {name} differs from its fixed task assignment; "
                f"missing={sorted(spec.task_numbers - numbers[name])}, "
                f"extra={sorted(numbers[name] - spec.task_numbers)}"
            )
        canonical_order = [
            number
            for number in canonical_by_number
            if number in spec.task_numbers
        ]
        actual_order = [task_number(task) for task in shards[name].tasks]
        if actual_order != canonical_order:
            raise ShardValidationError(
                f"shard {name} does not preserve canonical task order"
            )


def build_shard_manifests(
    canonical: runner.Manifest,
) -> dict[str, runner.Manifest]:
    _task_number_map(canonical)
    shards = {
        name: replace(
            canonical,
            tasks=tuple(
                task
                for task in canonical.tasks
                if task_number(task) in spec.task_numbers
            ),
        )
        for name, spec in SHARD_SPECS.items()
    }
    validate_shard_manifests(canonical, shards)
    return shards


def load_shard_manifest(
    path: Path | str,
    shard_name: str,
) -> runner.Manifest:
    canonical = runner.load_manifest(
        path,
        require_production_matrix=True,
    )
    return build_shard_manifests(canonical)[shard_name]


def _resolve_from_repo(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", choices=tuple(SHARD_SPECS), required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            SCRIPT_REPO_ROOT
            / "scripts/remote/windows/protocol_v3_manifest.json"
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=SCRIPT_REPO_ROOT,
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--graph-dir", type=Path)
    parser.add_argument("--cache-bundle-id", required=True)
    parser.add_argument("--cache-bundle-digest", required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
    )
    parser.add_argument("--recover-staging-task")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        canonical = runner.load_manifest(
            args.manifest,
            require_production_matrix=True,
        )
        manifest = build_shard_manifests(canonical)[args.shard]
        spec = SHARD_SPECS[args.shard]
        repo_root = args.repo_root.resolve()
        results_value = args.results_root or spec.default_results_root
        cache_value = args.cache_dir or manifest.default_cache_dir
        graph_value = args.graph_dir or manifest.default_graph_dir
        run_inputs = runner.ProtocolV3Inputs(
            results_root=_resolve_from_repo(repo_root, results_value),
            cache_dir=_resolve_from_repo(repo_root, cache_value),
            graph_dir=_resolve_from_repo(repo_root, graph_value),
            cache_bundle_id=args.cache_bundle_id,
            cache_bundle_digest=args.cache_bundle_digest.lower(),
            code_sha=args.code_sha.lower(),
            python_executable=args.python_executable,
        )
    except (runner.ManifestError, ShardValidationError) as exc:
        print(f"CONFIG_FAILED: {exc}", file=sys.stderr, flush=True)
        return 2

    return runner.run_queue(
        manifest,
        run_inputs,
        repo_root=repo_root,
        verify_git=True,
        recover_staging_task=args.recover_staging_task,
    )


if __name__ == "__main__":
    raise SystemExit(main())
