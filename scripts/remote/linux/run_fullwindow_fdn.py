#!/usr/bin/env python3
"""Launch runner for the full-window FDN comparison matrix.

Bypasses the production-matrix check because this manifest covers 42 tasks
(14 methods x 3 regions) rather than the canonical 45-task event matrix.

Usage:
    python3 scripts/remote/linux/run_fullwindow_fdn.py \
        --shard <0-5> --total-shards 6 \
        --cache-bundle-id bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4 \
        --cache-bundle-digest 854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5 \
        --code-sha <GIT_SHA>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.remote.windows.protocol_v3_runner import (
    load_manifest,
    run_queue,
    Manifest,
)


MANIFEST_PATH = ROOT / "config/protocol_v3/table4_fullwindow_fdn_v1.json"
CACHE_BUNDLE_ID = "bundle-c1b59931697a4c8d8ae7f2e3d9b61ac4"
CACHE_BUNDLE_DIGEST = "854237ec5886ba357d7d0080899e5078506a9b917dc4e1ff2bacde64b5d282e5"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=int, required=True,
                   help="0-indexed shard index for this GPU")
    p.add_argument("--total-shards", type=int, default=6,
                   help="total number of shards (one per GPU)")
    p.add_argument("--cache-bundle-id", default=CACHE_BUNDLE_ID)
    p.add_argument("--cache-bundle-digest", default=CACHE_BUNDLE_DIGEST)
    p.add_argument("--code-sha", required=True,
                   help="git SHA of the training code checkout")
    p.add_argument("--device", default=None,
                   help="override CUDA device, e.g. cuda:0 (default: cuda:<shard>)")
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--graph-dir", type=Path, default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--recover-staging-task", default=None)
    return p.parse_args()


def shard_manifest(manifest: Manifest, shard: int, total: int) -> Manifest:
    """Return a copy of manifest with only this shard's tasks."""
    tasks = [t for i, t in enumerate(manifest.tasks) if i % total == shard]
    return Manifest(
        manifest_version=manifest.manifest_version,
        protocol_version=manifest.protocol_version,
        cache_schema_version=manifest.cache_schema_version,
        default_results_root=manifest.default_results_root,
        default_cache_dir=manifest.default_cache_dir,
        default_graph_dir=manifest.default_graph_dir,
        training=manifest.training,
        tasks=tuple(tasks),
        source_path=manifest.source_path,
        digest=manifest.digest,
    )


def main() -> int:
    args = parse_args()

    manifest = load_manifest(MANIFEST_PATH, require_production_matrix=False)

    # Override device in training block based on shard
    device = args.device or f"cuda:{args.shard}"
    training = dict(manifest.training)
    training["device"] = device
    manifest = Manifest(
        manifest_version=manifest.manifest_version,
        protocol_version=manifest.protocol_version,
        cache_schema_version=manifest.cache_schema_version,
        default_results_root=manifest.default_results_root,
        default_cache_dir=manifest.default_cache_dir,
        default_graph_dir=manifest.default_graph_dir,
        training=training,
        tasks=manifest.tasks,
        source_path=manifest.source_path,
        digest=manifest.digest,
    )

    sharded = shard_manifest(manifest, args.shard, args.total_shards)
    print(
        f"Shard {args.shard}/{args.total_shards}: "
        f"{len(sharded.tasks)} tasks on {device}",
        flush=True,
    )
    for t in sharded.tasks:
        print(f"  {t.task_id}", flush=True)

    from scripts.remote.windows.protocol_v3_runner import (
        ProtocolV3Inputs,
        _resolve_from_repo,
    )

    repo_root = ROOT
    results_value = args.results_root or manifest.default_results_root
    cache_value = args.cache_dir or manifest.default_cache_dir
    graph_value = args.graph_dir or manifest.default_graph_dir

    run_inputs = ProtocolV3Inputs(
        results_root=_resolve_from_repo(repo_root, results_value),
        cache_dir=_resolve_from_repo(repo_root, cache_value),
        graph_dir=_resolve_from_repo(repo_root, graph_value),
        cache_bundle_id=args.cache_bundle_id,
        cache_bundle_digest=args.cache_bundle_digest.lower(),
        code_sha=args.code_sha.lower(),
        python_executable=args.python,
    )

    # Pass the full 42-task ID set so preflight accepts results written by other shards
    all_task_ids = {t.task_id for t in manifest.tasks}

    return run_queue(
        sharded,
        run_inputs,
        repo_root=repo_root,
        verify_git=False,
        recover_staging_task=args.recover_staging_task,
        extra_known_task_ids=all_task_ids,
    )


if __name__ == "__main__":
    sys.exit(main())
