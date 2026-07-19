import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dist_net.run_integrity import RunIntegrityError
from dist_net.training_runtime import TrainingRunContext
from tests.remote.test_run_integrity import valid_environment


def test_local_run_refuses_a_nonempty_output_directory(tmp_path):
    output = tmp_path / "result"
    output.mkdir()
    (output / "old.ckpt").write_text("old", encoding="utf-8")

    with pytest.raises(RunIntegrityError, match="nonempty"):
        TrainingRunContext.prepare(
            output,
            data_dir=tmp_path / "cache",
            region="Alameda",
            model="staeformer",
            variant="base",
            seed=42,
            sampling_protocol="full_window",
            environment={},
        )


def test_formal_run_verifies_cache_manifest_and_writes_identity(tmp_path):
    stage = tmp_path / "stage"
    environment = valid_environment(stage)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "cache_bundle_manifest.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "protocol_version": environment[
                    "XTRAFFIC_PROTOCOL_VERSION"
                ],
                "cache_schema_version": 3,
                "cache_bundle_id": environment[
                    "XTRAFFIC_CACHE_BUNDLE_ID"
                ],
                "cache_bundle_digest": environment[
                    "XTRAFFIC_CACHE_BUNDLE_DIGEST"
                ],
                "regions": {
                    "Alameda": {
                        "cache_bundle_digest": "d" * 64,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = stage.resolve() / "work" / "result"

    context = TrainingRunContext.prepare(
        output,
        data_dir=cache,
        region="Alameda",
        model="staeformer_deseason",
        variant="s0a",
        seed=42,
        sampling_protocol="event",
        environment=environment,
    )
    summary_path = context.write_summary(
        {"all": 1.0, "affected": 2.0, "unaffected": 0.5},
        valid_count=10,
        required_metrics=("all", "affected", "unaffected"),
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["task_id"] == environment["XTRAFFIC_TASK_ID"]
    assert summary["valid_count"] == 10
    assert summary["code_sha"] == environment["XTRAFFIC_CODE_SHA"]
    context.verify_region_cache(
        SimpleNamespace(
            region_name="Alameda",
            protocol_version=environment["XTRAFFIC_PROTOCOL_VERSION"],
            cache_schema_version=3,
            bundle_id=environment["XTRAFFIC_CACHE_BUNDLE_ID"],
            cache_bundle_digest="d" * 64,
        )
    )

    with pytest.raises(RunIntegrityError, match="region cache digest"):
        context.verify_region_cache(
            SimpleNamespace(
                region_name="Alameda",
                protocol_version=environment["XTRAFFIC_PROTOCOL_VERSION"],
                cache_schema_version=3,
                bundle_id=environment["XTRAFFIC_CACHE_BUNDLE_ID"],
                cache_bundle_digest="e" * 64,
            )
        )


def test_local_summary_is_strict_json_and_never_overwrites(tmp_path):
    context = TrainingRunContext.prepare(
        tmp_path / "result",
        data_dir=tmp_path / "cache",
        region="Alameda",
        model="staeformer",
        variant="base",
        seed=42,
        sampling_protocol="full_window",
        environment={},
    )

    with pytest.raises(RunIntegrityError, match="finite JSON"):
        context.write_summary(
            {"all": 1.0, "optional": float("nan")},
            valid_count=3,
            required_metrics=("all",),
        )
