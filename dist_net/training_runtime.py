"""Shared fail-closed lifecycle for formal and local training entrypoints."""

from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from dist_net.protocol import CACHE_SCHEMA_VERSION, PROTOCOL_VERSION
from dist_net.run_integrity import (
    RunIdentity,
    RunIntegrityError,
    prepare_fresh_output_dir,
    verify_cache_bundle_manifest,
    write_summary_json,
)


def _metric_value(summary: Mapping[str, object], name: str) -> object:
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and name in metrics:
        return metrics[name]
    return summary.get(name)


@dataclass(frozen=True)
class TrainingRunContext:
    output_dir: Path
    identity: RunIdentity | None
    local_metadata: Mapping[str, object]
    cache_manifest: Mapping[str, object] | None

    @classmethod
    def prepare(
        cls,
        output_dir: Path | str,
        *,
        data_dir: Path | str,
        region: str,
        model: str,
        variant: str,
        seed: int,
        sampling_protocol: str,
        environment: Mapping[str, str] | None = None,
    ) -> "TrainingRunContext":
        env = os.environ if environment is None else environment
        formal = any(str(key).startswith("XTRAFFIC_") for key in env)
        destination = Path(output_dir).resolve()
        identity: RunIdentity | None = None
        cache_manifest: Mapping[str, object] | None = None
        if formal:
            identity = RunIdentity.from_environment(env)
            identity.require_expected(
                region=region,
                model=model,
                variant=variant,
                seed=int(seed),
                sampling_protocol=sampling_protocol,
            )
            cache_manifest = verify_cache_bundle_manifest(data_dir, identity)
            destination = prepare_fresh_output_dir(
                destination, identity
            )
        else:
            if destination.exists():
                if not destination.is_dir():
                    raise RunIntegrityError(
                        f"output path is not a directory: {destination}"
                    )
                if next(destination.iterdir(), None) is not None:
                    raise RunIntegrityError(
                        "output directory must be fresh and nonempty data "
                        f"was found: {destination}"
                    )
            else:
                destination.mkdir(parents=True, exist_ok=False)

        metadata = {
            "region": region,
            "model": model,
            "variant": variant,
            "seed": int(seed),
            "protocol_version": PROTOCOL_VERSION,
            "sampling_protocol": sampling_protocol,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
        }
        return cls(destination, identity, metadata, cache_manifest)

    def verify_region_cache(self, cache: object) -> None:
        """Bind the loaded region artifacts to the audited bundle manifest."""
        if self.identity is None:
            return
        region = str(getattr(cache, "region_name", ""))
        expected_region = self.identity.region
        if region != expected_region:
            raise RunIntegrityError(
                f"loaded region mismatch: {region!r} != {expected_region!r}"
            )
        checks = {
            "protocol_version": self.identity.protocol_version,
            "cache_schema_version": self.identity.cache_schema_version,
            "bundle_id": self.identity.cache_bundle_id,
        }
        for name, expected in checks.items():
            actual = getattr(cache, name, None)
            if actual != expected:
                raise RunIntegrityError(
                    f"loaded cache {name} mismatch: "
                    f"{actual!r} != {expected!r}"
                )
        regions = (
            self.cache_manifest.get("regions")
            if isinstance(self.cache_manifest, Mapping)
            else None
        )
        report = regions.get(region) if isinstance(regions, Mapping) else None
        if not isinstance(report, Mapping):
            raise RunIntegrityError(
                f"cache bundle manifest has no audited region {region!r}"
            )
        expected_digest = report.get("cache_bundle_digest")
        actual_digest = getattr(cache, "cache_bundle_digest", None)
        if actual_digest != expected_digest:
            raise RunIntegrityError(
                "loaded region cache digest mismatch: "
                f"{actual_digest!r} != {expected_digest!r}"
            )

    def write_summary(
        self,
        summary: Mapping[str, object],
        *,
        valid_count: int,
        required_metrics: Sequence[str],
    ) -> Path:
        destination = self.output_dir / "summary.json"
        if self.identity is not None:
            return write_summary_json(
                destination,
                summary,
                identity=self.identity,
                valid_count=int(valid_count),
                required_metrics=required_metrics,
            )

        if destination.exists():
            raise RunIntegrityError(
                f"summary path already exists and will not be overwritten: "
                f"{destination}"
            )
        if isinstance(valid_count, bool) or int(valid_count) <= 0:
            raise RunIntegrityError(
                f"valid_count must be positive, got {valid_count!r}"
            )
        payload = dict(summary)
        for key, value in self.local_metadata.items():
            payload.setdefault(key, value)
        payload["valid_count"] = int(valid_count)
        for metric in required_metrics:
            value = _metric_value(payload, metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RunIntegrityError(
                    f"required metric {metric!r} must be finite"
                )
        try:
            text = (
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
        except (TypeError, ValueError) as exc:
            raise RunIntegrityError(
                f"summary is not strict finite JSON: {exc}"
            ) from exc

        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination
