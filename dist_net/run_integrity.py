"""Strict run identity and fresh-output helpers for formal experiments."""
from __future__ import annotations

import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


PROTOCOL_VERSION = "protocol_v3_noleak_dedup"
CACHE_SCHEMA_VERSION = 3
TASK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{2,127}$")
HEX_40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")

ENVIRONMENT_FIELDS = (
    "XTRAFFIC_TASK_ID",
    "XTRAFFIC_REGION",
    "XTRAFFIC_MODEL",
    "XTRAFFIC_VARIANT",
    "XTRAFFIC_SEED",
    "XTRAFFIC_PROTOCOL_VERSION",
    "XTRAFFIC_SAMPLING_PROTOCOL",
    "XTRAFFIC_CACHE_SCHEMA_VERSION",
    "XTRAFFIC_CACHE_BUNDLE_ID",
    "XTRAFFIC_CACHE_BUNDLE_DIGEST",
    "XTRAFFIC_CODE_SHA",
    "XTRAFFIC_CONFIG_HASH",
    "XTRAFFIC_RUN_ATTEMPT",
    "XTRAFFIC_STAGING_ROOT",
)


class RunIntegrityError(RuntimeError):
    """Raised when a formal run violates its identity or output contract."""


def _required_text(
    environment: Mapping[str, str],
    name: str,
) -> str:
    value = environment.get(name)
    if value is None or not value.strip():
        raise RunIntegrityError(f"missing required environment field {name}")
    return value.strip()


def _required_int(
    environment: Mapping[str, str],
    name: str,
) -> int:
    text = _required_text(environment, name)
    try:
        return int(text)
    except ValueError as exc:
        raise RunIntegrityError(f"{name} must be an integer, got {text!r}") from exc


@dataclass(frozen=True)
class RunIdentity:
    task_id: str
    region: str
    model: str
    variant: str
    seed: int
    protocol_version: str
    sampling_protocol: str
    cache_schema_version: int
    cache_bundle_id: str
    cache_bundle_digest: str
    code_sha: str
    config_hash: str
    run_attempt: int
    staging_root: Path

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "RunIdentity":
        env = os.environ if environment is None else environment
        missing = [
            name
            for name in ENVIRONMENT_FIELDS
            if name not in env or not str(env[name]).strip()
        ]
        if missing:
            raise RunIntegrityError(
                "missing required environment fields: " + ", ".join(missing)
            )

        task_id = _required_text(env, "XTRAFFIC_TASK_ID")
        if not TASK_ID_RE.fullmatch(task_id):
            raise RunIntegrityError(f"invalid task id: {task_id!r}")
        seed = _required_int(env, "XTRAFFIC_SEED")
        protocol_version = _required_text(
            env,
            "XTRAFFIC_PROTOCOL_VERSION",
        )
        if protocol_version != PROTOCOL_VERSION:
            raise RunIntegrityError(
                f"invalid protocol version: expected {PROTOCOL_VERSION!r}, "
                f"got {protocol_version!r}"
            )
        sampling_protocol = _required_text(
            env,
            "XTRAFFIC_SAMPLING_PROTOCOL",
        )
        if sampling_protocol not in {"event", "full_window"}:
            raise RunIntegrityError(
                f"invalid sampling protocol: {sampling_protocol!r}"
            )
        cache_schema_version = _required_int(
            env,
            "XTRAFFIC_CACHE_SCHEMA_VERSION",
        )
        if cache_schema_version != CACHE_SCHEMA_VERSION:
            raise RunIntegrityError(
                f"invalid cache schema: expected {CACHE_SCHEMA_VERSION}, "
                f"got {cache_schema_version}"
            )
        cache_bundle_digest = _required_text(
            env,
            "XTRAFFIC_CACHE_BUNDLE_DIGEST",
        ).lower()
        if not HEX_64_RE.fullmatch(cache_bundle_digest):
            raise RunIntegrityError(
                "cache bundle digest must be a 64-character lowercase SHA-256"
            )
        code_sha = _required_text(env, "XTRAFFIC_CODE_SHA").lower()
        if not HEX_40_RE.fullmatch(code_sha):
            raise RunIntegrityError(
                "code_sha must be a full 40-character lowercase Git SHA"
            )
        config_hash = _required_text(env, "XTRAFFIC_CONFIG_HASH").lower()
        if not HEX_64_RE.fullmatch(config_hash):
            raise RunIntegrityError(
                "config_hash must be a 64-character lowercase SHA-256"
            )
        run_attempt = _required_int(env, "XTRAFFIC_RUN_ATTEMPT")
        if run_attempt <= 0:
            raise RunIntegrityError(
                f"run attempt must be positive, got {run_attempt}"
            )
        staging_text = _required_text(env, "XTRAFFIC_STAGING_ROOT")
        staging_path = Path(staging_text)
        if not staging_path.is_absolute():
            raise RunIntegrityError(
                f"staging root must be absolute, got {staging_text!r}"
            )

        return cls(
            task_id=task_id,
            region=_required_text(env, "XTRAFFIC_REGION"),
            model=_required_text(env, "XTRAFFIC_MODEL"),
            variant=_required_text(env, "XTRAFFIC_VARIANT"),
            seed=seed,
            protocol_version=protocol_version,
            sampling_protocol=sampling_protocol,
            cache_schema_version=cache_schema_version,
            cache_bundle_id=_required_text(
                env,
                "XTRAFFIC_CACHE_BUNDLE_ID",
            ),
            cache_bundle_digest=cache_bundle_digest,
            code_sha=code_sha,
            config_hash=config_hash,
            run_attempt=run_attempt,
            staging_root=staging_path.resolve(),
        )

    def summary_metadata(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "region": self.region,
            "model": self.model,
            "variant": self.variant,
            "seed": self.seed,
            "protocol_version": self.protocol_version,
            "sampling_protocol": self.sampling_protocol,
            "cache_schema_version": self.cache_schema_version,
            "cache_bundle_id": self.cache_bundle_id,
            "cache_bundle_digest": self.cache_bundle_digest,
            "code_sha": self.code_sha,
            "config_hash": self.config_hash,
        }

    def require_expected(self, **expected: object) -> None:
        allowed = {
            "task_id",
            "region",
            "model",
            "variant",
            "seed",
            "protocol_version",
            "sampling_protocol",
            "cache_schema_version",
            "cache_bundle_id",
            "cache_bundle_digest",
            "code_sha",
            "config_hash",
            "run_attempt",
            "staging_root",
        }
        unknown = set(expected) - allowed
        if unknown:
            raise RunIntegrityError(
                f"unknown expected identity fields: {sorted(unknown)}"
            )
        for name, value in expected.items():
            actual = getattr(self, name)
            if actual != value:
                raise RunIntegrityError(
                    f"run identity {name} mismatch: expected {value!r}, "
                    f"got {actual!r}"
                )


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def prepare_fresh_output_dir(
    output_dir: Path | str,
    identity: RunIdentity,
) -> Path:
    staging_root = identity.staging_root
    candidate = _lexical_absolute(Path(output_dir))
    try:
        relative = candidate.relative_to(staging_root)
    except ValueError as exc:
        raise RunIntegrityError(
            f"output directory is outside staging root: {candidate}"
        ) from exc
    if not relative.parts:
        raise RunIntegrityError("output directory cannot equal the staging root")

    cursor = staging_root
    if cursor.is_symlink():
        raise RunIntegrityError(f"staging root cannot be a symlink: {cursor}")
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RunIntegrityError(
                f"output path contains a symlink: {cursor}"
            )
    try:
        candidate.resolve().relative_to(staging_root)
    except ValueError as exc:
        raise RunIntegrityError(
            f"output directory resolves outside staging root: {candidate}"
        ) from exc

    if candidate.exists():
        if not candidate.is_dir():
            raise RunIntegrityError(
                f"output path exists and is not a directory: {candidate}"
            )
        if next(candidate.iterdir(), None) is not None:
            raise RunIntegrityError(
                f"output directory must be fresh and nonempty data was found: "
                f"{candidate}"
            )
    else:
        candidate.mkdir(parents=True, exist_ok=False)
    return candidate.resolve()


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RunIntegrityError(
                f"cache bundle manifest contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value}")


def verify_cache_bundle_manifest(
    data_dir: Path | str,
    identity: RunIdentity,
) -> dict[str, object]:
    root = Path(data_dir)
    manifest_path = root / "cache_bundle_manifest.json"
    if manifest_path.is_symlink():
        raise RunIntegrityError(
            f"cache bundle manifest cannot be a symlink: {manifest_path}"
        )
    if not manifest_path.is_file():
        raise RunIntegrityError(
            f"cache bundle manifest is missing: {manifest_path}"
        )
    try:
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except RunIntegrityError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunIntegrityError(
            f"cache bundle manifest is malformed: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RunIntegrityError(
            "cache bundle manifest must contain a JSON object"
        )

    expected = {
        "status": "verified",
        "protocol_version": identity.protocol_version,
        "cache_schema_version": identity.cache_schema_version,
        "cache_bundle_id": identity.cache_bundle_id,
        "cache_bundle_digest": identity.cache_bundle_digest,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise RunIntegrityError(
                f"cache bundle manifest {name} mismatch: "
                f"expected {value!r}, got {payload.get(name)!r}"
            )
    return payload


def _metric_value(summary: Mapping[str, object], name: str) -> object:
    metrics = summary.get("metrics")
    if isinstance(metrics, dict) and name in metrics:
        return metrics[name]
    return summary.get(name)


def _strict_json_text(value: object) -> str:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
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


def write_summary_json(
    path: Path | str,
    summary: Mapping[str, object],
    *,
    identity: RunIdentity,
    valid_count: int,
    required_metrics: Sequence[str],
) -> Path:
    destination = _lexical_absolute(Path(path))
    try:
        destination.relative_to(identity.staging_root)
    except ValueError as exc:
        raise RunIntegrityError(
            f"summary path is outside staging root: {destination}"
        ) from exc
    if destination.exists():
        raise RunIntegrityError(
            f"summary path already exists and will not be overwritten: "
            f"{destination}"
        )
    if not destination.parent.is_dir():
        raise RunIntegrityError(
            f"summary parent directory does not exist: {destination.parent}"
        )
    if (
        isinstance(valid_count, bool)
        or not isinstance(valid_count, int)
        or valid_count <= 0
    ):
        raise RunIntegrityError(
            f"valid_count must be a positive integer, got {valid_count!r}"
        )
    if not isinstance(summary, Mapping):
        raise RunIntegrityError("summary must be a mapping")

    payload = dict(summary)
    for name, expected in identity.summary_metadata().items():
        if name in payload and payload[name] != expected:
            raise RunIntegrityError(
                f"summary {name} mismatch: expected {expected!r}, "
                f"got {payload[name]!r}"
            )
        payload[name] = expected
    if "valid_count" in payload and payload["valid_count"] != valid_count:
        raise RunIntegrityError(
            f"summary valid_count mismatch: expected {valid_count}, "
            f"got {payload['valid_count']!r}"
        )
    payload["valid_count"] = valid_count

    text = _strict_json_text(payload)
    for metric in required_metrics:
        value = _metric_value(payload, metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise RunIntegrityError(
                f"required metric {metric!r} must be finite, got {value!r}"
            )
        if float(value) < 0:
            raise RunIntegrityError(
                f"required metric {metric!r} must be nonnegative, got {value!r}"
            )

    lock = destination.with_name(f".{destination.name}.lock")
    temporary = destination.with_name(
        f".s.{uuid.uuid4().hex[:16]}.tmp"
    )
    try:
        with lock.open("x", encoding="utf-8"):
            pass
    except FileExistsError as exc:
        raise RunIntegrityError(
            f"summary write is already in progress: {destination}"
        ) from exc
    try:
        if destination.exists():
            raise RunIntegrityError(
                f"summary path already exists and will not be overwritten: "
                f"{destination}"
            )
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
        if lock.exists():
            lock.unlink()
    return destination
