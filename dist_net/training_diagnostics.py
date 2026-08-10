"""Small, auditable training-history artifact used by convergence runs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class TrainingDiagnostics:
    experiment_role: str
    method: str
    region: str
    seed: int
    max_epochs: int
    patience: int | None
    checkpoint_metric: str
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any] | None
    history: list[dict[str, Any]] = field(default_factory=list)

    def record_epoch(
        self,
        *,
        epoch: int,
        train_objective: float,
        validation_mae: float,
        learning_rate_used: float,
        learning_rate_next: float,
        stale_epochs: int,
        best_epoch: int,
    ) -> None:
        self.history.append(
            {
                "epoch": int(epoch),
                "train_objective": float(train_objective),
                "validation_mae": float(validation_mae),
                "learning_rate_used": float(learning_rate_used),
                "learning_rate_next": float(learning_rate_next),
                "stale_epochs": int(stale_epochs),
                "best_epoch": int(best_epoch),
            }
        )

    def finalize(
        self,
        path: Path,
        *,
        best_epoch: int,
        epochs_run: int,
        stop_reason: str,
    ) -> None:
        payload = {
            "schema_version": 1,
            "experiment_role": self.experiment_role,
            "method": self.method,
            "region": self.region,
            "seed": int(self.seed),
            "max_epochs": int(self.max_epochs),
            "patience": None if self.patience is None else int(self.patience),
            "best_epoch": int(best_epoch),
            "epochs_run": int(epochs_run),
            "stop_reason": str(stop_reason),
            "checkpoint_metric": self.checkpoint_metric,
            "optimizer": dict(self.optimizer),
            "scheduler": None if self.scheduler is None else dict(self.scheduler),
            "history": self.history,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
