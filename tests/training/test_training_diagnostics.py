from __future__ import annotations

import json

from dist_net.training_diagnostics import TrainingDiagnostics


def test_training_diagnostics_writes_auditable_convergence_history(tmp_path) -> None:
    diagnostics = TrainingDiagnostics(
        experiment_role="seed42_convergence_audit",
        method="dcrnn",
        region="Orange",
        seed=42,
        max_epochs=100,
        patience=30,
        checkpoint_metric="global_validation_masked_mae_all_valid_points",
        optimizer={"type": "Adam", "learning_rate": 0.01},
        scheduler={
            "type": "MultiStepLR",
            "milestones": [10, 50, 90],
            "gamma": 0.1,
        },
    )
    diagnostics.record_epoch(
        epoch=1,
        train_objective=12.5,
        validation_mae=13.0,
        learning_rate_used=0.01,
        learning_rate_next=0.01,
        stale_epochs=0,
        best_epoch=1,
    )
    diagnostics.finalize(
        tmp_path / "training_diagnostics.json",
        best_epoch=1,
        epochs_run=1,
        stop_reason="max_epochs_reached",
    )

    payload = json.loads((tmp_path / "training_diagnostics.json").read_text())
    assert payload["schema_version"] == 1
    assert payload["experiment_role"] == "seed42_convergence_audit"
    assert payload["best_epoch"] == payload["epochs_run"] == 1
    assert payload["stop_reason"] == "max_epochs_reached"
    assert payload["history"] == [
        {
            "epoch": 1,
            "train_objective": 12.5,
            "validation_mae": 13.0,
            "learning_rate_used": 0.01,
            "learning_rate_next": 0.01,
            "stale_epochs": 0,
            "best_epoch": 1,
        }
    ]
