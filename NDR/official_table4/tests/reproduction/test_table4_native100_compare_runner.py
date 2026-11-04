from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.reproduction.igstgnn_table4.native100_compare_matrix_runner import (
    build_native100_task_command,
)
from scripts.reproduction.igstgnn_table4.native100_compare_runner import (
    load_native100_task_spec,
    run_bist_probe_step,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config" / "reproduction" / "igstgnn_table4"
MANIFEST = CONFIG_ROOT / "matrix.native100_compare_v1.json"
BASE_MANIFEST = CONFIG_ROOT / "matrix.public_material_v2.json"
BASE_MANIFEST_SHA256 = (
    "d0c7fb88e24c6026d670a4a6d571357eb5320393417a83d344adb05151023a78"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(task_id: str):
    return load_native100_task_spec(
        manifest_path=MANIFEST,
        manifest_sha256=_sha256(MANIFEST),
        base_manifest_path=BASE_MANIFEST,
        base_manifest_sha256=BASE_MANIFEST_SHA256,
        task_id=task_id,
    )


def test_native100_task_specs_preserve_native_recipes_except_epoch_cap():
    bist = _load("n100001_bist_alameda_seed42")
    assert bist.task["method"] == "bist"
    assert bist.base_effective["max_epochs"] == 500
    assert bist.effective["max_epochs"] == 100
    assert bist.effective["patience"] == 20
    assert bist.effective["batch_size_by_region"]["Alameda"] == 64
    assert bist.effective["optimizer"]["learning_rate"] == 0.002

    ndr = _load("n100010_ndr_net_alameda_seed42")
    assert ndr.task["method"] == "ndr_net"
    assert ndr.base_effective["max_epochs"] == 30
    assert ndr.effective["max_epochs"] == 100
    assert ndr.effective["batch_size_by_region"]["Alameda"] == 16
    assert ndr.effective["optimizer"]["learning_rate"] == 0.001
    assert ndr.method_profile["effective"]["early_stopping"] is False


def test_bist_probe_step_consumes_exactly_one_real_batch():
    class Loader:
        yielded = 0

        @staticmethod
        def shuffle():
            return None

        @classmethod
        def get_iterator(cls):
            for offset in (1.0, 2.0):
                cls.yielded += 1
                x = np.full((2, 12, 3, 1), offset, dtype=np.float32)
                y = np.full((2, 12, 3, 1), offset + 1, dtype=np.float32)
                yield x, y

    class Model(torch.nn.Module):
        horizon = 12

        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, _x, label):
            return label + self.bias

    model = Model()
    engine = SimpleNamespace(
        model=model,
        _dataloader={"train_loader": Loader()},
        _optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        _loss_fn=lambda pred, label, _null: torch.mean(torch.abs(pred - label)),
        _clip_grad_value=5.0,
        _args=SimpleNamespace(horizon=12),
        _iter_cnt=0,
        _to_tensor=lambda values: [torch.as_tensor(value) for value in values],
        _to_device=lambda values: values,
        _inverse_transform=lambda values: values,
    )

    result = run_bist_probe_step(engine)

    assert Loader.yielded == 1
    assert result["prediction_shape"] == (2, 12, 3, 1)
    assert result["loss"] == 0.5


def test_native100_task_command_targets_the_same_table_runner():
    command = build_native100_task_command(
        python_executable=Path("/venv/bin/python"),
        manifest_path=Path("/repo/config/matrix.native100.json"),
        manifest_sha256="1" * 64,
        base_manifest_path=Path("/repo/config/matrix.base.json"),
        base_manifest_sha256="2" * 64,
        task_id="n100010_ndr_net_alameda_seed42",
        splits_root=Path("/data/official_splits"),
        results_root=Path("/results/native100"),
        device="cuda:0",
    )

    assert command[1:4] == ["-X", "utf8", "-m"]
    assert command[4] == (
        "scripts.reproduction.igstgnn_table4.native100_compare_runner"
    )
    assert command[command.index("--task-id") + 1] == (
        "n100010_ndr_net_alameda_seed42"
    )
    assert Path(command[command.index("--results-root") + 1]) == Path(
        "/results/native100"
    )
