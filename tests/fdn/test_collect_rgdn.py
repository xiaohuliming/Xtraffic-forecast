import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.collect_rgdn_results import compute_deltas


def test_compute_deltas_verdicts():
    s = {
        "v0a": {"all": 12.40},
        "v0b": {"all": 12.00},
        "v1":  {"all": 11.84},
        "v2":  {"all": 12.05},
    }
    rows = {r["label"]: r for r in compute_deltas(s, band=0.08)}
    assert abs(rows["deseason (v0b-v0a)"]["delta"] - (-0.40)) < 1e-9
    assert rows["deseason (v0b-v0a)"]["verdict"] == "below band (improves)"
    assert rows["headline RGDN (v1-v0b)"]["verdict"] == "below band (improves)"   # -0.16
    assert rows["injection (v1-v2)"]["verdict"] == "below band (improves)"        # -0.21


def test_within_band_and_missing():
    s = {"v0b": {"all": 12.00}, "v1": {"all": 11.97}, "v2": {"all": 11.96}}
    rows = {r["label"]: r for r in compute_deltas(s, band=0.08)}
    assert rows["headline RGDN (v1-v0b)"]["verdict"] == "within noise band"       # -0.03
    assert rows["injection (v1-v2)"]["verdict"] == "within noise band"            # +0.01
    assert rows["deseason (v0b-v0a)"]["verdict"] == "missing"                     # v0a absent
