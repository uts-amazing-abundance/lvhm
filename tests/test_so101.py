import json
from pathlib import Path

import numpy as np
import pytest

from lvhm.config import FAMILIAR, PRODUCTS, EnvConfig

TIMING = Path(__file__).resolve().parents[1] / "assets" / "so101" / "cell_timing.json"


def test_env_timing_matches_mujoco_calibration():
    timing = json.loads(TIMING.read_text())
    cfg = EnvConfig()
    assert cfg.travel_time == pytest.approx(timing["switch_time"], abs=1e-3)
    assert cfg.empty_time == pytest.approx(timing["empty_time"], abs=1e-3)
    assert cfg.loaded_time_base == pytest.approx(timing["loaded_time_base"], abs=1e-3)
    assert cfg.loaded_time_per_slowdown == pytest.approx(timing["loaded_time_per_slowdown"], abs=1e-3)


def test_heavier_or_more_fragile_items_take_longer():
    cfg = EnvConfig()
    assert cfg.proc_time(0.20, 0.3) > cfg.proc_time(0.05, 0.3)
    assert cfg.proc_time(0.10, 0.8) > cfg.proc_time(0.10, 0.1)
    assert cfg.proc_time(0.10, 0.5, careful=True) > cfg.proc_time(0.10, 0.5)
    assert cfg.drop_probability(0.10, 0.5, careful=True) < cfg.drop_probability(0.10, 0.5)


def test_products_stay_within_servo_limits():
    timing = json.loads(TIMING.read_text())
    torque = np.abs(timing["holding_torque_heaviest_product"]["torque_nm"])
    assert timing["holding_torque_heaviest_product"]["payload_kg"] == max(w for w, _ in PRODUCTS.values())
    assert torque.max() < 0.5 * timing["servo_force_range_nm"]
    assert all(PRODUCTS[p][0] <= EnvConfig().features.weight[1] for p in FAMILIAR)


def test_mujoco_timing_reproduces_calibration():
    pytest.importorskip("mujoco")
    from lvhm.so101 import CellLayout, measure_timing

    timing = json.loads(TIMING.read_text())
    result = measure_timing(CellLayout(**timing["layout"]))
    for key in ("empty_time", "loaded_time_base", "loaded_time_per_slowdown", "switch_time"):
        assert result[key] == pytest.approx(timing[key], abs=2e-3), key
