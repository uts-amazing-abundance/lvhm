"""Measure SO-ARM101 cycle timing in MuJoCo and write assets/so101/cell_timing.json.

EnvConfig's timing defaults must match this file (a test checks it), so rerun this
script and update EnvConfig after changing the cell layout or joint speed.

    uv run --group sim python scripts/calibrate_so101.py
"""

from __future__ import annotations

import json
from pathlib import Path

from lvhm.config import PRODUCTS
from lvhm.so101 import CellLayout, measure_timing

OUT = Path(__file__).resolve().parents[1] / "assets" / "so101" / "cell_timing.json"


def main() -> None:
    result = measure_timing(CellLayout())
    arm, wp = result.pop("_arm"), result.pop("_wp")

    heaviest = max(w for w, _ in PRODUCTS.values())
    torque = arm.holding_torque(wp["tray0_hover"], heaviest)
    result["holding_torque_heaviest_product"] = {"payload_kg": heaviest, "torque_nm": torque.round(3).tolist()}
    result["servo_force_range_nm"] = float(arm.model.actuator_forcerange[0, 1])

    for key in ("empty_time", "loaded_time_base", "loaded_time_per_slowdown", "switch_time"):
        result[key] = round(result[key], 3)
    OUT.write_text(json.dumps(result, indent=2) + "\n")

    print(f"empty  {result['empty_time']:.3f} s   (swing to tray, descend, grip)")
    print(f"loaded {result['loaded_time_base']:.3f} + {result['loaded_time_per_slowdown']:.3f} x slowdown s")
    print(f"switch {result['switch_time']:.3f} s   (extra swing to the other station)")
    print(f"holding torque with {heaviest} kg: {torque.round(3)} N m (servo limit {result['servo_force_range_nm']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
