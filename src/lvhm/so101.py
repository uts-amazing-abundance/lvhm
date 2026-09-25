"""SO-ARM101 cell geometry and motion timing, measured in MuJoCo.

The work cell has two stations, one per input tray. Each station holds its input
tray and its own output spot side by side, so consecutive jobs from the same tray
only need a short swing, while switching trays swings the arm across the cell.

Timing comes from the SO101 MJCF published by The Robot Studio (vendored under
assets/so101, STS3215 position-servo model included): waypoints are solved by
inverse kinematics, then each move is commanded as a joint-space trajectory at a
fixed joint speed and simulated until the servos settle.

Requires the `sim` dependency group (`uv sync --group sim`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "so101" / "scene.xml"
ARM_JOINTS = 4  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex; wrist_roll stays at 0
GRIPPER = 5
DOWN = np.array([0.0, 0.0, -1.0])


@dataclass(frozen=True)
class CellLayout:
    radius: float = 0.18  # m from the base axis, inside the top-down workspace
    grasp_z: float = 0.03  # gripper-frame height when grasping or placing
    hover_z: float = 0.08
    station_azimuth: float = 60.0  # deg; station 0 at +azimuth (left), station 1 at -azimuth
    spot_offset: float = 20.0  # deg between a station's centre and its tray / output spot
    joint_speed: float = 1.5  # rad/s, operating speed (STS3215 no-load speed is roughly 3x this)
    gripper_open: float = 1.0  # rad
    gripper_closed: float = 0.1
    settle_tol: float = 0.02  # rad
    settle_vel: float = 0.2  # rad/s

    def spot(self, station: int, kind: str, z: float) -> np.ndarray:
        """World position of a station's tray or output spot at height z."""
        sign = 1.0 if station == 0 else -1.0
        offset = self.spot_offset if kind == "tray" else -self.spot_offset
        az = np.radians(sign * (self.station_azimuth + offset))
        return np.array([self.radius * np.cos(az), self.radius * np.sin(az), z])


class SO101:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.low = self.model.jnt_range[:, 0].copy()
        self.high = self.model.jnt_range[:, 1].copy()

    # ------------------------------------------------------------ kinematics

    def _pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)
        approach = self.data.site_xmat[self.site].reshape(3, 3)[:, 0]  # finger direction
        return self.data.site_xpos[self.site].copy(), approach

    def _ik_from(self, target: np.ndarray, q: np.ndarray, iters: int = 300) -> np.ndarray:
        """Damped least squares on gripper position plus a top-down approach axis."""
        m, d = self.model, self.data
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            pos, approach = self._pose(q)
            err = np.concatenate([target - pos, 0.3 * np.cross(approach, DOWN)])
            if np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 3e-4:
                break
            mujoco.mj_jacSite(m, d, jp, jr, self.site)
            jac = np.vstack([jp[:, :ARM_JOINTS], 0.3 * jr[:, :ARM_JOINTS]])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
            q[:ARM_JOINTS] = np.clip(q[:ARM_JOINTS] + dq, self.low[:ARM_JOINTS], self.high[:ARM_JOINTS])
        return q

    def solve_ik(self, target: np.ndarray, gripper: float) -> np.ndarray:
        """Best of several seeds; raises if the top-down pose is out of reach."""
        best, best_err = None, np.inf
        pan = -np.arctan2(target[1], target[0])  # positive pan swings towards -y
        for lift in (-1.2, -0.6, 0.0, 0.6):
            for elbow in (-0.5, 0.5, 1.2):
                for wrist in (0.3, 0.9, 1.4):
                    q = self._ik_from(target, np.array([pan, lift, elbow, wrist, 0.0, gripper]))
                    pos, approach = self._pose(q)
                    err = np.linalg.norm(target - pos) * 1000 + np.degrees(np.arccos(np.clip(approach @ DOWN, -1, 1)))
                    if err < best_err:
                        best, best_err = q.copy(), err
        pos, approach = self._pose(best)
        if np.linalg.norm(target - pos) > 1e-3 or approach @ DOWN < np.cos(np.radians(2.0)):
            raise ValueError(f"top-down pose at {target} is out of reach")
        return best

    def waypoints(self, layout: CellLayout) -> dict[str, np.ndarray]:
        """Joint configurations for every station: hover and grasp/place poses."""
        wps = {}
        for s in (0, 1):
            for kind in ("tray", "out"):
                wps[f"{kind}{s}_hover"] = self.solve_ik(layout.spot(s, kind, layout.hover_z), layout.gripper_open)
                wps[f"{kind}{s}_low"] = self.solve_ik(layout.spot(s, kind, layout.grasp_z), layout.gripper_open)
        return wps

    # ------------------------------------------------------------ dynamics

    def move_time(self, q_from: np.ndarray, q_to: np.ndarray, speed: float, layout: CellLayout) -> float:
        """Seconds for the servos to follow a linear joint-space trajectory at `speed`
        (rad/s on the joint that moves furthest) and settle at q_to."""
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:] = q_from
        d.ctrl[:] = q_from
        for _ in range(500):  # let the arm settle under gravity at the start pose
            mujoco.mj_step(m, d)
        duration = np.abs(q_to - q_from).max() / speed
        t0 = d.time
        while True:
            s = min((d.time - t0) / duration, 1.0) if duration > 0 else 1.0
            d.ctrl[:] = q_from + s * (q_to - q_from)
            mujoco.mj_step(m, d)
            settled = np.abs(d.qpos - q_to).max() < layout.settle_tol and np.abs(d.qvel).max() < layout.settle_vel
            if s >= 1.0 and settled:
                return d.time - t0
            if d.time - t0 > duration + 5.0:
                raise RuntimeError("servos failed to settle")

    def holding_torque(self, q: np.ndarray, payload: float) -> np.ndarray:
        """Static joint torques (N m) needed to hold `payload` kg at pose q."""
        m, d = self.model, self.data
        body = m.site_bodyid[self.site]
        mass = m.body_mass[body]
        m.body_mass[body] = mass + payload
        try:
            d.qpos[:] = q
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            return d.qfrc_bias[:ARM_JOINTS].copy()
        finally:
            m.body_mass[body] = mass


def with_gripper(q: np.ndarray, value: float) -> np.ndarray:
    q = q.copy()
    q[GRIPPER] = value
    return q


def measure_timing(layout: CellLayout = CellLayout(), slowdowns: tuple[float, ...] = (1.0, 2.0, 3.0)) -> dict:
    """Cycle timing for the scheduling environment.

    empty:  swing from the last output spot to the tray, descend, close the gripper
    loaded: lift, swing to the output spot, descend, open the gripper, rise;
            measured at each loaded slowdown and fitted as a + b * slowdown
    switch: extra swing time when the next tray belongs to the other station
    """
    arm = SO101()
    wp = arm.waypoints(layout)
    v = layout.joint_speed
    closed, opened = layout.gripper_closed, layout.gripper_open

    empty, switch, loaded = [], [], {s: [] for s in slowdowns}
    for s in (0, 1):
        o = 1 - s
        approach = arm.move_time(wp[f"out{s}_hover"], wp[f"tray{s}_hover"], v, layout)
        cross = arm.move_time(wp[f"out{o}_hover"], wp[f"tray{s}_hover"], v, layout)
        empty.append(
            approach
            + arm.move_time(wp[f"tray{s}_hover"], wp[f"tray{s}_low"], v, layout)
            + arm.move_time(wp[f"tray{s}_low"], with_gripper(wp[f"tray{s}_low"], closed), v, layout)
        )
        switch.append(cross - approach)
        for slow in slowdowns:
            vs = v / slow
            path = [
                with_gripper(wp[f"tray{s}_low"], closed),
                with_gripper(wp[f"tray{s}_hover"], closed),
                with_gripper(wp[f"out{s}_hover"], closed),
                with_gripper(wp[f"out{s}_low"], closed),
            ]
            t = sum(arm.move_time(a, b, vs, layout) for a, b in zip(path, path[1:]))
            t += arm.move_time(path[-1], wp[f"out{s}_low"], v, layout)  # release
            t += arm.move_time(wp[f"out{s}_low"], wp[f"out{s}_hover"], v, layout)  # rise
            loaded[slow].append(t)

    xs = np.array(slowdowns)
    ys = np.array([np.mean(loaded[s]) for s in slowdowns])
    b, a = np.polyfit(xs, ys, 1)
    return {
        "layout": asdict(layout),
        "empty_time": float(np.mean(empty)),
        "loaded_time_base": float(a),
        "loaded_time_per_slowdown": float(b),
        "loaded_time_measured": {str(s): float(y) for s, y in zip(slowdowns, ys)},
        "switch_time": float(np.mean(switch)),
        "waypoints": {k: q.round(4).tolist() for k, q in wp.items()},
        "_arm": arm,
        "_wp": wp,
    }
