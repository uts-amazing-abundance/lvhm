"""Discrete-event Gymnasium environment for job sequencing in an SO-ARM101 work cell.

A decision happens whenever the arm is idle and the queue is non-empty. The action
picks one of the K oldest waiting jobs (and, in careful mode, the pick mode); the
simulator then runs travel + processing in one step and admits the jobs that
arrived meanwhile. When the queue is empty the clock jumps to the next arrival.
The episode terminates when the shift clock expires.

Reward per decision:
    + on_time_bonus                if the served job finishes by its deadline
    - alpha * late seconds         accrued by every job in the system during the step
    - beta  * travel seconds
    - delta                        per drop (re-queue on the first, scrap on the last)
    - unfinished_penalty * jobs    still waiting when the shift ends
Tardiness is charged as it accrues rather than at completion, so leaving a late job
unserved never saves its tardiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lvhm.config import EnvConfig
from lvhm.instance import ShiftInstance, sample_instance, validate_holdout


@dataclass(frozen=True)
class DecisionState:
    """Unclipped view of the visible jobs, for the rule-based baselines."""

    t: float
    arm: int
    arrival: np.ndarray
    proc: np.ndarray
    weight: np.ndarray
    fragility: np.ndarray
    deadline: np.ndarray
    tray: np.ndarray
    attempts: np.ndarray


class LVHMSchedulingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig | None = None):
        super().__init__()
        self.cfg = cfg = config or EnvConfig()
        if cfg.initial_jobs < 1:
            raise ValueError("initial_jobs must be >= 1 so every episode has a decision")
        validate_holdout(cfg)

        self.num_modes = 2 if cfg.careful_mode else 1
        self.slot_dim = 7 + cfg.num_trays  # present, proc, weight, fragility, ttd, tray one-hot, wait, retry
        self.global_dim = 1 + cfg.num_trays + 1 + 3  # shift left, arm one-hot, queue length, weights
        self.action_space = spaces.Discrete(cfg.num_slots * self.num_modes)

        slot_low = np.zeros(self.slot_dim, dtype=np.float32)
        slot_low[4] = -1.0  # time to deadline goes negative once late
        global_high = np.ones(self.global_dim, dtype=np.float32)
        global_high[1 + cfg.num_trays] = cfg.max_queue_obs / cfg.num_slots
        global_high[-3:] = 2.0  # test-time weights may exceed the training range
        low = np.concatenate([np.tile(slot_low, cfg.num_slots), np.zeros(self.global_dim, dtype=np.float32)])
        high = np.concatenate([np.ones(self.slot_dim * cfg.num_slots, dtype=np.float32), global_high])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self._inst: ShiftInstance | None = None

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Options: `instance` (a fixed ShiftInstance), `products` (product names to
        draw jobs from), `arrival_rate` (fixed rate instead of the training range),
        `weights` (fixed (alpha, beta, delta))."""
        super().reset(seed=seed)
        options = options or {}
        cfg = self.cfg

        inst = options.get("instance")
        if inst is None:
            rate = options.get("arrival_rate")
            if rate is None and cfg.arrival_rate_range is not None:
                rate = self.np_random.uniform(*cfg.arrival_rate_range)
            inst = sample_instance(cfg, self.np_random, options.get("products"), rate)
        if len(inst) == 0:
            raise ValueError("instance has no jobs")
        weights = options.get("weights")
        if weights is None:
            weights = self.np_random.uniform(cfg.weights.lows, cfg.weights.highs)

        self._inst = inst
        self._w = np.asarray(weights, dtype=float)
        self._attempts = np.zeros(len(inst), dtype=int)
        self._queue: list[int] = []  # job indices; indices are in arrival order
        self._next = 0
        self._t = 0.0
        self._arm = 0
        self._stats = dict.fromkeys(
            ("decisions", "completed", "on_time", "late", "drops", "scrapped", "unfinished", "careful"), 0
        ) | dict.fromkeys(("tardiness", "travel", "score"), 0.0)

        self._admit(self._t)
        if not self._queue:
            self._t = float(inst.arrival[self._next])
            self._admit(self._t)
        return self._obs(), {"t": self._t}

    def step(self, action: int):
        assert self._inst is not None, "call reset() first"
        cfg, inst = self.cfg, self._inst
        alpha, beta, delta = self._w

        if not self.action_masks()[action]:
            raise ValueError(f"action {action} selects an empty slot")
        slot, careful = int(action) % cfg.num_slots, int(action) // cfg.num_slots == 1
        j = self._queue[slot]

        t0 = self._t
        travel = cfg.travel_time if inst.tray[j] != self._arm else 0.0
        attempt = self._attempts[j]
        proc = cfg.proc_time(inst.weight[j], inst.fragility[j], careful) * inst.noise[j, attempt]
        dropped = bool(inst.u_drop[j, attempt] < self.drop_probability(j, careful))
        t1 = t0 + travel + (cfg.drop_time_fraction * proc if dropped else proc)

        # Every job in the system during [t0, t1] accrues lateness, including the one
        # being served and jobs that arrive meanwhile (their deadlines lie after t0).
        self._admit(t1)
        deadlines = inst.deadline[self._queue]
        late = float(np.maximum(0.0, t1 - np.maximum(t0, deadlines)).sum())

        reward = -alpha * late - beta * travel
        self._arm = int(inst.tray[j])
        if dropped:
            self._attempts[j] += 1
            self._stats["drops"] += 1
            reward -= delta
            if self._attempts[j] >= cfg.max_attempts:
                self._queue.remove(j)
                self._stats["scrapped"] += 1
        else:
            self._queue.remove(j)
            self._stats["completed"] += 1
            if t1 <= inst.deadline[j]:
                self._stats["on_time"] += 1
                reward += cfg.on_time_bonus
            else:
                self._stats["late"] += 1

        self._t = t1
        terminated = self._advance()
        if terminated:
            self._stats["unfinished"] = len(self._queue)
            reward -= cfg.unfinished_penalty * len(self._queue)

        self._stats["decisions"] += 1
        self._stats["careful"] += careful
        self._stats["tardiness"] += late
        self._stats["travel"] += travel
        self._stats["score"] += reward

        info: dict[str, Any] = {
            "t": self._t,
            "job": j,
            "slot": slot,
            "careful": careful,
            "start": t0,
            "end": t1,
            "travel": travel,
            "dropped": dropped,
        }
        if terminated:
            info["episode_metrics"] = dict(self._stats)
        return self._obs(), float(reward), terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Valid actions; the method name is what sb3-contrib's MaskablePPO looks for."""
        slots = np.zeros(self.cfg.num_slots, dtype=bool)
        slots[: min(len(self._queue), self.cfg.num_slots)] = True
        return np.tile(slots, self.num_modes)

    # ------------------------------------------------------------------ helpers

    def drop_probability(self, j: int, careful: bool = False) -> float:
        return float(self.cfg.drop_probability(self._inst.weight[j], self._inst.fragility[j], careful))

    def decision_state(self) -> DecisionState:
        inst = self._inst
        vis = np.array(self._queue[: self.cfg.num_slots], dtype=int)
        return DecisionState(
            t=self._t,
            arm=self._arm,
            arrival=inst.arrival[vis],
            proc=inst.proc[vis],
            weight=inst.weight[vis],
            fragility=inst.fragility[vis],
            deadline=inst.deadline[vis],
            tray=inst.tray[vis],
            attempts=self._attempts[vis],
        )

    def _admit(self, t: float) -> None:
        arrival, n = self._inst.arrival, len(self._inst)
        while self._next < n and arrival[self._next] <= t:
            self._queue.append(self._next)
            self._next += 1

    def _advance(self) -> bool:
        """Move the clock to the next decision point; return True when the shift is over."""
        if self._t >= self.cfg.shift_length:
            return True
        if self._queue:
            return False
        if self._next >= len(self._inst) or self._inst.arrival[self._next] >= self.cfg.shift_length:
            return True
        self._t = float(self._inst.arrival[self._next])
        self._admit(self._t)
        return False

    def _obs(self) -> np.ndarray:
        cfg, inst, t = self.cfg, self._inst, self._t
        vis = np.array(self._queue[: cfg.num_slots], dtype=int)
        n = len(vis)

        slots = np.zeros((cfg.num_slots, self.slot_dim), dtype=np.float32)
        slots[:n, 0] = 1.0
        slots[:n, 1] = np.minimum(inst.proc[vis] / cfg.proc_scale, 1.0)
        slots[:n, 2] = np.minimum(inst.weight[vis] / cfg.weight_scale, 1.0)
        slots[:n, 3] = inst.fragility[vis]
        slots[:n, 4] = np.clip((inst.deadline[vis] - t) / cfg.deadline_scale, -1.0, 1.0)
        slots[np.arange(n), 5 + inst.tray[vis]] = 1.0
        slots[:n, 5 + cfg.num_trays] = np.minimum((t - inst.arrival[vis]) / cfg.wait_scale, 1.0)
        slots[:n, 6 + cfg.num_trays] = self._attempts[vis] > 0

        w_lo, w_hi = cfg.weights.lows, cfg.weights.highs
        glob = np.concatenate(
            [
                [max(0.0, (cfg.shift_length - t) / cfg.shift_length)],
                np.eye(cfg.num_trays)[self._arm],
                [min(len(self._queue), cfg.max_queue_obs) / cfg.num_slots],
                np.clip((self._w - w_lo) / (w_hi - w_lo), 0.0, 2.0),
            ]
        ).astype(np.float32)
        return np.concatenate([slots.ravel(), glob])
