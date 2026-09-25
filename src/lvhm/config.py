"""Environment configuration and product definitions for the SO-ARM101 work cell.

Time is measured in seconds. Motion timing (empty, loaded and station-switch
times) comes from the MuJoCo calibration in scripts/calibrate_so101.py.
Processing time is not a free product feature: heavy or fragile items are moved
more slowly, so a job's processing time follows from its weight and fragility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Product feature vectors: (weight [kg], fragility [0, 1]).
# The agent never sees product identity, only these features.
PRODUCTS: dict[str, tuple[float, float]] = {
    "P1": (0.03, 0.10),  # light, robust
    "P2": (0.08, 0.60),  # light, fragile
    "P3": (0.18, 0.20),  # heavy, robust
    "P4": (0.05, 0.85),  # light, very fragile
    "P5": (0.12, 0.45),  # held out, inside the training range (interpolation)
    "P6": (0.23, 0.95),  # held out, outside the training range (extrapolation)
}
PRODUCT_NAMES: tuple[str, ...] = tuple(PRODUCTS)
FAMILIAR: tuple[str, ...] = ("P1", "P2", "P3", "P4")

# Fixed penalty weights (alpha, beta, delta) used for evaluation shifts.
EVAL_WEIGHTS: tuple[float, float, float] = (0.015, 0.15, 0.6)


@dataclass(frozen=True)
class FeatureRanges:
    """Bounds of the continuous feature distribution used during training."""

    weight: tuple[float, float] = (0.02, 0.20)
    fragility: tuple[float, float] = (0.0, 0.9)

    @property
    def lows(self) -> np.ndarray:
        return np.array([self.weight[0], self.fragility[0]])

    @property
    def highs(self) -> np.ndarray:
        return np.array([self.weight[1], self.fragility[1]])


@dataclass(frozen=True)
class WeightRanges:
    """Per-episode sampling ranges of the penalty weights during training."""

    alpha: tuple[float, float] = (0.005, 0.03)  # tardiness, per late second
    beta: tuple[float, float] = (0.05, 0.30)  # travel, per switch second
    delta: tuple[float, float] = (0.2, 1.5)  # damage, per drop

    @property
    def lows(self) -> np.ndarray:
        return np.array([self.alpha[0], self.beta[0], self.delta[0]])

    @property
    def highs(self) -> np.ndarray:
        return np.array([self.alpha[1], self.beta[1], self.delta[1]])


@dataclass(frozen=True)
class EnvConfig:
    # Shift and arrivals (an 8-minute run of the cell)
    shift_length: float = 480.0
    arrival_rate: float = 0.14  # jobs per second (Poisson) for fixed evaluation shifts; 0.28 is the busy day
    # Training draws a rate per episode from this range (None: always use arrival_rate)
    arrival_rate_range: tuple[float, float] | None = (0.07, 0.21)
    initial_jobs: int = 3  # backlog waiting at the start of the shift
    slack: tuple[float, float] = (5.0, 50.0)  # deadline = arrival + nominal processing + slack

    # Cell layout: one station per tray; the arm stays at the station it last served.
    # Timing measured in MuJoCo (assets/so101/cell_timing.json).
    num_trays: int = 2
    travel_time: float = 1.15  # extra swing when switching stations
    empty_time: float = 1.388  # swing to the tray, descend, close the gripper
    loaded_time_base: float = 0.929  # loaded motion: base + per_slowdown * slowdown
    loaded_time_per_slowdown: float = 0.987

    # Handling speed: loaded motions run at 1 / slowdown of the normal joint speed
    slow_weight: float = 1.5  # multiplied by weight / weight_scale
    slow_fragility: float = 1.5

    # Stochastic processing and drops
    proc_noise_sigma: float = 0.10  # log-normal multiplicative noise per attempt
    drop_base: float = 0.01
    drop_fragility: float = 0.20
    drop_weight: float = 0.05  # multiplied by weight / weight_scale
    drop_max: float = 0.5
    drop_time_fraction: float = 0.5  # share of the processing time lost on a dropped pick
    max_attempts: int = 2  # a job dropped this many times is scrapped

    # Careful pick mode (doubles the action space): slower loaded motion, lower drop risk
    careful_mode: bool = True
    careful_drop_factor: float = 0.3
    careful_slowdown: float = 1.5  # extra slowdown on the loaded motion

    # Reward terms that are not sampled per episode
    on_time_bonus: float = 1.0
    unfinished_penalty: float = 1.0

    # Observation
    num_slots: int = 8
    proc_scale: float = 8.0
    weight_scale: float = 0.25
    deadline_scale: float = 60.0
    wait_scale: float = 120.0
    max_queue_obs: int = 32

    features: FeatureRanges = field(default_factory=FeatureRanges)
    weights: WeightRanges = field(default_factory=WeightRanges)
    # L-inf radius around P5, in normalised feature space, that training never samples
    holdout_radius: float = 0.1

    def slowdown(self, weight, fragility):
        return 1.0 + self.slow_weight * np.asarray(weight) / self.weight_scale + self.slow_fragility * np.asarray(fragility)

    def proc_time(self, weight, fragility, careful: bool = False):
        """Nominal processing time of one pick-and-place, before noise."""
        slow = self.slowdown(weight, fragility) * (self.careful_slowdown if careful else 1.0)
        return self.empty_time + self.loaded_time_base + self.loaded_time_per_slowdown * slow

    def drop_probability(self, weight, fragility, careful: bool = False):
        q = self.drop_base + self.drop_fragility * np.asarray(fragility) + self.drop_weight * np.asarray(weight) / self.weight_scale
        q = np.minimum(q, self.drop_max)
        return q * self.careful_drop_factor if careful else q
