"""Environment configuration and product definitions for the LVHM work cell.

Time is measured in simulated minutes throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Product feature vectors: (nominal processing time [min], weight [kg], fragility [0, 1]).
# The agent never sees product identity, only these features.
PRODUCTS: dict[str, tuple[float, float, float]] = {
    "P1": (2.0, 0.10, 0.10),  # quick, light, robust
    "P2": (3.5, 0.25, 0.60),  # medium, fragile
    "P3": (5.5, 0.40, 0.20),  # long, heavy, robust
    "P4": (2.5, 0.15, 0.85),  # quick, very fragile
    "P5": (4.5, 0.30, 0.45),  # held out, inside the training range (interpolation)
    "P6": (6.6, 0.50, 0.95),  # held out, outside the training range (extrapolation)
}
PRODUCT_NAMES: tuple[str, ...] = tuple(PRODUCTS)
FAMILIAR: tuple[str, ...] = ("P1", "P2", "P3", "P4")

# Fixed penalty weights (alpha, beta, delta) used for evaluation shifts.
EVAL_WEIGHTS: tuple[float, float, float] = (0.015, 0.05, 0.6)


@dataclass(frozen=True)
class FeatureRanges:
    """Bounds of the continuous feature distribution used during training."""

    proc: tuple[float, float] = (1.5, 6.0)
    weight: tuple[float, float] = (0.05, 0.45)
    fragility: tuple[float, float] = (0.0, 0.9)

    @property
    def lows(self) -> np.ndarray:
        return np.array([self.proc[0], self.weight[0], self.fragility[0]])

    @property
    def highs(self) -> np.ndarray:
        return np.array([self.proc[1], self.weight[1], self.fragility[1]])


@dataclass(frozen=True)
class WeightRanges:
    """Per-episode sampling ranges of the penalty weights during training."""

    alpha: tuple[float, float] = (0.005, 0.03)  # tardiness, per late minute
    beta: tuple[float, float] = (0.02, 0.10)  # travel, per travel minute
    delta: tuple[float, float] = (0.2, 1.5)  # damage, per drop

    @property
    def lows(self) -> np.ndarray:
        return np.array([self.alpha[0], self.beta[0], self.delta[0]])

    @property
    def highs(self) -> np.ndarray:
        return np.array([self.alpha[1], self.beta[1], self.delta[1]])


@dataclass(frozen=True)
class EnvConfig:
    # Shift and arrivals
    shift_length: float = 480.0
    arrival_rate: float = 0.12  # jobs per minute (Poisson); double it for the busy-day test
    initial_jobs: int = 3  # backlog waiting at the start of the shift
    slack: tuple[float, float] = (5.0, 45.0)  # deadline = arrival + nominal processing + slack

    # Cell layout: the arm stays at the tray it last picked from
    num_trays: int = 2
    travel_time: float = 5.0

    # Stochastic processing and drops
    proc_noise_sigma: float = 0.15  # log-normal multiplicative noise per attempt
    drop_base: float = 0.01
    drop_fragility: float = 0.20
    drop_weight: float = 0.05  # multiplied by weight / weight_scale
    drop_max: float = 0.5
    drop_time_fraction: float = 0.5  # share of the processing time lost on a dropped pick
    max_attempts: int = 2  # a job dropped this many times is scrapped

    # Optional careful pick mode (doubles the action space): lower drop risk, slower pick
    careful_mode: bool = False
    careful_drop_factor: float = 0.3
    careful_time_factor: float = 1.5

    # Reward terms that are not sampled per episode
    on_time_bonus: float = 1.0
    unfinished_penalty: float = 1.0

    # Observation
    num_slots: int = 8
    proc_scale: float = 8.0
    weight_scale: float = 0.6
    deadline_scale: float = 60.0
    wait_scale: float = 120.0
    max_queue_obs: int = 32

    features: FeatureRanges = field(default_factory=FeatureRanges)
    weights: WeightRanges = field(default_factory=WeightRanges)
    # L-inf radius around P5, in normalised feature space, that training never samples
    holdout_radius: float = 0.1
