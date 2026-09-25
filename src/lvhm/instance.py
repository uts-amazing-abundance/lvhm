"""Shift instances: every random quantity of one shift, drawn up front.

Pre-drawing arrivals, processing noise and drop draws per job and attempt gives
common random numbers: every policy evaluated on the same instance faces the same
jobs and the same drop outcome for the same pick, whatever order it serves them in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from lvhm.config import FAMILIAR, PRODUCT_NAMES, PRODUCTS, EnvConfig


@dataclass(frozen=True)
class ShiftInstance:
    arrival: np.ndarray  # (n,), ascending
    proc: np.ndarray  # (n,) nominal processing time
    weight: np.ndarray  # (n,)
    fragility: np.ndarray  # (n,)
    tray: np.ndarray  # (n,) int
    deadline: np.ndarray  # (n,)
    noise: np.ndarray  # (n, max_attempts) multiplicative processing-time noise
    u_drop: np.ndarray  # (n, max_attempts) uniform draw compared with the drop probability
    product: np.ndarray  # (n,) index into PRODUCT_NAMES, -1 for continuous training samples

    def __len__(self) -> int:
        return len(self.arrival)


def normalise_features(config: EnvConfig, feats: np.ndarray) -> np.ndarray:
    lows, highs = config.features.lows, config.features.highs
    return (np.asarray(feats, dtype=float) - lows) / (highs - lows)


def in_holdout(config: EnvConfig, feats: np.ndarray) -> np.ndarray:
    """True where a feature vector falls inside the excluded box around P5."""
    p5 = normalise_features(config, np.array(PRODUCTS["P5"]))
    dist = np.abs(normalise_features(config, feats) - p5).max(axis=-1)
    return dist < config.holdout_radius


def validate_holdout(config: EnvConfig) -> None:
    lows, highs = config.features.lows, config.features.highs
    p5, p6 = np.array(PRODUCTS["P5"]), np.array(PRODUCTS["P6"])
    assert np.all((p5 >= lows) & (p5 <= highs)), "P5 must lie inside the training range"
    assert np.any((p6 < lows) | (p6 > highs)), "P6 must lie outside the training range"
    for name in FAMILIAR:
        feats = np.array(PRODUCTS[name])
        assert np.all((feats >= lows) & (feats <= highs)), f"{name} must lie inside the training range"
        assert not in_holdout(config, feats), f"{name} overlaps the P5 hold-out region"


def sample_training_features(config: EnvConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform samples from the training ranges, rejecting the P5 hold-out region."""
    lows, highs = config.features.lows, config.features.highs
    feats = rng.uniform(lows, highs, size=(n, 3))
    reject = in_holdout(config, feats)
    while reject.any():
        feats[reject] = rng.uniform(lows, highs, size=(int(reject.sum()), 3))
        reject = in_holdout(config, feats)
    return feats


def sample_instance(
    config: EnvConfig,
    rng: np.random.Generator,
    products: Sequence[str] | None = None,
) -> ShiftInstance:
    """Draw one shift. `products=None` samples continuous training features;
    otherwise each job is one of the named products, chosen uniformly."""
    n_arrivals = rng.poisson(config.arrival_rate * config.shift_length)
    arrival = np.concatenate(
        [np.zeros(config.initial_jobs), np.sort(rng.uniform(0.0, config.shift_length, n_arrivals))]
    )
    n = len(arrival)

    if products is None:
        feats = sample_training_features(config, rng, n)
        product = np.full(n, -1)
    else:
        product = np.array([PRODUCT_NAMES.index(p) for p in products])[rng.integers(len(products), size=n)]
        feats = np.array([PRODUCTS[PRODUCT_NAMES[i]] for i in product]).reshape(n, 3)

    proc = feats[:, 0]
    attempts = (n, config.max_attempts)
    return ShiftInstance(
        arrival=arrival,
        proc=proc,
        weight=feats[:, 1],
        fragility=feats[:, 2],
        tray=rng.integers(config.num_trays, size=n),
        deadline=arrival + proc + rng.uniform(*config.slack, size=n),
        noise=np.exp(rng.normal(0.0, config.proc_noise_sigma, size=attempts)),
        u_drop=rng.uniform(size=attempts),
        product=product,
    )


def make_instances(
    config: EnvConfig,
    products: Sequence[str] | None,
    n: int,
    seed: int,
) -> list[ShiftInstance]:
    """A fixed, reproducible set of shifts (e.g. the 100 evaluation shifts)."""
    return [sample_instance(config, np.random.default_rng([seed, i]), products) for i in range(n)]
