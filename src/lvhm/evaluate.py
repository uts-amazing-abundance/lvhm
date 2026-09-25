"""Evaluate a policy on a fixed set of shifts with fixed penalty weights."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lvhm.baselines import Policy
from lvhm.config import EVAL_WEIGHTS, EnvConfig
from lvhm.env import LVHMSchedulingEnv
from lvhm.instance import ShiftInstance

METRICS = ("on_time", "tardiness", "travel", "drops", "scrapped", "unfinished", "score")


def run_episode(
    env: LVHMSchedulingEnv,
    policy: Policy,
    instance: ShiftInstance,
    weights: Sequence[float] = EVAL_WEIGHTS,
) -> dict:
    obs, _ = env.reset(options={"instance": instance, "weights": weights})
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs, env))
        if terminated or truncated:
            return info["episode_metrics"] | {"jobs": len(instance)}


def evaluate(
    policy: Policy,
    instances: Sequence[ShiftInstance],
    config: EnvConfig | None = None,
    weights: Sequence[float] = EVAL_WEIGHTS,
) -> list[dict]:
    env = LVHMSchedulingEnv(config)
    return [run_episode(env, policy, inst, weights) for inst in instances]


def summarise(results: Sequence[dict]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of each metric across shifts."""
    return {m: (float(np.mean([r[m] for r in results])), float(np.std([r[m] for r in results]))) for m in METRICS}
