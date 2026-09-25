"""Part B test conditions. Every policy faces the same fixed shifts in each condition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from lvhm.agents.common import TEST_SEED
from lvhm.baselines import Policy
from lvhm.config import EVAL_WEIGHTS, FAMILIAR, EnvConfig
from lvhm.evaluate import evaluate
from lvhm.instance import make_instances


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    products: tuple[str, ...]
    arrival_rate: float
    weights: tuple[float, float, float]


def conditions(config: EnvConfig | None = None) -> list[Condition]:
    """The four Part B tests. Unseen products join the familiar mix, so P5 or P6
    makes up a fifth of the jobs."""
    cfg = config or EnvConfig()
    rate = cfg.arrival_rate
    alpha, beta, _ = EVAL_WEIGHTS
    return [
        Condition("familiar", "1. Normal day, familiar products P1-P4", FAMILIAR, rate, EVAL_WEIGHTS),
        Condition("unseen_p5", "2a. P1-P4 plus unseen P5 (inside the training range)", (*FAMILIAR, "P5"), rate, EVAL_WEIGHTS),
        Condition("unseen_p6", "2b. P1-P4 plus unseen P6 (outside the training range)", (*FAMILIAR, "P6"), rate, EVAL_WEIGHTS),
        Condition("busy", "3. Busy day, arrival rate doubled", FAMILIAR, 2 * rate, EVAL_WEIGHTS),
        Condition("high_damage", "4. Damage weight raised to the top of its range", FAMILIAR, rate, (alpha, beta, cfg.weights.delta[1])),
    ]


def evaluate_conditions(
    policies: Mapping[str, Policy | Sequence[Policy]],
    config: EnvConfig | None = None,
    n_shifts: int = 100,
    seed: int = TEST_SEED,
) -> list[dict]:
    """One row per (condition, policy, run, shift). A sequence of policies is one
    policy name trained with several seeds; `run` indexes them."""
    cfg = config or EnvConfig()
    rows = []
    for cond in conditions(cfg):
        cond_cfg = replace(cfg, arrival_rate=cond.arrival_rate)
        instances = make_instances(cond_cfg, cond.products, n_shifts, seed)
        for name, runs in policies.items():
            for run, policy in enumerate(runs if isinstance(runs, Sequence) else [runs]):
                for shift, metrics in enumerate(evaluate(policy, instances, cond_cfg, cond.weights)):
                    rows.append({"condition": cond.name, "policy": name, "run": run, "shift": shift} | metrics)
    return rows
