"""Validation shared by all learned agents: score a policy on fixed shifts, keep the
best checkpoint, and log the learning curve."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from lvhm.baselines import Policy
from lvhm.config import EVAL_WEIGHTS, EnvConfig
from lvhm.evaluate import evaluate, summarise
from lvhm.instance import ShiftInstance

VAL_SEED, TEST_SEED = 20_000, 10_000


class Validator:
    def __init__(
        self,
        instances: Sequence[ShiftInstance],
        config: EnvConfig,
        out_dir: Path,
        weights: Sequence[float] = EVAL_WEIGHTS,
        verbose: bool = True,
    ):
        self.instances, self.config, self.weights = instances, config, weights
        self.out_dir, self.verbose = out_dir, verbose
        self.best = -np.inf
        self.rows: list[dict] = []

    def __call__(self, policy: Policy, timesteps: int, save_best: Callable[[], None]) -> dict:
        summary = summarise(evaluate(policy, self.instances, self.config, self.weights))
        row = {"timesteps": timesteps} | {k: round(v[0], 3) for k, v in summary.items()}
        self.rows.append(row)
        if row["score"] > self.best:
            self.best = row["score"]
            save_best()
        with open(self.out_dir / "validation.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.rows)
        if self.verbose:
            print(json.dumps(row), flush=True)
        return row
