"""Rule-based dispatch baselines (Part B, Table 4).

Each rule sees the same K visible jobs as the agent and always picks in normal mode.
Ties go to the earliest arrival, since slots are ordered by arrival.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from lvhm.env import DecisionState, LVHMSchedulingEnv

Policy = Callable[[np.ndarray, LVHMSchedulingEnv], int]


def _rule(key: Callable[[DecisionState], np.ndarray]) -> Policy:
    def policy(obs: np.ndarray, env: LVHMSchedulingEnv) -> int:
        return int(np.argmin(key(env.decision_state())))

    return policy


fcfs = _rule(lambda s: s.arrival)
spt = _rule(lambda s: s.proc)
lrf = _rule(lambda s: s.fragility)
edf = _rule(lambda s: s.deadline)
ntf = _rule(lambda s: (s.tray != s.arm).astype(float))


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs: np.ndarray, env: LVHMSchedulingEnv) -> int:
        valid = np.flatnonzero(env.action_masks()[: env.cfg.num_slots])
        return int(self.rng.choice(valid))


def rule_baselines(seed: int = 0) -> dict[str, Policy]:
    return {"FCFS": fcfs, "SPT": spt, "LRF": lrf, "EDF": edf, "NTF": ntf, "Random": RandomPolicy(seed)}
