"""Train masked double dueling DQN and evaluate the best checkpoint on the 100 test shifts.

    uv run python scripts/train_dqn.py --timesteps 300000 --seed 0 --device auto
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from lvhm.agents.common import TEST_SEED
from lvhm.agents.dqn import DQNConfig, load_dqn, run_dqn
from lvhm.config import FAMILIAR, EnvConfig
from lvhm.evaluate import evaluate, summarise
from lvhm.instance import make_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    for f in fields(DQNConfig):
        parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=f.default)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=25_000)
    parser.add_argument("--test-shifts", type=int, default=100)
    parser.add_argument("--out", default="runs/dqn")
    args = parser.parse_args()
    dqn_cfg = DQNConfig(**{f.name: getattr(args, f.name) for f in fields(DQNConfig)})

    cfg = EnvConfig()
    out = Path(args.out) / f"seed{args.seed}"
    info = run_dqn(cfg, out, seed=args.seed, dqn_config=dqn_cfg, val_every=args.val_every)

    policy = load_dqn(out / "best_model.pt", cfg, hidden=dqn_cfg.hidden)
    results = evaluate(policy, make_instances(cfg, FAMILIAR, args.test_shifts, TEST_SEED), cfg)
    summary = summarise(results)
    (out / "test_results.json").write_text(json.dumps(info | {"test_summary": summary, "test_shifts": results}, indent=2))
    print(json.dumps(info))
    print("test: " + ", ".join(f"{k} {m:.2f} ± {s:.2f}" for k, (m, s) in summary.items()))


if __name__ == "__main__":
    main()
