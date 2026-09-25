"""Train masked PPO and evaluate the best checkpoint on the 100 test shifts.

    uv run python scripts/train_ppo.py --timesteps 300000 --seed 0 --device auto
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lvhm.agents.common import TEST_SEED
from lvhm.agents.ppo import PPO_DEFAULTS, load_ppo, train_ppo
from lvhm.config import FAMILIAR, EnvConfig
from lvhm.evaluate import evaluate, summarise
from lvhm.instance import make_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    for name, value in PPO_DEFAULTS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    parser.add_argument("--val-every", type=int, default=25_000)
    parser.add_argument("--test-shifts", type=int, default=100)
    parser.add_argument("--out", default="runs/ppo")
    args = parser.parse_args()

    cfg = EnvConfig()
    out = Path(args.out) / f"seed{args.seed}"
    info = train_ppo(
        cfg,
        out,
        seed=args.seed,
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        hidden=args.hidden,
        device=args.device,
        val_every=args.val_every,
        **{name: getattr(args, name) for name in PPO_DEFAULTS},
    )

    results = evaluate(load_ppo(out / "best_model.zip"), make_instances(cfg, FAMILIAR, args.test_shifts, TEST_SEED), cfg)
    summary = summarise(results)
    (out / "test_results.json").write_text(json.dumps(info | {"test_summary": summary, "test_shifts": results}, indent=2))
    print(json.dumps(info))
    print("test: " + ", ".join(f"{k} {m:.2f} ± {s:.2f}" for k, (m, s) in summary.items()))


if __name__ == "__main__":
    main()
