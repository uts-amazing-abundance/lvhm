"""Run the six dispatch rules on fixed evaluation shifts and print a results table.

Doubles as the pre-training gap check: if the rules barely differ, job ordering has
little room to matter and the environment parameters need recalibrating.

    uv run python scripts/baseline_check.py --busy
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

from lvhm.baselines import rule_baselines
from lvhm.config import FAMILIAR, EnvConfig
from lvhm.evaluate import METRICS, evaluate, summarise
from lvhm.instance import make_instances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shifts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--busy", action="store_true", help="double the arrival rate")
    parser.add_argument("--products", nargs="+", default=list(FAMILIAR))
    args = parser.parse_args()

    cfg = EnvConfig()
    if args.busy:
        cfg = replace(cfg, arrival_rate=2 * cfg.arrival_rate)
    instances = make_instances(cfg, args.products, args.shifts, args.seed)
    jobs = sum(len(i) for i in instances) / len(instances)
    print(f"{args.shifts} shifts, products {args.products}, arrival rate {cfg.arrival_rate}/s, {jobs:.1f} jobs/shift\n")

    print("| Policy | " + " | ".join(METRICS) + " |")
    print("|---" * (len(METRICS) + 1) + "|")
    steps, start = 0, time.perf_counter()
    for name, policy in rule_baselines(seed=args.seed).items():
        results = evaluate(policy, instances, cfg)
        steps += sum(r["decisions"] for r in results)
        cells = [f"{mean:.2f} ± {std:.2f}" for mean, std in summarise(results).values()]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print(f"\n{steps / (time.perf_counter() - start):,.0f} env steps/s (single process, rule policies)")


if __name__ == "__main__":
    main()
