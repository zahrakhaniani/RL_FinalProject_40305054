"""Experiment: model-free SARSA(lambda).

Two studies live here:

1. the main run, one agent per (reward mode, seed), directly comparable to the
   Q-Learning experiment;
2. a lambda sweep, because lambda is the whole point of this algorithm and its
   best value is not obvious on a task with episodes this long.

    python experiments/run_sarsa_lambda.py
    python experiments/run_sarsa_lambda.py --skip-sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import paths
from experiments import common

ALGORITHM = "sarsa_lambda"


def run(config: dict, reward_modes, seeds, episodes=None, verbose: bool = True) -> dict:
    records = []
    for reward_mode in reward_modes:
        for seed in seeds:
            records.append(
                common.train_learner(
                    ALGORITHM,
                    config,
                    reward_mode=reward_mode,
                    seed=seed,
                    episodes=episodes,
                    verbose=verbose,
                )
            )

    payload = {"run": common.run_stamp(config), "records": records}
    common.save_json(payload, common.raw_path(ALGORITHM, f"{ALGORITHM}_results.json"))
    frame = common.write_summary(records, ALGORITHM)
    if verbose:
        print()
        print(frame.to_string(index=False))
        print()
        print(
            frame.groupby("reward_mode")[["success_rate", "mean_return", "mean_steps"]]
            .mean()
            .to_string()
        )
    return payload


def run_lambda_sweep(
    config: dict,
    reward_mode: str = "shaped",
    seeds=None,
    episodes=None,
    verbose: bool = True,
) -> dict:
    """How the trace decay parameter changes what SARSA(lambda) converges to."""
    lambdas = config["lambda_sweep"]
    seeds = seeds or config["training"]["seeds"][:3]
    records = []

    for lam in lambdas:
        for seed in seeds:
            records.append(
                common.train_learner(
                    ALGORITHM,
                    config,
                    reward_mode=reward_mode,
                    seed=seed,
                    overrides={"lam": lam},
                    episodes=episodes,
                    tag=f"sweep_lam{lam}_seed{seed}",
                    verbose=verbose,
                )
            )
            records[-1]["lam"] = lam

    payload = {"run": common.run_stamp(config), "reward_mode": reward_mode, "records": records}
    common.save_json(
        payload, common.raw_path(ALGORITHM, f"{ALGORITHM}_lambda_sweep_{reward_mode}.json")
    )

    frame = common.summary_frame(records)
    frame["lam"] = [record["lam"] for record in records]
    frame.to_csv(
        common.raw_path(ALGORITHM, f"{ALGORITHM}_lambda_sweep_{reward_mode}.csv"),
        index=False,
    )
    if verbose:
        print()
        print(f"lambda sweep, {reward_mode} rewards (mean over seeds):")
        print(
            frame.groupby("lam")[["success_rate", "mean_return", "mean_steps"]]
            .mean()
            .to_string()
        )
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the SARSA(lambda) experiment")
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    parser.add_argument("--reward-modes", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--skip-sweep", action="store_true", help="skip the lambda sweep")
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    common.ensure_map(config, verbose=False)
    reward_modes = args.reward_modes or config["reward_modes"]
    seeds = args.seeds or config["training"]["seeds"]

    common.banner("SARSA(lambda) (model-free, on-policy, eligibility traces)")
    run(config, reward_modes, seeds, episodes=args.episodes)

    if not args.skip_sweep:
        for reward_mode in reward_modes:
            common.banner(f"SARSA(lambda) -- lambda sweep ({reward_mode} rewards)")
            run_lambda_sweep(config, reward_mode=reward_mode, episodes=args.episodes)


if __name__ == "__main__":
    main()
