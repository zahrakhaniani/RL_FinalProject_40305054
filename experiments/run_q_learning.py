"""Experiment: model-free Q-Learning.

Trains one agent per (reward mode, seed) pair so the learning curves can be
reported with a spread across seeds rather than a single lucky run.

    python experiments/run_q_learning.py
    python experiments/run_q_learning.py --episodes 500 --seeds 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths
from experiments import common

ALGORITHM = "q_learning"


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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the Q-Learning experiment")
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    parser.add_argument("--reward-modes", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    common.ensure_map(config, verbose=False)
    reward_modes = args.reward_modes or config["reward_modes"]
    seeds = args.seeds or config["training"]["seeds"]

    common.banner("Q-Learning (model-free, off-policy)")
    run(config, reward_modes, seeds, episodes=args.episodes)


if __name__ == "__main__":
    main()
