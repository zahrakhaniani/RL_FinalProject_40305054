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
            # One sample of individual Q-updates per reward mode is plenty for the
            # report, so only the first seed records them.
            sample_updates = seed == seeds[0]
            records.append(
                common.train_learner(
                    ALGORITHM,
                    config,
                    reward_mode=reward_mode,
                    seed=seed,
                    overrides={"q_log_max": 500 if sample_updates else 0},
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


def run_epsilon_study(
    config: dict, reward_mode: str, seeds, episodes=None, verbose: bool = True
) -> dict:
    """Same agent, two ways of cooling epsilon down.

    Exponential decay spends a long tail at a low epsilon, while linear decay
    holds a higher epsilon for longer and then stops exploring abruptly. Both
    reach the same floor, so the comparison isolates the shape of the schedule.
    """
    schedules = config.get("epsilon_schedules", ["exponential", "linear"])
    records = []

    for schedule in schedules:
        for seed in seeds:
            records.append(
                common.train_learner(
                    ALGORITHM,
                    config,
                    reward_mode=reward_mode,
                    seed=seed,
                    overrides={"epsilon_schedule": schedule, "q_log_max": 0},
                    episodes=episodes,
                    tag=f"{reward_mode}_eps-{schedule}_seed{seed}",
                    verbose=verbose,
                )
            )
            records[-1]["epsilon_schedule"] = schedule

    payload = {
        "run": common.run_stamp(config),
        "reward_mode": reward_mode,
        "schedules": list(schedules),
        "records": records,
    }
    common.save_json(
        payload, common.raw_path(ALGORITHM, f"{ALGORITHM}_epsilon_study_{reward_mode}.json")
    )

    frame = common.summary_frame(records)
    frame["epsilon_schedule"] = [record["epsilon_schedule"] for record in records]
    frame.to_csv(
        common.raw_path(ALGORITHM, f"{ALGORITHM}_epsilon_study_{reward_mode}.csv"), index=False
    )
    if verbose:
        print()
        print(
            frame.groupby("epsilon_schedule")[
                ["success_rate", "mean_return", "mean_steps", "episodes_to_threshold"]
            ]
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
    parser.add_argument(
        "--skip-epsilon-study", action="store_true", help="only run the main training grid"
    )
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    common.ensure_map(config, verbose=False)
    reward_modes = args.reward_modes or config["reward_modes"]
    seeds = args.seeds or config["training"]["seeds"]

    common.banner("Q-Learning (model-free, off-policy)")
    run(config, reward_modes, seeds, episodes=args.episodes)

    if not args.skip_epsilon_study:
        for reward_mode in reward_modes:
            common.banner(f"Q-Learning: epsilon decay comparison ({reward_mode})")
            run_epsilon_study(config, reward_mode, seeds, episodes=args.episodes)


if __name__ == "__main__":
    main()
