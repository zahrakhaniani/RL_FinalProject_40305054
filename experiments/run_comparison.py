"""Experiment: Value Iteration vs Q-Learning vs SARSA(lambda).

All three are compared on the same maze, the same reward definition and the same
seeds, so the differences that show up are down to the algorithms rather than the
setup. Alongside the usual success/return/runtime numbers this computes policy
agreement: the share of states where the learner's greedy action is the one
Value Iteration proved optimal.

Agreement is reported twice. The plain number covers every reachable state, most
of which a converged agent rarely visits; the visit-weighted number counts each
state as often as the learner actually stood in it, and is the better measure of
whether the two would behave the same in practice.

    python experiments/run_comparison.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import paths
from agents.base import policy_agreement
from experiments import common

ALGORITHM = "comparison"
LEARNERS = ("q_learning", "sarsa_lambda")


def run(config: dict, reward_modes, seeds, verbose: bool = True) -> dict:
    rows = []
    difference_maps = {}

    for reward_mode in reward_modes:
        planner, env = common.load_trained_planner(config, reward_mode)
        if planner is None:
            raise FileNotFoundError(
                "missing the Value Iteration model -- "
                "run experiments/run_value_iteration.py first"
            )
        planner_results = common.load_json(
            common.raw_path("value_iteration", "value_iteration_results.json")
        )
        planner_record = next(
            record
            for record in planner_results["records"]
            if record["reward_mode"] == reward_mode
        )
        planner_eval = planner_record["final_eval"]
        rows.append(
            {
                "algorithm": "value_iteration",
                "reward_mode": reward_mode,
                "seed": None,
                "episodes_or_iterations": planner_record["train_stats"]["iterations"],
                "success_rate": planner_eval["success_rate"],
                "mean_return": planner_eval["mean_return"],
                "mean_steps_when_solved": planner_eval["mean_steps_when_solved"],
                "mean_wall_collisions": planner_eval.get("mean_wall_collisions"),
                "mean_penalty_entries": planner_eval.get("mean_penalty_entries"),
                "train_seconds": planner_record["train_seconds"],
                "memory_kilobytes": planner_record["hyperparameters"]["memory_kilobytes"],
                "policy_agreement": 1.0,
                "weighted_policy_agreement": 1.0,
                "late_return_std": float("nan"),
            }
        )

        for algorithm in LEARNERS:
            results = common.load_json(common.raw_path(algorithm, f"{algorithm}_results.json"))
            records = {
                (record["reward_mode"], record["seed"]): record
                for record in results["records"]
            }

            for seed in seeds:
                record = records.get((reward_mode, seed))
                agent, _ = common.load_trained_learner(
                    config, algorithm, reward_mode, seed
                )
                if record is None or agent is None:
                    continue

                agreement = policy_agreement(
                    planner, agent, env, weights=agent.visit_counts
                )
                evaluation = record["final_eval"]
                rows.append(
                    {
                        "algorithm": algorithm,
                        "reward_mode": reward_mode,
                        "seed": seed,
                        "episodes_or_iterations": record["episodes"],
                        "success_rate": evaluation["success_rate"],
                        "mean_return": evaluation["mean_return"],
                        "mean_steps_when_solved": evaluation["mean_steps_when_solved"],
                        "mean_wall_collisions": evaluation.get("mean_wall_collisions"),
                        "mean_penalty_entries": evaluation.get("mean_penalty_entries"),
                        "train_seconds": record["train_seconds"],
                        "memory_kilobytes": record.get("memory_kilobytes"),
                        "policy_agreement": agreement["agreement"],
                        "weighted_policy_agreement": agreement["weighted_agreement"],
                        "late_return_std": record.get("stability", {}).get("late_return_std"),
                    }
                )

                if seed == seeds[0]:
                    key = f"{reward_mode}__{algorithm}"
                    difference_maps[key] = agreement["per_cell_full_battery"]
                    difference_maps[f"{key}__visits"] = agent.visit_grid()

                if verbose:
                    print(
                        f"  {algorithm:<13} {reward_mode:<7} seed {seed} | "
                        f"agreement {agreement['agreement']:.3f} "
                        f"(visit-weighted {agreement['weighted_agreement']:.3f})"
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(common.raw_path(ALGORITHM, "algorithm_comparison.csv"), index=False)

    averaged = (
        frame.groupby(["reward_mode", "algorithm"])[
            [
                "success_rate",
                "mean_return",
                "mean_steps_when_solved",
                "mean_penalty_entries",
                "policy_agreement",
                "weighted_policy_agreement",
                "train_seconds",
                "memory_kilobytes",
                "late_return_std",
            ]
        ]
        .mean()
        .round(3)
    )
    averaged.to_csv(common.raw_path(ALGORITHM, "algorithm_comparison_mean.csv"))

    map_file = common.raw_path(ALGORITHM, "policy_difference_maps.npz")
    np.savez_compressed(map_file, **difference_maps)

    payload = {
        "run": common.run_stamp(config),
        "rows": rows,
        "difference_maps_file": paths.rel(map_file),
    }
    common.save_json(payload, common.raw_path(ALGORITHM, "algorithm_comparison.json"))

    if verbose:
        print()
        print(averaged.to_string())
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Compare the three algorithms")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reward-modes", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    common.ensure_map(config, verbose=False)
    reward_modes = args.reward_modes or config["reward_modes"]
    seeds = args.seeds or config["training"]["seeds"]

    common.banner("Algorithm comparison and policy agreement")
    run(config, reward_modes, seeds)


if __name__ == "__main__":
    main()
