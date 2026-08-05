"""Experiment: model-based Value Iteration.

Value Iteration is deterministic given the maze and the reward mode, so there is
no seed loop here. What it does provide is the optimal reference that the two
model-free learners are measured against, plus a proof of convergence in the
form of the Bellman optimality residual.

    python experiments/run_value_iteration.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths
from agents.base import evaluate_policy, rollout
from experiments import common

ALGORITHM = "value_iteration"


def run(config: dict, reward_modes, verbose: bool = True) -> dict:
    records = []
    for reward_mode in reward_modes:
        env = common.build_env(config, reward_mode=reward_mode, seed=0)
        agent = common.make_planner(env, config)
        if verbose:
            print(f"  [{ALGORITHM}] {reward_mode}: solving {env.n_passable} cells x 2 x "
                  f"{env.max_energy} energy levels")

        stats = agent.train(verbose=verbose)
        evaluation = evaluate_policy(
            env,
            agent,
            episodes=config["training"]["final_eval_episodes"],
            seed=config["training"]["eval_seed"],
        )
        trace = rollout(env, agent, seed=config["training"]["eval_seed"])
        model_file = common.model_path(ALGORITHM, f"{ALGORITHM}_{reward_mode}.npz")
        agent.save(model_file)

        if verbose:
            print(
                f"    -> success {evaluation['success_rate']:.3f} | "
                f"return {evaluation['mean_return']:8.2f} | "
                f"steps {evaluation['mean_steps']:6.1f} | "
                f"energy left {evaluation['mean_energy_left']:.1f}"
            )

        records.append(
            {
                "algorithm": ALGORITHM,
                "reward_mode": reward_mode,
                "seed": None,
                "label": reward_mode,
                "episodes": 0,
                "hyperparameters": agent.hyperparameters(),
                "train_seconds": stats["train_seconds"],
                "memory_kilobytes": stats["memory_kilobytes"],
                "train_stats": stats,
                "final_eval": evaluation,
                "example_trajectory": {
                    "states": [list(state) for state in trace["states"]],
                    "actions": trace["actions"],
                    "total_reward": trace["total_reward"],
                    "steps": trace["steps"],
                    "outcome": trace["outcome"],
                },
                "model_file": paths.rel(model_file),
            }
        )

    payload = {"run": common.run_stamp(config), "records": records}
    common.save_json(payload, common.raw_path(ALGORITHM, f"{ALGORITHM}_results.json"))
    frame = common.write_summary(records, ALGORITHM)
    if verbose:
        print()
        print(frame.to_string(index=False))
    return payload


def run_gamma_sweep(config: dict, reward_modes, verbose: bool = True) -> dict:
    """Re-solve the same maze under several discount factors.

    Gamma decides how far ahead the plan looks. With a step cost and a distant
    goal, a low gamma discounts the goal reward so heavily that reaching it stops
    being worth the walk, so this sweep shows where that break-even sits.
    """
    gammas = config["value_iteration"].get("gamma_sweep", [0.90, 0.95, 0.99])
    records = []

    for reward_mode in reward_modes:
        for gamma in gammas:
            env = common.build_env(config, reward_mode=reward_mode, seed=0)
            agent = common.make_planner(env, config, gamma=gamma)
            stats = agent.train(verbose=False)
            evaluation = evaluate_policy(
                env,
                agent,
                episodes=config["training"]["final_eval_episodes"],
                seed=config["training"]["eval_seed"],
            )
            if verbose:
                print(
                    f"    gamma {gamma:.2f} [{reward_mode}] -> "
                    f"success {evaluation['success_rate']:.3f} | "
                    f"return {evaluation['mean_return']:8.2f} | "
                    f"steps {evaluation['mean_steps']:6.1f} | "
                    f"residual {stats['bellman_residual']:.2e} | "
                    f"{stats['train_seconds']:.1f}s"
                )
            records.append(
                {
                    "algorithm": ALGORITHM,
                    "reward_mode": reward_mode,
                    "gamma": gamma,
                    "seed": None,
                    "label": f"{reward_mode}_gamma{gamma}",
                    "episodes": 0,
                    "iterations": stats["iterations"],
                    "theta": stats["theta"],
                    "bellman_residual": stats["bellman_residual"],
                    "memory_kilobytes": stats["memory_kilobytes"],
                    "train_seconds": stats["train_seconds"],
                    "final_eval": evaluation,
                }
            )

    payload = {"run": common.run_stamp(config), "gammas": list(gammas), "records": records}
    common.save_json(payload, common.raw_path(ALGORITHM, f"{ALGORITHM}_gamma_sweep.json"))

    frame = common.summary_frame(records)
    frame["gamma"] = [record["gamma"] for record in records]
    frame["iterations"] = [record["iterations"] for record in records]
    frame["bellman_residual"] = [record["bellman_residual"] for record in records]
    frame.to_csv(common.raw_path(ALGORITHM, f"{ALGORITHM}_gamma_sweep.csv"), index=False)
    if verbose:
        print()
        print(frame.to_string(index=False))
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the Value Iteration experiment")
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    parser.add_argument(
        "--reward-modes", nargs="*", default=None, help="sparse and/or shaped"
    )
    parser.add_argument(
        "--skip-gamma-sweep", action="store_true", help="only solve at the config gamma"
    )
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    common.ensure_map(config, verbose=True)
    reward_modes = args.reward_modes or config["reward_modes"]

    common.banner("Value Iteration (model-based)")
    run(config, reward_modes)

    if not args.skip_gamma_sweep:
        common.banner("Value Iteration: discount factor sweep")
        run_gamma_sweep(config, reward_modes)


if __name__ == "__main__":
    main()
