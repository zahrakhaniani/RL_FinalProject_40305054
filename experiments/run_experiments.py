"""Run every experiment end to end, then build all figures and tables.

    python experiments/run_experiments.py
    python experiments/run_experiments.py --quick        # small smoke run
    python experiments/run_experiments.py --only q_learning transfer
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths
from experiments import (
    analysis,
    common,
    run_comparison,
    run_q_learning,
    run_sarsa_lambda,
    run_transfer,
    run_value_iteration,
)

STAGES = ("value_iteration", "q_learning", "sarsa_lambda", "comparison", "transfer")

QUICK_OVERRIDES = {
    "episodes": 300,
    "seeds": [1, 2],
    "eval_every": 50,
    "eval_episodes": 10,
    "final_eval_episodes": 50,
    "transfer_episodes": 300,
    "transfer_seeds": [1],
}


def apply_quick(config: dict) -> dict:
    """Shrink every loop so the whole pipeline runs in a couple of minutes."""
    config["training"]["eval_every"] = QUICK_OVERRIDES["eval_every"]
    config["training"]["eval_episodes"] = QUICK_OVERRIDES["eval_episodes"]
    config["training"]["final_eval_episodes"] = QUICK_OVERRIDES["final_eval_episodes"]
    config["training"]["seeds"] = QUICK_OVERRIDES["seeds"]
    for algorithm in ("q_learning", "sarsa_lambda"):
        config[algorithm]["episodes"] = QUICK_OVERRIDES["episodes"]
    config["lambda_sweep"] = [0.0, 0.7, 0.9]
    config["transfer"]["episodes"] = QUICK_OVERRIDES["transfer_episodes"]
    config["transfer"]["seeds"] = QUICK_OVERRIDES["transfer_seeds"]
    return config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the full experiment suite")
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    parser.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    parser.add_argument(
        "--only", nargs="*", choices=STAGES, default=None, help="run a subset of stages"
    )
    parser.add_argument("--skip-sweeps", action="store_true",
                        help="skip the gamma, epsilon and lambda studies")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--regenerate-map", action="store_true")
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    if args.quick:
        config = apply_quick(config)

    common.banner("Shared maze")
    common.ensure_map(config, force=args.regenerate_map, verbose=True)

    stages = args.only or STAGES
    reward_modes = config["reward_modes"]
    seeds = config["training"]["seeds"]
    sweeps = not args.skip_sweeps
    started = time.perf_counter()

    if "value_iteration" in stages:
        common.banner("1/5  Value Iteration (model-based)")
        run_value_iteration.run(config, reward_modes)
        if sweeps:
            common.banner("1/5  Value Iteration -- discount factor sweep")
            run_value_iteration.run_gamma_sweep(config, reward_modes)

    if "q_learning" in stages:
        common.banner("2/5  Q-Learning (model-free, off-policy)")
        run_q_learning.run(config, reward_modes, seeds)
        if sweeps:
            for reward_mode in reward_modes:
                common.banner(f"2/5  Q-Learning -- epsilon decay comparison ({reward_mode})")
                run_q_learning.run_epsilon_study(config, reward_mode, seeds)

    if "sarsa_lambda" in stages:
        common.banner("3/5  SARSA(lambda) (model-free, on-policy)")
        run_sarsa_lambda.run(config, reward_modes, seeds)
        if sweeps:
            for reward_mode in reward_modes:
                common.banner(f"3/5  SARSA(lambda) -- lambda sweep ({reward_mode} rewards)")
                run_sarsa_lambda.run_lambda_sweep(config, reward_mode=reward_mode)

    if "comparison" in stages:
        common.banner("4/5  Algorithm comparison and policy agreement")
        run_comparison.run(config, reward_modes, seeds)

    if "transfer" in stages:
        common.banner("5/5  Transfer learning (Q-Learning only)")
        payload = run_transfer.run_transfer_study(config, verbose=True)
        common.save_json(payload, common.raw_path("transfer", "transfer_results.json"))
        run_transfer.summarise(payload)

    if not args.skip_analysis:
        common.banner("Analysis: figures and comparison tables")
        analysis.main([])

    print()
    print(f"all done in {time.perf_counter() - started:.1f}s")
    print(f"raw data  -> {paths.rel(paths.RAW_DATA)}")
    print(f"models    -> {paths.rel(paths.MODELS)}")
    print(f"figures   -> {paths.rel(paths.FIGURES)}")


if __name__ == "__main__":
    main()
