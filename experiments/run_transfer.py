"""Experiment: transfer learning from the original maze to a new one.

    python experiments/run_transfer.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import paths
from experiments import common
from transfer.transfer_learning import run_transfer_study

ALGORITHM = "transfer"


def summarise(records) -> pd.DataFrame:
    rows = []
    for record in records:
        evaluation = record["final_eval"]
        metrics = record["transfer_metrics"]
        rows.append(
            {
                "strategy": record["strategy"],
                "reward_mode": record["reward_mode"],
                "seed": record["seed"],
                "episodes": record["episodes"],
                "zero_shot_success": metrics["zero_shot_success"],
                "zero_shot_key_rate": metrics["zero_shot_key_rate"],
                "zero_shot_return": metrics["zero_shot_return"],
                "first_eval_success": metrics["jumpstart_success"],
                "episodes_to_threshold": metrics["episodes_to_threshold"],
                "mean_eval_success": metrics["mean_eval_success"],
                "final_success_rate": evaluation["success_rate"],
                "final_mean_return": evaluation["mean_return"],
                "final_mean_steps": evaluation["mean_steps"],
                "train_seconds": record["train_seconds"],
            }
        )
    frame = pd.DataFrame(rows)
    # "never reached the threshold" is stored as None; make it a real NaN so the
    # column stays numeric and aggregates sensibly.
    frame["episodes_to_threshold"] = pd.to_numeric(
        frame["episodes_to_threshold"], errors="coerce"
    )
    return frame


def run(config: dict, reward_mode: str, seeds=None, episodes=None, verbose=True) -> dict:
    payload = run_transfer_study(
        config, reward_mode=reward_mode, seeds=seeds, episodes=episodes, verbose=verbose
    )
    common.save_json(payload, common.raw_path(ALGORITHM, f"{ALGORITHM}_results.json"))

    frame = summarise(payload["records"])
    frame.to_csv(common.raw_path(ALGORITHM, f"{ALGORITHM}_summary.csv"), index=False)
    if verbose:
        print()
        print(frame.to_string(index=False))
        print()
        print("mean over seeds:")
        print(
            frame.groupby("strategy")[
                [
                    "zero_shot_success",
                    "zero_shot_key_rate",
                    "mean_eval_success",
                    "episodes_to_threshold",
                    "final_success_rate",
                    "final_mean_return",
                ]
            ]
            .mean()
            .to_string()
        )
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the transfer learning experiment")
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    parser.add_argument("--reward-mode", default="shaped")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    common.ensure_map(config, verbose=False)

    common.banner("Transfer learning (source maze -> new maze)")
    run(
        config,
        reward_mode=args.reward_mode,
        seeds=args.seeds,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()
