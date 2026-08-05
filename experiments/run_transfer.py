"""Experiment: transfer learning with Q-Learning.

    python experiments/run_transfer.py
    python experiments/run_transfer.py --episodes 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import paths
from experiments import common
from transfer.transfer_learning import run_transfer_study  # noqa: F401  (re-exported)

ALGORITHM = "transfer"

SUMMARY_COLUMNS = (
    "variant",
    "scenario",
    "beta",
    "seed",
    "episodes",
    "zero_shot_success",
    "zero_shot_return",
    "early_success",
    "early_return",
    "episodes_to_threshold",
    "final_success",
    "final_return",
    "final_steps",
    "jumpstart_delta",
    "early_delta",
    "final_delta",
    "speed_delta",
    "verdict",
    "q_shift_mean_abs",
    "train_seconds",
)


def summarise(payload: dict, verbose: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{key: record.get(key) for key in SUMMARY_COLUMNS} for record in payload["records"]]
    )
    frame["episodes_to_threshold"] = pd.to_numeric(
        frame["episodes_to_threshold"], errors="coerce"
    )
    frame.to_csv(common.raw_path(ALGORITHM, "transfer_summary.csv"), index=False)

    averaged = (
        frame.groupby(["variant", "scenario"])[
            [
                "zero_shot_success",
                "early_success",
                "final_success",
                "episodes_to_threshold",
                "final_delta",
                "early_delta",
            ]
        ]
        .mean()
        .round(3)
    )
    averaged["verdict"] = (
        frame.groupby(["variant", "scenario"])["verdict"]
        .agg(lambda values: values.mode().iat[0])
    )
    averaged.to_csv(common.raw_path(ALGORITHM, "transfer_by_scenario.csv"))

    if verbose:
        print()
        print("mean over seeds, per target and scenario:")
        print(averaged.to_string())
    return frame


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the transfer learning experiment")
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    parser.add_argument("--episodes", type=int, default=None, help="target episodes per run")
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    common.ensure_map(config, verbose=False)

    common.banner("Transfer learning (Q-Learning only)")
    payload = run_transfer_study(config, episodes=args.episodes, verbose=True)
    common.save_json(payload, common.raw_path(ALGORITHM, "transfer_results.json"))
    summarise(payload)


if __name__ == "__main__":
    main()
