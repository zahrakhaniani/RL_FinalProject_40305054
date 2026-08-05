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
from transfer.transfer_learning import (  # noqa: F401  (run_transfer_study re-exported)
    classify_transfer,
    run_transfer_study,
)

ALGORITHM = "transfer"

SUMMARY_COLUMNS = (
    "variant",
    "scenario",
    "beta",
    "seed",
    "episodes",
    "zero_shot_success",
    "zero_shot_key_rate",
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

    averaged = scenario_table(frame)
    averaged.to_csv(common.raw_path(ALGORITHM, "transfer_by_scenario.csv"), index=False)

    if verbose:
        print()
        print("mean over seeds, per target and scenario:")
        print(averaged.to_string(index=False))
    return frame


def scenario_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged results per scenario, classified against the scratch baseline.

    Individual seeds are noisy enough that a majority vote over per-seed verdicts
    can disagree with the averaged deltas, so the headline verdict is decided from
    the seed means instead.
    """
    columns = [
        "zero_shot_success",
        "zero_shot_key_rate",
        "early_success",
        "final_success",
        "episodes_to_threshold",
        "episodes",
    ]
    averaged = frame.groupby(["variant", "scenario"])[columns].mean().reset_index()
    spread = (
        frame.groupby(["variant", "scenario"])["final_success"]
        .std()
        .reset_index(name="final_success_std")
    )
    averaged = averaged.merge(spread, on=["variant", "scenario"])

    rows = []
    for variant, group in averaged.groupby("variant"):
        baseline = group[group["scenario"] == "scratch"].iloc[0].to_dict()
        for record in group.to_dict("records"):
            rows.append({**record, **classify_transfer(record, baseline)})

    table = pd.DataFrame(rows).drop(columns=["episodes"])
    return table.round(3).sort_values(["variant", "scenario"])


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the transfer learning experiment")
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    parser.add_argument("--episodes", type=int, default=None, help="target episodes per run")
    parser.add_argument(
        "--summarise-only",
        action="store_true",
        help="rebuild the tables from the saved results without retraining",
    )
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    common.ensure_map(config, verbose=False)

    results = common.raw_path(ALGORITHM, "transfer_results.json")
    if args.summarise_only:
        payload = common.load_json(results)
        for record in payload["records"]:
            record.update(
                classify_transfer(
                    record,
                    next(
                        other for other in payload["records"]
                        if other["variant"] == record["variant"]
                        and other["seed"] == record["seed"]
                        and other["scenario"] == "scratch"
                    ),
                )
            )
        common.save_json(payload, results)
        summarise(payload)
        return

    common.banner("Transfer learning (Q-Learning only)")
    payload = run_transfer_study(config, episodes=args.episodes, verbose=True)
    common.save_json(payload, results)
    summarise(payload)


if __name__ == "__main__":
    main()
