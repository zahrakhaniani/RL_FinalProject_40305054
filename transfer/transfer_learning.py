"""Transfer learning for Q-Learning.

A Q-table trained on the source maze is reused on two perturbed mazes, and the
only thing that varies is *how much of it* is carried over:

``scratch``
    zero-initialised table, the baseline every other scenario is measured against.
``full``
    copy the source table verbatim.
``scaled_<beta>``
    copy the source table multiplied by beta, so the prior is still a hint about
    which direction to walk but is weak enough for a few real rewards to override.
``selective``
    copy only the cells whose 3x3 neighbourhood of walls is identical in both
    mazes, and leave the rest at zero.

Every scenario is evaluated greedily *before* any target training, which is what
makes transfer measurable: that zero-shot number is the initial performance the
prior gives for free, and comparing it and the learning curve against ``scratch``
is what separates positive from negative transfer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths
from agents.base import evaluate_policy
from agents.q_learning import QLearningAgent
from environments.maze import MazeEnv
from environments.variants import (
    make_variant,
    obstacle_difference,
    unchanged_neighbourhood_mask,
)
from experiments import common

ALGORITHM = "transfer"

#: Difference in success rate below which a scenario counts as neutral.
CLASSIFY_TOLERANCE = 0.05

#: Difference in episodes-to-threshold, as a fraction of the budget, that counts.
SPEED_TOLERANCE = 0.10


# --------------------------------------------------------------- scenario setup


def scenario_specs(config: dict) -> List[Tuple[str, dict]]:
    """The four scenarios, with the scaled one expanded over its betas."""
    wanted = config["transfer"]["scenarios"]
    specs: List[Tuple[str, dict]] = []
    for scenario in wanted:
        if scenario == "scaled":
            for beta in config["transfer"]["betas"]:
                specs.append((f"scaled_{beta}", {"family": "scaled", "beta": float(beta)}))
        else:
            specs.append((scenario, {"family": scenario}))
    return specs


def build_initial_q(
    scenario: str,
    options: dict,
    source_q: np.ndarray,
    reuse_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Initial target Q-table for one scenario."""
    family = options.get("family", scenario)
    if family == "scratch":
        return np.zeros_like(source_q)
    if family == "full":
        return source_q.copy()
    if family == "scaled":
        return source_q * float(options["beta"])
    if family == "selective":
        if reuse_mask is None:
            raise ValueError("selective transfer needs a neighbourhood mask")
        initial = np.zeros_like(source_q)
        initial[reuse_mask] = source_q[reuse_mask]
        return initial
    raise KeyError(f"unknown transfer scenario {scenario!r}")


# ------------------------------------------------------------------ environments


def ensure_targets(
    config: dict, source_env: MazeEnv, verbose: bool = True
) -> Dict[str, MazeEnv]:
    """Build (and save) the two target mazes."""
    settings = config["transfer"]
    targets: Dict[str, MazeEnv] = {}

    for variant in settings["variants"]:
        target = make_variant(
            source_env,
            variant,
            seed=int(settings["variant_seed"]),
            reward_mode=settings.get("reward_mode", "shaped"),
            rewards=config["rewards"],
        )
        destination = paths.map_file(config["student_id"], variant=variant)
        target.save_map(destination)
        targets[variant] = target
        if verbose:
            meta = target.metadata
            print(
                f"  target {variant:<9} -> {meta['obstacle_change_fraction']:.0%} of cells changed"
                f" | key moved {str(bool(meta['key_moved'])):<5}"
                f" | penalties {meta['n_penalty_cells']}"
                f" | optimal path {meta['optimal_path_length']}"
                f" | {paths.rel(destination)}"
            )
    return targets


# ---------------------------------------------------------------------- training


def train_source(config: dict, verbose: bool = True) -> Tuple[MazeEnv, QLearningAgent, Path]:
    """Train the Q-Learning agent that every transfer scenario starts from."""
    reward_mode = config["transfer"].get("reward_mode", "shaped")
    seed = int(config["transfer"]["seeds"][0])
    env = common.build_env(config, reward_mode=reward_mode, seed=seed)
    agent = common.make_learner("q_learning", env, config, seed)
    episodes = int(config["q_learning"]["episodes"])

    if verbose:
        print(f"  source training: {episodes} episodes on the original maze")
    agent.train(
        episodes=episodes,
        eval_every=config["training"]["eval_every"],
        eval_episodes=config["training"]["eval_episodes"],
        eval_seed=config["training"]["eval_seed"],
        verbose=False,
    )
    evaluation = evaluate_policy(
        env, agent, episodes=config["training"]["final_eval_episodes"],
        seed=config["training"]["eval_seed"],
    )
    model_file = common.model_path(ALGORITHM, "source_q_learning.npz")
    agent.save(model_file)
    if verbose:
        print(
            f"    source policy -> success {evaluation['success_rate']:.3f} | "
            f"return {evaluation['mean_return']:8.2f} | saved {paths.rel(model_file)}"
        )
    return env, agent, model_file


def run_scenario(
    config: dict,
    variant: str,
    target_env: MazeEnv,
    scenario: str,
    options: dict,
    source_q: np.ndarray,
    reuse_mask: np.ndarray,
    seed: int,
    verbose: bool = True,
) -> dict:
    """One (target, scenario, seed) run, from zero-shot check to final policy."""
    settings = config["transfer"]
    episodes = int(settings["episodes"])
    env = target_env.copy()
    env.reset(seed=seed)

    agent = common.make_learner("q_learning", env, config, seed)
    agent.q = build_initial_q(scenario, options, source_q, reuse_mask)
    q_before = np.stack([agent.max_q_grid(0), agent.max_q_grid(1)])

    # Initial performance: greedy on the target with no target experience at all.
    zero_shot = evaluate_policy(
        env, agent, episodes=config["training"]["eval_episodes"], seed=settings.get("zero_shot_seed", 777_000)
    )

    log = agent.train(
        episodes=episodes,
        eval_every=config["training"]["eval_every"],
        eval_episodes=config["training"]["eval_episodes"],
        eval_seed=config["training"]["eval_seed"],
        verbose=False,
    )
    final = evaluate_policy(
        env, agent, episodes=config["training"]["final_eval_episodes"],
        seed=config["training"]["eval_seed"],
    )
    q_after = np.stack([agent.max_q_grid(0), agent.max_q_grid(1)])
    log_dict = log.to_dict()

    early_window = int(settings.get("early_window", 300))
    early_evals = [
        rate
        for episode, rate in zip(log_dict["eval_episode"], log_dict["eval_success_rate"])
        if episode <= early_window
    ]
    early_returns = log_dict["reward"][:early_window]

    metrics = {
        "variant": variant,
        "scenario": scenario,
        "family": options.get("family", scenario),
        "beta": options.get("beta"),
        "seed": seed,
        "episodes": episodes,
        "zero_shot_success": zero_shot["success_rate"],
        "zero_shot_key_rate": zero_shot["key_rate"],
        "zero_shot_return": zero_shot["mean_return"],
        "early_success": float(np.mean(early_evals)) if early_evals else float("nan"),
        "early_return": float(np.mean(early_returns)) if early_returns else float("nan"),
        "episodes_to_threshold": common.episodes_to_threshold(
            log_dict, threshold=settings.get("success_threshold", 0.8)
        ),
        "final_success": final["success_rate"],
        "final_return": final["mean_return"],
        "final_steps": final["mean_steps"],
        "train_seconds": agent.train_seconds,
        "q_before_mean": float(np.mean(q_before)),
        "q_after_mean": float(np.mean(q_after)),
        "q_shift_mean_abs": float(np.mean(np.abs(q_after - q_before))),
    }

    if verbose:
        print(
            f"    {variant:<9} {scenario:<14} seed {seed} | "
            f"zero-shot {metrics['zero_shot_success']:.2f} | "
            f"early {metrics['early_success']:.2f} | "
            f"final {metrics['final_success']:.2f} | "
            f"to-threshold {metrics['episodes_to_threshold']}"
        )

    return {
        "metrics": metrics,
        "log": log_dict,
        "q_before": q_before,
        "q_after": q_after,
        "agent": agent,
    }


# ------------------------------------------------------------------ classification


def _threshold_or_none(value) -> Optional[float]:
    """A run that never reached the threshold shows up as ``None`` or ``NaN``."""
    if value is None:
        return None
    value = float(value)
    return None if np.isnan(value) else value


def classify_transfer(record: dict, baseline: dict) -> dict:
    """Positive, negative or neutral transfer, against the scratch baseline.

    Decided on numbers, in this order: a scenario that ends up worse than scratch
    is negative regardless of how it started; otherwise a better start, a better
    ending or a clearly earlier arrival at the success threshold makes it
    positive; and anything inside the tolerances is neutral.
    """
    final_delta = record["final_success"] - baseline["final_success"]
    early_delta = record["early_success"] - baseline["early_success"]
    jumpstart_delta = record["zero_shot_success"] - baseline["zero_shot_success"]

    record_threshold = _threshold_or_none(record["episodes_to_threshold"])
    baseline_threshold = _threshold_or_none(baseline["episodes_to_threshold"])
    if record_threshold is None and baseline_threshold is None:
        speed_delta = 0
    elif record_threshold is None:
        speed_delta = -record["episodes"]
    elif baseline_threshold is None:
        speed_delta = record["episodes"]
    else:
        speed_delta = baseline_threshold - record_threshold

    # A speed difference only counts if it is a real slice of the training budget.
    speed_margin = SPEED_TOLERANCE * float(record["episodes"])
    faster = speed_delta > speed_margin
    slower = speed_delta < -speed_margin

    if final_delta < -CLASSIFY_TOLERANCE:
        verdict = "negative"
    elif early_delta > CLASSIFY_TOLERANCE or final_delta > CLASSIFY_TOLERANCE or faster:
        verdict = "positive"
    elif early_delta < -CLASSIFY_TOLERANCE or slower:
        verdict = "negative"
    else:
        verdict = "neutral"

    return {
        "jumpstart_delta": jumpstart_delta,
        "early_delta": early_delta,
        "final_delta": final_delta,
        "speed_delta": speed_delta,
        "verdict": verdict,
    }


# --------------------------------------------------------------------- the study


def run_transfer_study(config: dict, episodes: Optional[int] = None, verbose: bool = True) -> dict:
    settings = config["transfer"]
    if episodes is not None:
        settings = dict(settings)
        settings["episodes"] = int(episodes)
        config = {**config, "transfer": settings}

    source_env, source_agent, source_model = train_source(config, verbose=verbose)
    source_q = source_agent.q.copy()
    targets = ensure_targets(config, source_env, verbose=verbose)

    specs = scenario_specs(config)
    seeds = [int(seed) for seed in settings["seeds"]]
    radius = int(settings.get("neighbourhood_radius", 1))

    records: List[dict] = []
    arrays: Dict[str, np.ndarray] = {}
    target_meta: Dict[str, dict] = {}

    for variant, target_env in targets.items():
        reuse_mask = unchanged_neighbourhood_mask(source_env, target_env, radius=radius)
        target_meta[variant] = {
            "metadata": target_env.metadata,
            "reuse_fraction": float(reuse_mask.mean()),
            "map_file": paths.rel(paths.map_file(config["student_id"], variant=variant)),
        }
        arrays[f"{variant}__obstacle_diff"] = obstacle_difference(source_env, target_env)
        arrays[f"{variant}__reuse_mask"] = reuse_mask.astype(np.int8)

        if verbose:
            print(
                f"  {variant}: selective transfer keeps "
                f"{reuse_mask.mean():.0%} of the cells"
            )

        for scenario, options in specs:
            for seed in seeds:
                outcome = run_scenario(
                    config, variant, target_env, scenario, options,
                    source_q, reuse_mask, seed, verbose=verbose,
                )
                records.append({**outcome["metrics"], "log": outcome["log"]})

                if seed == seeds[0]:
                    key = f"{variant}__{scenario}"
                    arrays[f"{key}__q_before"] = outcome["q_before"]
                    arrays[f"{key}__q_after"] = outcome["q_after"]
                    if scenario in ("scratch", "full", "selective"):
                        outcome["agent"].save(
                            common.model_path(ALGORITHM, f"{variant}_{scenario}_seed{seed}.npz")
                        )

    # Compare every scenario against scratch on the same target and seed.
    baselines = {
        (record["variant"], record["seed"]): record
        for record in records
        if record["scenario"] == "scratch"
    }
    for record in records:
        baseline = baselines[(record["variant"], record["seed"])]
        record.update(classify_transfer(record, baseline))

    arrays["source__q_max"] = np.stack(
        [source_agent.max_q_grid(0), source_agent.max_q_grid(1)]
    )
    array_file = common.raw_path(ALGORITHM, "transfer_q_maps.npz")
    np.savez_compressed(array_file, **arrays)

    payload = {
        "run": common.run_stamp(config),
        "source_model": paths.rel(source_model),
        "source_map": paths.rel(paths.map_file(config["student_id"])),
        "q_maps_file": paths.rel(array_file),
        "targets": target_meta,
        "scenarios": [scenario for scenario, _ in specs],
        "records": records,
    }
    return payload
