"""Shared plumbing for the experiment scripts.

Everything that the per-algorithm runners have in common lives here: loading the
config, materialising the one shared maze, building agents, running a training
job and writing results into the per-algorithm folders under ``results/``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import paths
from agents.base import TabularAgent, evaluate_policy
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from agents.value_iteration import ValueIterationAgent
from environments.generator import MazeGenerator
from environments.maze import MazeEnv

LEARNERS = {"q_learning": QLearningAgent, "sarsa_lambda": SarsaLambdaAgent}


# --------------------------------------------------------------------- config


def load_config(path: Optional[Path] = None) -> dict:
    path = Path(path) if path else paths.DEFAULT_CONFIG
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_file"] = paths.rel(path)
    return config


def config_hash(config: dict) -> str:
    payload = {k: v for k, v in config.items() if not k.startswith("_")}
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def run_stamp(config: dict) -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_file": config.get("_config_file"),
        "config_hash": config_hash(config),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


# ----------------------------------------------------------------------- maze


def make_generator(config: dict) -> MazeGenerator:
    maze = config["maze"]
    return MazeGenerator(
        student_id=config["student_id"],
        seed=maze["seed"],
        size=maze.get("size"),
        min_wall_fraction=maze["min_wall_fraction"],
        n_penalty_cells=maze["n_penalty_cells"],
        loop_openings=maze["loop_openings"],
        vault_span=maze["vault_span"],
        energy_slack=maze["energy_slack"],
    )


def ensure_map(config: dict, force: bool = False, verbose: bool = True) -> Path:
    """Generate and save the shared maze once; every run reloads that file."""
    target = paths.map_file(config["student_id"])
    if target.exists() and not force:
        return target

    generator = make_generator(config)
    episode = config.get("episode", {})
    env = generator.generate(
        reward_mode="shaped",
        rewards=config["rewards"],
        max_energy=episode.get("max_energy"),
        max_steps=episode.get("max_steps"),
    )
    env.save_map(target)
    if verbose:
        print(generator.info())
        print(env.summary())
        env.render()
        print(f"saved shared map to {paths.rel(target)}")
    return target


def build_env(
    config: dict,
    reward_mode: str = "shaped",
    seed: int = 0,
    map_path: Optional[Path] = None,
) -> MazeEnv:
    """Load the shared maze so every algorithm sees the same environment."""
    map_path = Path(map_path) if map_path else ensure_map(config, verbose=False)
    episode = config.get("episode", {})
    return MazeEnv.load_map(
        map_path,
        reward_mode=reward_mode,
        rewards=config["rewards"],
        seed=seed,
        max_energy=episode.get("max_energy"),
        max_steps=episode.get("max_steps"),
    )


# --------------------------------------------------------------------- agents


#: Config keys that describe the experiment rather than the agent constructor.
NON_AGENT_KEYS = ("episodes", "gamma_sweep")


def make_learner(
    algorithm: str,
    env: MazeEnv,
    config: dict,
    seed: int,
    overrides: Optional[dict] = None,
) -> TabularAgent:
    if algorithm not in LEARNERS:
        raise KeyError(f"unknown learner {algorithm!r}; choose from {sorted(LEARNERS)}")
    settings = {
        key: value
        for key, value in config[algorithm].items()
        if key not in NON_AGENT_KEYS
    }
    if overrides:
        settings.update(overrides)
    return LEARNERS[algorithm](env, seed=seed, **settings)


def make_planner(
    env: MazeEnv, config: dict, gamma: Optional[float] = None
) -> ValueIterationAgent:
    settings = config["value_iteration"]
    return ValueIterationAgent(
        env,
        gamma=settings["gamma"] if gamma is None else float(gamma),
        theta=settings.get("theta", 1e-10),
    )


def load_trained_learner(
    config: dict,
    algorithm: str,
    reward_mode: str,
    seed: int,
    label: Optional[str] = None,
) -> Tuple[Optional[TabularAgent], MazeEnv]:
    """Rebuild a learner and restore its saved Q-table, if the model exists."""
    env = build_env(config, reward_mode=reward_mode, seed=seed)
    agent = make_learner(algorithm, env, config, seed)
    filename = f"{algorithm}_{label or f'{reward_mode}_seed{seed}'}.npz"
    model_file = model_path(algorithm, filename)
    if not model_file.exists():
        return None, env
    agent.load(model_file)
    return agent, env


def load_trained_planner(
    config: dict, reward_mode: str
) -> Tuple[Optional[ValueIterationAgent], MazeEnv]:
    """Restore the saved optimal value table without rebuilding the model."""
    env = build_env(config, reward_mode=reward_mode, seed=0)
    agent = ValueIterationAgent(
        env,
        gamma=config["value_iteration"]["gamma"],
        theta=config["value_iteration"].get("theta", 1e-10),
        build_model=False,
    )
    model_file = model_path("value_iteration", f"value_iteration_{reward_mode}.npz")
    if not model_file.exists():
        return None, env
    agent.load(model_file)
    return agent, env


# ------------------------------------------------------------------- training


def train_learner(
    algorithm: str,
    config: dict,
    reward_mode: str,
    seed: int,
    overrides: Optional[dict] = None,
    episodes: Optional[int] = None,
    tag: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Train one learner on one reward mode with one seed, and save the model."""
    training = config["training"]
    env = build_env(config, reward_mode=reward_mode, seed=seed)
    agent = make_learner(algorithm, env, config, seed, overrides)
    episodes = int(episodes if episodes is not None else config[algorithm]["episodes"])

    label = tag or f"{reward_mode}_seed{seed}"
    if verbose:
        print(f"  [{algorithm}] {label}: {episodes} episodes")

    started = time.perf_counter()
    log = agent.train(
        episodes=episodes,
        eval_every=training["eval_every"],
        eval_episodes=training["eval_episodes"],
        eval_seed=training["eval_seed"],
        verbose=False,
    )
    train_seconds = time.perf_counter() - started

    final_eval = evaluate_policy(
        env, agent, episodes=training["final_eval_episodes"], seed=training["eval_seed"]
    )
    model_file = model_path(algorithm, f"{algorithm}_{label}.npz")
    agent.save(model_file)

    episode_csv = write_episode_csv(log, algorithm, label, agent, config)
    update_csv = write_q_update_csv(agent, algorithm, label)

    if verbose:
        print(
            f"    -> success {final_eval['success_rate']:.3f} | "
            f"return {final_eval['mean_return']:8.2f} | "
            f"steps {final_eval['mean_steps']:6.1f} | {train_seconds:.1f}s"
        )

    log_dict = log.to_dict()
    record = {
        "algorithm": algorithm,
        "reward_mode": reward_mode,
        "seed": seed,
        "episodes": episodes,
        "label": label,
        "hyperparameters": agent.hyperparameters(),
        "train_seconds": train_seconds,
        "memory_kilobytes": round(agent.memory_bytes / 1024, 1),
        "final_eval": final_eval,
        "stability": stability_metrics(log_dict),
        "episodes_to_threshold": episodes_to_threshold(log_dict, threshold=0.8),
        "log": log_dict,
        "visit_counts": agent.visit_grid().tolist(),
        "model_file": paths.rel(model_file),
        "episode_csv": paths.rel(episode_csv),
    }
    if update_csv is not None:
        record["q_update_csv"] = paths.rel(update_csv)
    return record


def stability_metrics(log_dict: dict) -> dict:
    """Spread of late-training performance -- how settled the policy is."""
    evaluations = [value for value in log_dict.get("eval_success_rate", []) if value is not None]
    returns = [value for value in log_dict.get("reward", []) if value is not None]
    late_evals = evaluations[-5:]
    late_returns = returns[-200:]
    return {
        "late_eval_success_mean": float(np.mean(late_evals)) if late_evals else float("nan"),
        "late_eval_success_std": float(np.std(late_evals)) if late_evals else float("nan"),
        "late_return_mean": float(np.mean(late_returns)) if late_returns else float("nan"),
        "late_return_std": float(np.std(late_returns)) if late_returns else float("nan"),
    }


def write_episode_csv(
    log,
    algorithm: str,
    label: str,
    agent,
    config: dict,
) -> Path:
    """Per-episode CSV log, prefixed with the algorithm/config provenance."""
    frame = pd.DataFrame(log.rows)
    hyper = agent.hyperparameters()
    frame.insert(0, "algorithm", algorithm)
    frame.insert(1, "run_label", label)
    frame.insert(2, "reward_mode", hyper.get("reward_mode"))
    frame.insert(3, "seed", hyper.get("seed"))
    frame.insert(4, "config_hash", config_hash(config))
    for key in ("gamma", "energy_bins", "epsilon_schedule", "lam", "replacing_traces"):
        if key in hyper:
            frame[f"cfg_{key}"] = hyper[key]

    destination = raw_path(algorithm, f"episodes_{label}.csv")
    frame.to_csv(destination, index=False)
    return destination


def write_q_update_csv(agent, algorithm: str, label: str) -> Optional[Path]:
    """Sampled individual Q-updates, when the agent recorded any."""
    samples = getattr(agent, "q_update_samples", None)
    if not samples:
        return None
    destination = raw_path(algorithm, f"q_updates_{label}.csv")
    pd.DataFrame(samples).to_csv(destination, index=False)
    return destination


# ------------------------------------------------------------------- results


def raw_path(algorithm: str, filename: str) -> Path:
    return paths.subdir(paths.RAW_DATA, algorithm) / filename


def model_path(algorithm: str, filename: str) -> Path:
    return paths.subdir(paths.MODELS, algorithm) / filename


def figure_path(algorithm: str, filename: str) -> Path:
    return paths.subdir(paths.FIGURES, algorithm) / filename


def save_json(data: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=_json_default)
    return path


def load_json(path: Path) -> dict:
    with open(Path(path), encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return paths.rel(value)
    raise TypeError(f"cannot serialise {type(value)!r}")


def summary_frame(records: Sequence[dict]) -> pd.DataFrame:
    """One row per training run, with the headline evaluation metrics."""
    rows = []
    for record in records:
        evaluation = record["final_eval"]
        stability = record.get("stability", {})
        rows.append(
            {
                "algorithm": record["algorithm"],
                "reward_mode": record["reward_mode"],
                "seed": record.get("seed"),
                "label": record.get("label"),
                "episodes": record.get("episodes", 0),
                "success_rate": evaluation["success_rate"],
                "key_rate": evaluation["key_rate"],
                "mean_return": evaluation["mean_return"],
                "std_return": evaluation["std_return"],
                "mean_steps": evaluation["mean_steps"],
                "mean_steps_when_solved": evaluation["mean_steps_when_solved"],
                "mean_energy_left": evaluation["mean_energy_left"],
                "mean_wall_collisions": evaluation.get("mean_wall_collisions", float("nan")),
                "mean_penalty_entries": evaluation.get("mean_penalty_entries", float("nan")),
                "episodes_to_threshold": record.get("episodes_to_threshold"),
                "late_eval_success_std": stability.get("late_eval_success_std", float("nan")),
                "late_return_std": stability.get("late_return_std", float("nan")),
                "memory_kilobytes": record.get("memory_kilobytes", float("nan")),
                "train_seconds": record.get("train_seconds", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def write_summary(records: Sequence[dict], algorithm: str) -> pd.DataFrame:
    frame = summary_frame(records)
    frame.to_csv(raw_path(algorithm, f"{algorithm}_summary.csv"), index=False)
    return frame


def episodes_to_threshold(log: dict, threshold: float = 0.8) -> Optional[int]:
    """First evaluation checkpoint whose greedy success rate reaches ``threshold``."""
    for episode, rate in zip(log.get("eval_episode", []), log.get("eval_success_rate", [])):
        if rate is not None and rate >= threshold:
            return int(episode)
    return None


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
