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
from typing import Dict, List, Optional, Sequence

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


def make_learner(
    algorithm: str,
    env: MazeEnv,
    config: dict,
    seed: int,
    overrides: Optional[dict] = None,
) -> TabularAgent:
    if algorithm not in LEARNERS:
        raise KeyError(f"unknown learner {algorithm!r}; choose from {sorted(LEARNERS)}")
    settings = dict(config[algorithm])
    settings.pop("episodes", None)
    if overrides:
        settings.update(overrides)
    return LEARNERS[algorithm](env, seed=seed, **settings)


def make_planner(env: MazeEnv, config: dict) -> ValueIterationAgent:
    settings = config["value_iteration"]
    return ValueIterationAgent(
        env, gamma=settings["gamma"], theta=settings.get("theta", 1e-10)
    )


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

    if verbose:
        print(
            f"    -> success {final_eval['success_rate']:.3f} | "
            f"return {final_eval['mean_return']:8.2f} | "
            f"steps {final_eval['mean_steps']:6.1f} | {train_seconds:.1f}s"
        )

    return {
        "algorithm": algorithm,
        "reward_mode": reward_mode,
        "seed": seed,
        "episodes": episodes,
        "label": label,
        "hyperparameters": agent.hyperparameters(),
        "train_seconds": train_seconds,
        "final_eval": final_eval,
        "log": log.to_dict(),
        "model_file": paths.rel(model_file),
    }


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
                "train_seconds": record.get("train_seconds", float("nan")),
            }
        )
    return pd.DataFrame(rows)


def write_summary(records: Sequence[dict], algorithm: str) -> pd.DataFrame:
    frame = summary_frame(records)
    frame.to_csv(raw_path(algorithm, f"{algorithm}_summary.csv"), index=False)
    return frame


def episodes_to_threshold(
    eval_episode: Sequence[int],
    eval_success_rate: Sequence[float],
    threshold: float = 0.8,
) -> Optional[int]:
    """First evaluation checkpoint whose greedy success rate reaches ``threshold``."""
    for episode, rate in zip(eval_episode, eval_success_rate):
        if rate >= threshold:
            return int(episode)
    return None


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
