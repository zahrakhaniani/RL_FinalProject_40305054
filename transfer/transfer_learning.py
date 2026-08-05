"""Transfer learning between two mazes.

A second maze is generated from a different layout seed (same size, same
dynamics, same reward function) and a Q-Learning agent has to solve it. Three
strategies are compared:

``scratch``
    the control: learn the target maze from a zero-initialised table.
``warm_start``
    initialise the table with the Q-values learned on the source maze, scaled by
    ``transfer_weight``, then keep learning normally.
``policy_reuse``
    start from a zero table but, while exploring, follow the source policy with
    probability ``psi``, which decays each episode. The source acts as a
    behaviour prior instead of a value prior.

Because the tabular state is ``(row, col, has_key, energy_bin)`` and both mazes
share the grid size and the number of energy bins, the tables are shape
compatible. The energy bins are relative to each maze's own ``max_energy``, so a
bin means "roughly this fraction of the battery left" in both tasks, which is
what makes the transfer meaningful even though the optimal path lengths differ.

Reported metrics follow the usual transfer-learning vocabulary: jumpstart (how
good the very first evaluation is), episodes-to-threshold (how fast it becomes
reliable) and asymptotic performance (where it ends up).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import paths
from agents.base import evaluate_policy
from agents.q_learning import QLearningAgent
from environments.generator import MazeGenerator
from environments.maze import MazeEnv
from experiments import common

ALGORITHM = "transfer"


class PolicyReuseAgent(QLearningAgent):
    """Q-Learning that borrows the source policy while exploring."""

    name = "policy_reuse"

    def __init__(
        self,
        *args,
        source_policy: Optional[QLearningAgent] = None,
        psi_start: float = 0.8,
        psi_decay: float = 0.995,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.source_policy = source_policy
        self.psi_start = float(psi_start)
        self.psi = float(psi_start)
        self.psi_decay = float(psi_decay)

    def get_action(self, state, greedy: bool = True) -> int:
        if (
            not greedy
            and self.source_policy is not None
            and self.rng.random() < self.psi
        ):
            return self.source_policy.greedy_action(state)
        return super().get_action(state, greedy=greedy)

    def decay_epsilon(self) -> None:
        super().decay_epsilon()
        self.psi *= self.psi_decay

    def hyperparameters(self) -> Dict[str, float]:
        params = super().hyperparameters()
        params.update({"psi_start": self.psi_start, "psi_decay": self.psi_decay})
        return params


def learner_settings(config: dict) -> dict:
    return {
        key: value
        for key, value in config["q_learning"].items()
        if key != "episodes"
    }


def ensure_target_map(config: dict, verbose: bool = False) -> Path:
    """Generate the transfer target maze once and cache it next to the source."""
    settings = config["transfer"]
    maze = config["maze"]
    target = paths.MAPS / f"maze_transfer_target_{settings['target_seed']}.json"
    if target.exists():
        return target

    generator = MazeGenerator(
        student_id=config["student_id"],
        seed=settings["target_seed"],
        size=maze.get("size"),
        min_wall_fraction=maze["min_wall_fraction"],
        n_penalty_cells=maze["n_penalty_cells"],
        loop_openings=maze["loop_openings"],
        vault_span=maze["vault_span"],
        energy_slack=maze["energy_slack"],
    )
    env = generator.generate(reward_mode="shaped", rewards=config["rewards"])
    env.save_map(target)
    if verbose:
        print("transfer target maze:")
        print(env.summary())
        env.render()
    return target


def build_target_env(config: dict, reward_mode: str, seed: int = 0) -> MazeEnv:
    return common.build_env(
        config,
        reward_mode=reward_mode,
        seed=seed,
        map_path=ensure_target_map(config),
    )


def train_source(
    config: dict, reward_mode: str, seed: int, episodes: int, verbose: bool = True
) -> QLearningAgent:
    """Q-Learning on the original maze; its table is what gets transferred."""
    env = common.build_env(config, reward_mode=reward_mode, seed=seed)
    agent = QLearningAgent(env, seed=seed, **learner_settings(config))
    agent.train(episodes=episodes, eval_every=0)
    if verbose:
        metrics = evaluate_policy(env, agent, episodes=100, seed=config["training"]["eval_seed"])
        print(
            f"  source maze policy: success {metrics['success_rate']:.3f} | "
            f"return {metrics['mean_return']:.2f}"
        )
    agent.save(common.model_path(ALGORITHM, f"source_q_learning_{reward_mode}_seed{seed}.npz"))
    return agent


def _metrics_from_log(log: dict, threshold: float) -> dict:
    success_curve = log["eval_success_rate"]
    return {
        "jumpstart_success": float(success_curve[0]) if success_curve else float("nan"),
        "jumpstart_return": float(log["eval_return"][0]) if success_curve else float("nan"),
        "episodes_to_threshold": common.episodes_to_threshold(
            log["eval_episode"], success_curve, threshold
        ),
        "mean_eval_success": float(np.mean(success_curve)) if success_curve else float("nan"),
        "final_eval_success": float(success_curve[-1]) if success_curve else float("nan"),
    }


def run_strategy(
    config: dict,
    strategy: str,
    reward_mode: str,
    seed: int,
    source_q: Optional[np.ndarray],
    episodes: int,
    verbose: bool = True,
) -> dict:
    settings = config["transfer"]
    training = config["training"]
    env = build_target_env(config, reward_mode=reward_mode, seed=seed)

    if strategy == "policy_reuse":
        if source_q is None:
            raise ValueError("policy_reuse needs a source table")
        # Bound to the *target* env so the energy bins use the target battery.
        source_policy = QLearningAgent(env, seed=seed, **learner_settings(config))
        source_policy.q = source_q.copy()
        agent = PolicyReuseAgent(
            env,
            seed=seed,
            source_policy=source_policy,
            psi_start=settings["psi_start"],
            psi_decay=settings["psi_decay"],
            **learner_settings(config),
        )
    else:
        agent = QLearningAgent(env, seed=seed, **learner_settings(config))
        if strategy == "warm_start":
            if source_q is None:
                raise ValueError("warm_start needs a source table")
            agent.q = source_q.copy() * float(settings["transfer_weight"])
        elif strategy != "scratch":
            raise ValueError(f"unknown transfer strategy {strategy!r}")

    # Zero-shot transfer: how the policy performs on the target maze before it
    # has seen a single target episode. This is the honest jumpstart measurement.
    zero_shot = evaluate_policy(
        env, agent, episodes=training["eval_episodes"], seed=training["eval_seed"]
    )

    started = time.perf_counter()
    log = agent.train(
        episodes=episodes,
        eval_every=training["eval_every"],
        eval_episodes=training["eval_episodes"],
        eval_seed=training["eval_seed"],
    )
    train_seconds = time.perf_counter() - started

    final_eval = evaluate_policy(
        env, agent, episodes=training["final_eval_episodes"], seed=training["eval_seed"]
    )
    model_file = common.model_path(
        ALGORITHM, f"{strategy}_{reward_mode}_seed{seed}.npz"
    )
    agent.save(model_file)

    log_dict = log.to_dict()
    transfer_metrics = _metrics_from_log(log_dict, settings["success_threshold"])
    transfer_metrics["zero_shot_success"] = zero_shot["success_rate"]
    transfer_metrics["zero_shot_return"] = zero_shot["mean_return"]
    transfer_metrics["zero_shot_key_rate"] = zero_shot["key_rate"]

    record = {
        "algorithm": ALGORITHM,
        "strategy": strategy,
        "reward_mode": reward_mode,
        "seed": seed,
        "episodes": episodes,
        "label": f"{strategy}_{reward_mode}_seed{seed}",
        "hyperparameters": agent.hyperparameters(),
        "train_seconds": train_seconds,
        "final_eval": final_eval,
        "zero_shot_eval": zero_shot,
        "transfer_metrics": transfer_metrics,
        "log": log_dict,
        "model_file": paths.rel(model_file),
    }
    if verbose:
        print(
            f"    {strategy:<12} seed {seed} -> zero-shot {transfer_metrics['zero_shot_success']:.2f} "
            f"(key {transfer_metrics['zero_shot_key_rate']:.2f}) | "
            f"to-threshold {transfer_metrics['episodes_to_threshold']} | "
            f"final success {final_eval['success_rate']:.3f} | "
            f"return {final_eval['mean_return']:8.2f}"
        )
    return record


def run_transfer_study(
    config: dict,
    reward_mode: str = "shaped",
    seeds: Optional[Sequence[int]] = None,
    episodes: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    settings = config["transfer"]
    seeds = list(seeds or settings["seeds"])
    episodes = int(episodes if episodes is not None else settings["episodes"])
    ensure_target_map(config, verbose=verbose)

    records: List[dict] = []
    for seed in seeds:
        if verbose:
            print(f"  seed {seed}: training the source policy on the original maze")
        source_agent = train_source(
            config, reward_mode=reward_mode, seed=seed, episodes=episodes, verbose=verbose
        )
        source_q = source_agent.q
        for strategy in settings["strategies"]:
            records.append(
                run_strategy(
                    config,
                    strategy=strategy,
                    reward_mode=reward_mode,
                    seed=seed,
                    source_q=source_q,
                    episodes=episodes,
                    verbose=verbose,
                )
            )

    payload = {
        "run": common.run_stamp(config),
        "reward_mode": reward_mode,
        "target_seed": settings["target_seed"],
        "records": records,
    }
    return payload
