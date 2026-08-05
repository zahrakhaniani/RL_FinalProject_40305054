"""Shared scaffolding for the three agents.

The environment state is ``(row, col, has_key, energy_remaining)``. Value
Iteration works on that state exactly. The two tabular learners would need a
table entry for every single energy level, which is far more resolution than
they can ever fill from experience, so they bucket the energy into
``energy_bins`` groups. The abstraction is the only difference between what the
planner and the learners see, and it is reported with every saved model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from environments.maze import N_ACTIONS, MazeEnv, State


def energy_bin(energy: int, max_energy: int, bins: int) -> int:
    """Map remaining energy to one of ``bins`` buckets (0 = almost empty)."""
    if bins <= 1:
        return 0
    return min(bins - 1, int(energy) * bins // (max_energy + 1))


class BaseAgent:
    name = "agent"

    def get_action(self, state: State, greedy: bool = True) -> int:
        raise NotImplementedError

    def save(self, path) -> Path:
        raise NotImplementedError


class TabularAgent(BaseAgent):
    """Tabular Q-function over ``(row, col, has_key, energy_bin)``."""

    def __init__(
        self,
        env: MazeEnv,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.997,
        energy_bins: int = 8,
        seed: int = 0,
        optimistic_init: float = 0.0,
    ) -> None:
        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon = float(epsilon_start)
        self.energy_bins = int(energy_bins)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

        self.q = np.full(
            (env.rows, env.cols, 2, self.energy_bins, N_ACTIONS),
            float(optimistic_init),
            dtype=np.float64,
        )
        self.training_episodes = 0

    # ------------------------------------------------------------ state access

    def feature(self, state: State) -> Tuple[int, int, int, int]:
        r, c, has_key, energy = state
        return (
            int(r),
            int(c),
            int(has_key),
            energy_bin(energy, self.env.max_energy, self.energy_bins),
        )

    def q_values(self, state: State) -> np.ndarray:
        return self.q[self.feature(state)]

    def flat_index(self, feature_action: Tuple[int, int, int, int, int]) -> int:
        """Index of ``(row, col, has_key, energy_bin, action)`` in a ravelled Q."""
        row, col, has_key, bin_index, action = feature_action
        flat = (row * self.env.cols + col) * 2 + has_key
        return (flat * self.energy_bins + bin_index) * N_ACTIONS + action

    def get_action(self, state: State, greedy: bool = True) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        return self.greedy_action(state)

    def greedy_action(self, state: State) -> int:
        values = self.q_values(state)
        best = np.flatnonzero(values == values.max())
        if best.size == 1:
            return int(best[0])
        return int(best[self.rng.integers(best.size)])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def greedy_policy_grid(self, has_key: int, energy: int) -> np.ndarray:
        """Greedy action per cell, for the GUI policy overlay."""
        bin_index = energy_bin(energy, self.env.max_energy, self.energy_bins)
        return self.q[:, :, has_key, bin_index, :].argmax(axis=-1)

    def value_grid(self, has_key: int, energy: int) -> np.ndarray:
        bin_index = energy_bin(energy, self.env.max_energy, self.energy_bins)
        return self.q[:, :, has_key, bin_index, :].max(axis=-1)

    # -------------------------------------------------------------- persistence

    def hyperparameters(self) -> Dict[str, float]:
        return {
            "algorithm": self.name,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "energy_bins": self.energy_bins,
            "seed": self.seed,
            "reward_mode": self.env.reward_mode,
            "training_episodes": self.training_episodes,
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            q=self.q,
            meta=json.dumps(self.hyperparameters()),
        )
        return path

    def load(self, path) -> "TabularAgent":
        with np.load(Path(path), allow_pickle=False) as data:
            q = data["q"]
            if q.shape != self.q.shape:
                raise ValueError(
                    f"saved Q table has shape {q.shape}, expected {self.q.shape}"
                )
            self.q = q.astype(np.float64)
            meta = json.loads(str(data["meta"]))
        self.training_episodes = int(meta.get("training_episodes", 0))
        return self


class TrainingLog:
    """Per-episode training statistics plus periodic greedy evaluations."""

    def __init__(self) -> None:
        self.episode: List[int] = []
        self.reward: List[float] = []
        self.steps: List[int] = []
        self.success: List[int] = []
        self.energy_left: List[int] = []
        self.epsilon: List[float] = []
        self.outcome: List[str] = []
        self.eval_episode: List[int] = []
        self.eval_success_rate: List[float] = []
        self.eval_return: List[float] = []
        self.eval_steps: List[float] = []

    def add_episode(
        self,
        episode: int,
        reward: float,
        steps: int,
        success: bool,
        energy_left: int,
        epsilon: float,
        outcome: str,
    ) -> None:
        self.episode.append(int(episode))
        self.reward.append(float(reward))
        self.steps.append(int(steps))
        self.success.append(int(bool(success)))
        self.energy_left.append(int(energy_left))
        self.epsilon.append(float(epsilon))
        self.outcome.append(str(outcome))

    def add_eval(self, episode: int, metrics: Dict[str, float]) -> None:
        self.eval_episode.append(int(episode))
        self.eval_success_rate.append(float(metrics["success_rate"]))
        self.eval_return.append(float(metrics["mean_return"]))
        self.eval_steps.append(float(metrics["mean_steps"]))

    def to_dict(self) -> dict:
        return {
            "episode": self.episode,
            "reward": self.reward,
            "steps": self.steps,
            "success": self.success,
            "energy_left": self.energy_left,
            "epsilon": self.epsilon,
            "outcome": self.outcome,
            "eval_episode": self.eval_episode,
            "eval_success_rate": self.eval_success_rate,
            "eval_return": self.eval_return,
            "eval_steps": self.eval_steps,
        }


def evaluate_policy(
    env: MazeEnv,
    agent: BaseAgent,
    episodes: int = 200,
    seed: int = 999_000,
) -> Dict[str, float]:
    """Run ``episodes`` greedy episodes on a private copy of ``env``."""
    eval_env = env.copy()
    returns = np.zeros(episodes)
    steps = np.zeros(episodes)
    successes = np.zeros(episodes)
    energy_left = np.zeros(episodes)
    keys = np.zeros(episodes)
    outcomes: Dict[str, int] = {}

    for episode in range(episodes):
        state = eval_env.reset(seed=seed + episode)
        total = 0.0
        done = False
        while not done:
            action = agent.get_action(state, greedy=True)
            state, reward, done, info = eval_env.step(action)
            total += reward
        returns[episode] = total
        steps[episode] = eval_env.steps
        successes[episode] = 1.0 if eval_env.outcome == "success" else 0.0
        energy_left[episode] = eval_env.energy
        keys[episode] = float(eval_env.has_key)
        outcomes[eval_env.outcome] = outcomes.get(eval_env.outcome, 0) + 1

    solved = successes.astype(bool)
    return {
        "episodes": int(episodes),
        "success_rate": float(successes.mean()),
        "key_rate": float(keys.mean()),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "mean_steps": float(steps.mean()),
        "mean_steps_when_solved": float(steps[solved].mean()) if solved.any() else float("nan"),
        "mean_energy_left": float(energy_left.mean()),
        "outcomes": outcomes,
    }


def rollout(
    env: MazeEnv,
    agent: BaseAgent,
    seed: int = 0,
    greedy: bool = True,
) -> dict:
    """Single episode trace, used by the GUI and by the report figures."""
    trace_env = env.copy()
    state = trace_env.reset(seed=seed)
    states: List[State] = [state]
    actions: List[int] = []
    rewards: List[float] = []
    flags: List[dict] = []
    done = False
    while not done:
        action = agent.get_action(state, greedy=greedy)
        state, reward, done, info = trace_env.step(action)
        states.append(state)
        actions.append(int(action))
        rewards.append(float(reward))
        flags.append(info)
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "flags": flags,
        "total_reward": float(sum(rewards)),
        "steps": trace_env.steps,
        "outcome": trace_env.outcome,
    }
