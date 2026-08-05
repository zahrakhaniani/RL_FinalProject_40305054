"""Shared scaffolding for the three agents.

The environment state is ``(row, col, has_key, energy_remaining)``. Value
Iteration works on that state exactly. The two tabular learners would need a
table entry for every single energy level, which is far more resolution than
they can ever fill from experience, so they bucket the energy into
``energy_bins`` groups. That abstraction is the only difference between what the
planner and the learners see, and it is recorded with every saved model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from environments.maze import EVENT_KEYS, N_ACTIONS, MazeEnv, State

EPSILON_SCHEDULES = ("exponential", "linear")

#: Columns written to the per-episode CSV logs.
EPISODE_LOG_COLUMNS = (
    "episode",
    "steps",
    "reward",
    "mean_return_100",
    "epsilon",
    "alpha",
    "energy_left",
    "outcome",
    "success",
    "goal_success",
    "energy_exhausted",
    "max_steps_reached",
    "wall_collisions",
    "penalty_entries",
    "door_blocked",
    "door_crossings",
    "key_pickups",
    "slips",
    "mean_abs_td_error",
    "max_abs_td_error",
    "mean_active_traces",
    "max_active_traces",
)


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

    @property
    def memory_bytes(self) -> int:
        return 0


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
        epsilon_schedule: str = "exponential",
        epsilon_decay_episodes: Optional[int] = None,
        energy_bins: int = 8,
        seed: int = 0,
        optimistic_init: float = 0.0,
    ) -> None:
        if epsilon_schedule not in EPSILON_SCHEDULES:
            raise ValueError(
                f"epsilon_schedule must be one of {EPSILON_SCHEDULES}, got {epsilon_schedule!r}"
            )
        self.env = env
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_schedule = epsilon_schedule
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = float(epsilon_start)
        self.energy_bins = int(energy_bins)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

        self.q = np.full(
            (env.rows, env.cols, 2, self.energy_bins, N_ACTIONS),
            float(optimistic_init),
            dtype=np.float64,
        )
        self.visit_counts = np.zeros((env.rows, env.cols, 2), dtype=np.int64)
        self.training_episodes = 0
        self.train_seconds = 0.0

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

    def record_visit(self, state: State) -> None:
        self.visit_counts[int(state[0]), int(state[1]), int(state[2])] += 1

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

    def update_epsilon(self, episode: int, total_episodes: int) -> None:
        """Two schedules, so the experiment can compare them directly."""
        if self.epsilon_schedule == "linear":
            span = self.epsilon_decay_episodes or max(1, int(0.7 * total_episodes))
            fraction = min(1.0, episode / float(span))
            self.epsilon = self.epsilon_start + (
                self.epsilon_end - self.epsilon_start
            ) * fraction
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # -------------------------------------------------------------- inspection

    def greedy_policy_grid(self, has_key: int, energy: int) -> np.ndarray:
        """Greedy action per cell, for the policy overlay and difference maps."""
        bin_index = energy_bin(energy, self.env.max_energy, self.energy_bins)
        return self.q[:, :, has_key, bin_index, :].argmax(axis=-1)

    def value_grid(self, has_key: int, energy: int) -> np.ndarray:
        bin_index = energy_bin(energy, self.env.max_energy, self.energy_bins)
        return self.q[:, :, has_key, bin_index, :].max(axis=-1)

    def max_q_grid(self, has_key: int) -> np.ndarray:
        """Best Q-value per cell, maximised over energy bins and actions."""
        return self.q[:, :, has_key, :, :].max(axis=(-1, -2))

    def visit_grid(self) -> np.ndarray:
        return self.visit_counts.sum(axis=2)

    @property
    def memory_bytes(self) -> int:
        return int(self.q.nbytes + self.visit_counts.nbytes)

    # -------------------------------------------------------------- persistence

    def hyperparameters(self) -> Dict[str, object]:
        return {
            "algorithm": self.name,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_schedule": self.epsilon_schedule,
            "epsilon_decay_episodes": self.epsilon_decay_episodes,
            "energy_bins": self.energy_bins,
            "seed": self.seed,
            "reward_mode": self.env.reward_mode,
            "training_episodes": self.training_episodes,
            "q_table_shape": list(self.q.shape),
            "q_table_kilobytes": round(self.q.nbytes / 1024, 1),
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            q=self.q,
            visit_counts=self.visit_counts,
            policy=self.q.argmax(axis=-1).astype(np.int8),
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
            if "visit_counts" in data:
                self.visit_counts = data["visit_counts"]
            meta = json.loads(str(data["meta"]))
        self.training_episodes = int(meta.get("training_episodes", 0))
        return self


class TrainingLog:
    """Per-episode statistics plus periodic greedy evaluations."""

    def __init__(self) -> None:
        self.rows: List[dict] = []
        self.evals: List[dict] = []

    def add_episode(self, **fields) -> None:
        recent = [row["reward"] for row in self.rows[-99:]] + [fields["reward"]]
        fields["mean_return_100"] = float(np.mean(recent))
        self.rows.append({column: fields.get(column) for column in EPISODE_LOG_COLUMNS})

    def add_eval(self, episode: int, metrics: Dict[str, float]) -> None:
        self.evals.append(
            {
                "episode": int(episode),
                "success_rate": float(metrics["success_rate"]),
                "mean_return": float(metrics["mean_return"]),
                "mean_steps": float(metrics["mean_steps"]),
            }
        )

    def column(self, name: str) -> List:
        return [row.get(name) for row in self.rows]

    def to_dict(self) -> dict:
        """Just the arrays the figures need.

        Everything else per episode already lives in the CSV log, so it is not
        duplicated into the JSON results.
        """
        return {
            "reward": self.column("reward"),
            "success": self.column("success"),
            "epsilon": self.column("epsilon"),
            "eval_episode": [row["episode"] for row in self.evals],
            "eval_success_rate": [row["success_rate"] for row in self.evals],
            "eval_return": [row["mean_return"] for row in self.evals],
            "eval_steps": [row["mean_steps"] for row in self.evals],
        }


def log_episode(
    log: TrainingLog,
    env: MazeEnv,
    episode: int,
    epsilon: float,
    alpha: float,
    td_errors: Optional[Sequence[float]] = None,
    trace_counts: Optional[Sequence[int]] = None,
) -> None:
    """Copy the environment's event counters into the training log."""
    summary = env.episode_summary()
    absolute = np.abs(np.asarray(td_errors, dtype=float)) if td_errors else None
    counts = np.asarray(trace_counts, dtype=float) if trace_counts else None

    log.add_episode(
        episode=episode,
        steps=summary["steps"],
        reward=summary["episode_reward"],
        epsilon=epsilon,
        alpha=alpha,
        energy_left=summary["energy_left"],
        outcome=summary["outcome"],
        success=summary["goal_success"],
        goal_success=summary["goal_success"],
        energy_exhausted=summary["energy_exhausted"],
        max_steps_reached=summary["max_steps_reached"],
        wall_collisions=summary["wall_collisions"],
        penalty_entries=summary["penalty_entries"],
        door_blocked=summary["door_blocked"],
        door_crossings=summary["door_crossings"],
        key_pickups=summary["key_pickups"],
        slips=summary["slips"],
        mean_abs_td_error=float(absolute.mean()) if absolute is not None and absolute.size else 0.0,
        max_abs_td_error=float(absolute.max()) if absolute is not None and absolute.size else 0.0,
        mean_active_traces=float(counts.mean()) if counts is not None and counts.size else 0.0,
        max_active_traces=float(counts.max()) if counts is not None and counts.size else 0.0,
    )


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
    collisions = np.zeros(episodes)
    penalties = np.zeros(episodes)
    outcomes: Dict[str, int] = {}

    for episode in range(episodes):
        state = eval_env.reset(seed=seed + episode)
        done = False
        while not done:
            action = agent.get_action(state, greedy=True)
            state, reward, done, info = eval_env.step(action)
        returns[episode] = eval_env.episode_reward
        steps[episode] = eval_env.steps
        successes[episode] = 1.0 if eval_env.outcome == "success" else 0.0
        energy_left[episode] = eval_env.energy
        keys[episode] = float(eval_env.has_key)
        collisions[episode] = eval_env.events["wall_collisions"]
        penalties[episode] = eval_env.events["penalty_entries"]
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
        "mean_wall_collisions": float(collisions.mean()),
        "mean_penalty_entries": float(penalties.mean()),
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
        "events": dict(trace_env.events),
    }


def policy_agreement(
    reference,
    learner: TabularAgent,
    env: MazeEnv,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """How often the learner's greedy action equals Value Iteration's.

    Compared at the midpoint energy of every energy bin, since that is the
    finest resolution the learner actually represents. ``weights`` (typically
    visit counts) gives the visit-weighted agreement, which says how often the
    two agree *where the agent actually goes* rather than across states it never
    reaches.
    """
    bins = learner.energy_bins
    matches: List[float] = []
    weighted_hits = 0.0
    weighted_total = 0.0
    per_cell = np.full((2, env.rows, env.cols), np.nan)

    for bin_index in range(bins):
        low = bin_index * (env.max_energy + 1) // bins
        high = (bin_index + 1) * (env.max_energy + 1) // bins - 1
        energy = max(1, (low + high) // 2)

        for has_key in (0, 1):
            reference_grid = reference.greedy_policy_grid(has_key=has_key, energy=energy)
            learner_grid = learner.greedy_policy_grid(has_key=has_key, energy=energy)
            for row, col in env.passable_cells:
                if (row, col) == env.goal and has_key == 1:
                    continue
                agrees = float(reference_grid[row, col] == learner_grid[row, col])
                matches.append(agrees)
                if weights is not None:
                    weight = float(weights[row, col, has_key])
                    weighted_hits += agrees * weight
                    weighted_total += weight
                if bin_index == bins - 1:
                    per_cell[has_key, row, col] = agrees

    return {
        "agreement": float(np.mean(matches)) if matches else float("nan"),
        "weighted_agreement": (
            weighted_hits / weighted_total if weighted_total > 0 else float("nan")
        ),
        "states_compared": len(matches),
        "per_cell_full_battery": per_cell,
    }
