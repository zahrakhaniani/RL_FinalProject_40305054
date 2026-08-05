"""Model-based Value Iteration on the exact stochastic transition model.

The energy component of the state strictly decreases by one on every action, so
the MDP is acyclic in that dimension: the optimal values at energy ``e`` depend
only on the values at energy ``e - 1``. Sweeping the energy levels in ascending
order therefore turns Value Iteration into exact backward induction that
converges in a single pass, and ``bellman_residual()`` verifies it afterwards by
re-deriving the Bellman optimality error straight from ``env.transitions()``.

To keep the sweeps vectorised, the transition model is cached once per
``(cell, has_key, action)``. That is exact because the only energy-dependent
term in the reward is the energy-exhaustion penalty, which is re-applied
explicitly at the last energy level.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.base import BaseAgent
from environments.maze import N_ACTIONS, PERPENDICULAR, Action, MazeEnv, State


class ValueIterationAgent(BaseAgent):
    name = "value_iteration"

    def __init__(
        self,
        env: MazeEnv,
        gamma: float = 0.99,
        theta: float = 1e-10,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.gamma = float(gamma)
        self.theta = float(theta)
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)

        self.cells: List[Tuple[int, int]] = list(env.passable_cells)
        self.cell_index: Dict[Tuple[int, int], int] = {
            cell: i for i, cell in enumerate(self.cells)
        }
        self.n_cells = len(self.cells)
        self.max_energy = env.max_energy
        self.goal_index = self.cell_index[env.goal]

        self.values = np.zeros((self.n_cells, 2, self.max_energy + 1))
        self.policy = np.zeros((self.n_cells, 2, self.max_energy + 1), dtype=np.int8)
        self.trained = False
        self._model = self._build_model()

    # ------------------------------------------------------------------- model

    def _build_model(self) -> Dict[Tuple[int, int], List[Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
        """Cache ``(prob, next_cell, next_has_key, reward, success)`` per action.

        Evaluated at a reference energy high enough that no outcome exhausts the
        battery, so the cached rewards carry no energy-dependent term.
        """
        if self.max_energy < 2:
            raise ValueError("max_energy must be at least 2")
        reference_energy = self.max_energy
        model: Dict[Tuple[int, int], List] = {}

        for has_key in (0, 1):
            for action in range(N_ACTIONS):
                left, right = PERPENDICULAR[Action(action)]
                moves = (
                    (Action(action), self.env.p_intended),
                    (left, self.env.p_slip),
                    (right, self.env.p_slip),
                )
                outcomes = []
                for move, prob in moves:
                    next_cell = np.zeros(self.n_cells, dtype=np.int32)
                    next_key = np.zeros(self.n_cells, dtype=np.int8)
                    reward = np.zeros(self.n_cells, dtype=np.float64)
                    success = np.zeros(self.n_cells, dtype=bool)
                    for index, (r, c) in enumerate(self.cells):
                        state: State = (r, c, has_key, reference_energy)
                        (nr, nc, nk, _), rew, _, info = self.env.apply_move(state, move)
                        next_cell[index] = self.cell_index[(nr, nc)]
                        next_key[index] = nk
                        reward[index] = rew
                        success[index] = info["success"]
                    outcomes.append((prob, next_cell, next_key, reward, success))
                model[(has_key, action)] = outcomes
        return model

    # ------------------------------------------------------------------- train

    def train(self, verbose: bool = False) -> dict:
        """Backward induction over the energy dimension."""
        started = time.perf_counter()
        exhaustion_penalty = (
            self.env.rewards["energy_exhausted"]
            if self.env.reward_mode == "shaped"
            else 0.0
        )
        self.values[:] = 0.0
        self.policy[:] = 0

        deltas: List[float] = []
        for energy in range(1, self.max_energy + 1):
            next_values = self.values[:, :, energy - 1]
            # At energy 1 every non-terminal outcome empties the battery.
            terminal_level = energy == 1
            level_delta = 0.0

            for has_key in (0, 1):
                q = np.empty((self.n_cells, N_ACTIONS))
                for action in range(N_ACTIONS):
                    total = np.zeros(self.n_cells)
                    for prob, next_cell, next_key, reward, success in self._model[
                        (has_key, action)
                    ]:
                        step_reward = reward
                        if terminal_level and exhaustion_penalty:
                            step_reward = reward + np.where(
                                success, 0.0, exhaustion_penalty
                            )
                        bootstrap = np.where(
                            success, 0.0, next_values[next_cell, next_key]
                        )
                        total += prob * (step_reward + self.gamma * bootstrap)
                    q[:, action] = total

                best = q.max(axis=1)
                self.values[:, has_key, energy] = best
                self.policy[:, has_key, energy] = q.argmax(axis=1)
                # How much the optimal value still moves when one more unit of
                # energy is granted; this shrinking gap is the convergence curve
                # towards the infinite-horizon optimum.
                level_delta = max(
                    level_delta, float(np.abs(best - next_values[:, has_key]).max())
                )

            # The goal with the key in hand is terminal: its value stays zero.
            self.values[self.goal_index, 1, energy] = 0.0
            self.policy[self.goal_index, 1, energy] = 0
            deltas.append(level_delta)

        self.trained = True
        elapsed = time.perf_counter() - started
        residual = self.bellman_residual()
        if verbose:
            print(
                f"  value iteration: {self.max_energy} energy sweeps in {elapsed:.2f}s, "
                f"Bellman residual {residual:.2e}"
            )
        return {
            "algorithm": self.name,
            "reward_mode": self.env.reward_mode,
            "gamma": self.gamma,
            "energy_sweeps": self.max_energy,
            "states_evaluated": int(self.n_cells * 2 * self.max_energy),
            "train_seconds": elapsed,
            "bellman_residual": residual,
            "max_delta_per_sweep": deltas,
            "start_state_value": float(
                self.values[self.cell_index[self.env.start], 0, self.max_energy]
            ),
        }

    def bellman_residual(self, sample: int = 6000, seed: int = 7) -> float:
        """Largest Bellman optimality error, computed from ``env.transitions``."""
        rng = np.random.default_rng(seed)
        states: List[State] = []
        for _ in range(sample):
            cell = self.cells[int(rng.integers(self.n_cells))]
            has_key = int(rng.integers(2))
            energy = int(rng.integers(1, self.max_energy + 1))
            states.append((cell[0], cell[1], has_key, energy))

        worst = 0.0
        for state in states:
            if self.env.is_terminal(state):
                continue
            best = -np.inf
            for action in range(N_ACTIONS):
                total = 0.0
                for prob, next_state, reward, done in self.env.transitions(state, action):
                    bootstrap = 0.0 if done else self.value_of(next_state)
                    total += prob * (reward + self.gamma * bootstrap)
                best = max(best, total)
            worst = max(worst, abs(best - self.value_of(state)))
        return float(worst)

    # ------------------------------------------------------------------ policy

    def value_of(self, state: State) -> float:
        r, c, has_key, energy = state
        if energy <= 0:
            return 0.0
        return float(self.values[self.cell_index[(int(r), int(c))], int(has_key), int(energy)])

    def get_action(self, state: State, greedy: bool = True) -> int:
        r, c, has_key, energy = state
        index = self.cell_index.get((int(r), int(c)))
        if index is None or energy <= 0:
            return int(self.rng.integers(N_ACTIONS))
        return int(self.policy[index, int(has_key), int(energy)])

    def value_grid(self, has_key: int, energy: int) -> np.ndarray:
        """Values laid out on the grid, for the GUI heat map."""
        grid = np.full((self.env.rows, self.env.cols), np.nan)
        energy = int(np.clip(energy, 0, self.max_energy))
        for index, cell in enumerate(self.cells):
            grid[cell] = self.values[index, int(has_key), energy]
        return grid

    def greedy_policy_grid(self, has_key: int, energy: int) -> np.ndarray:
        grid = np.zeros((self.env.rows, self.env.cols), dtype=int)
        energy = int(np.clip(energy, 1, self.max_energy))
        for index, cell in enumerate(self.cells):
            grid[cell] = self.policy[index, int(has_key), energy]
        return grid

    # -------------------------------------------------------------- persistence

    def hyperparameters(self) -> dict:
        return {
            "algorithm": self.name,
            "gamma": self.gamma,
            "theta": self.theta,
            "reward_mode": self.env.reward_mode,
            "max_energy": self.max_energy,
            "n_cells": self.n_cells,
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            values=self.values,
            policy=self.policy,
            cells=np.array(self.cells, dtype=np.int16),
            meta=json.dumps(self.hyperparameters()),
        )
        return path

    def load(self, path) -> "ValueIterationAgent":
        with np.load(Path(path), allow_pickle=False) as data:
            values = data["values"]
            if values.shape != self.values.shape:
                raise ValueError(
                    f"saved values have shape {values.shape}, expected {self.values.shape}"
                )
            self.values = values
            self.policy = data["policy"]
        self.trained = True
        return self
