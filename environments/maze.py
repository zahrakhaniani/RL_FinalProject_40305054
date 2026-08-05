"""Stochastic maze environment with a key, a locked door and limited energy.

State
-----
``(row, col, has_key, energy_remaining)``

* ``has_key`` is 0 or 1 and flips permanently when the agent enters the key cell.
* ``energy_remaining`` is the extra capability of this project: every action
  costs one unit of energy and the episode fails when it reaches zero.

Dynamics
--------
Actions are UP/RIGHT/DOWN/LEFT. The intended move happens with probability
``p_intended`` (0.8) and each of the two perpendicular moves happens with
probability ``p_slip`` (0.1). The agent never moves backwards. Walking into a
wall, into the outer boundary, or into a still-locked door leaves the agent in
place, and the energy is spent anyway.

Terminal conditions
-------------------
1. success -- the agent stands on the goal while ``has_key == 1``
2. failure -- energy reaches zero
3. failure -- ``max_steps`` actions have been taken (episode truncation)

The first two are part of the MDP, so ``transitions()`` models them exactly and
Value Iteration solves the true problem. ``max_steps`` is an episode-level
safety limit applied by ``step()``; the config keeps ``max_energy <= max_steps``
so energy is always the binding constraint.
"""

from __future__ import annotations

import json
from collections import deque
from enum import IntEnum
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Cell(IntEnum):
    PATH = 0
    WALL = 1
    START = 2
    KEY = 3
    DOOR = 4
    GOAL = 5
    PENALTY = 6


DELTAS: Dict[Action, Tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.RIGHT: (0, 1),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
}

PERPENDICULAR: Dict[Action, Tuple[Action, Action]] = {
    Action.UP: (Action.LEFT, Action.RIGHT),
    Action.DOWN: (Action.LEFT, Action.RIGHT),
    Action.LEFT: (Action.UP, Action.DOWN),
    Action.RIGHT: (Action.UP, Action.DOWN),
}

RENDER_SYMBOLS = {
    Cell.PATH: ".",
    Cell.WALL: "#",
    Cell.START: "S",
    Cell.KEY: "K",
    Cell.DOOR: "D",
    Cell.GOAL: "G",
    Cell.PENALTY: "x",
}

REWARD_MODES = ("sparse", "shaped")

#: Per-episode event counters, reset by ``MazeEnv.reset()``.
EVENT_KEYS = (
    "wall_collisions",
    "penalty_entries",
    "door_blocked",
    "door_crossings",
    "key_pickups",
    "slips",
)

#: Base rewards (used by both modes) plus the shaped-mode extras.
DEFAULT_REWARDS: Dict[str, float] = {
    # base -- active in "sparse" and "shaped"
    "step_cost": -0.05,
    "key_reward": 10.0,
    "goal_reward": 100.0,
    # extras -- active in "shaped" only
    "shaping_scale": 0.5,
    "wall_collision": -1.0,
    "locked_door": -1.0,
    "penalty_cell": -5.0,
    "energy_exhausted": -20.0,
}

N_ACTIONS = len(Action)

State = Tuple[int, int, int, int]


def bfs_distances(
    grid: np.ndarray,
    sources: Iterable[Tuple[int, int]],
    door_passable: bool = True,
    blocked: Iterable[Tuple[int, int]] = (),
) -> np.ndarray:
    """Breadth-first grid distances from ``sources``; ``-1`` means unreachable."""
    rows, cols = grid.shape
    dist = np.full((rows, cols), -1, dtype=int)
    blocked = set(blocked)
    queue: deque = deque()
    for cell in sources:
        if cell in blocked:
            continue
        dist[cell] = 0
        queue.append(cell)

    while queue:
        r, c = queue.popleft()
        for dr, dc in DELTAS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if dist[nr, nc] != -1 or (nr, nc) in blocked:
                continue
            cell_type = grid[nr, nc]
            if cell_type == Cell.WALL:
                continue
            if cell_type == Cell.DOOR and not door_passable:
                continue
            dist[nr, nc] = dist[r, c] + 1
            queue.append((nr, nc))
    return dist


class MazeEnv:
    """The stochastic maze MDP."""

    n_actions = N_ACTIONS

    def __init__(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        key: Tuple[int, int],
        door: Tuple[int, int],
        goal: Tuple[int, int],
        penalties: Sequence[Tuple[int, int]],
        max_energy: int,
        max_steps: int,
        p_intended: float = 0.8,
        p_slip: float = 0.1,
        reward_mode: str = "shaped",
        rewards: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        self.grid = np.array(grid, dtype=np.int8)
        self.rows, self.cols = self.grid.shape
        self.start = tuple(start)
        self.key = tuple(key)
        self.door = tuple(door)
        self.goal = tuple(goal)
        self.penalties = [tuple(p) for p in penalties]
        self.penalty_set = set(self.penalties)

        self.max_energy = int(max_energy)
        self.max_steps = int(max_steps)

        if abs(p_intended + 2 * p_slip - 1.0) > 1e-9:
            raise ValueError(
                f"transition probabilities must sum to 1, got "
                f"{p_intended} + 2*{p_slip}"
            )
        self.p_intended = float(p_intended)
        self.p_slip = float(p_slip)

        self.rewards = dict(DEFAULT_REWARDS)
        if rewards:
            unknown = set(rewards) - set(DEFAULT_REWARDS)
            if unknown:
                raise ValueError(f"unknown reward keys: {sorted(unknown)}")
            self.rewards.update({k: float(v) for k, v in rewards.items()})

        self.set_reward_mode(reward_mode)
        self.metadata = dict(metadata or {})

        self.passable_cells = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r, c] != Cell.WALL
        ]
        self.n_passable = len(self.passable_cells)

        # Potentials for reward shaping: distance to the key while the agent has
        # no key (the door blocks the way) and distance to the goal once it has.
        self.dist_to_key = bfs_distances(self.grid, [self.key], door_passable=False)
        self.dist_to_goal = bfs_distances(self.grid, [self.goal], door_passable=True)

        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.agent_pos = self.start
        self.has_key = 0
        self.energy = self.max_energy
        self.steps = 0
        self.done = False
        self.outcome: Optional[str] = None
        self.reset()

    # ------------------------------------------------------------------ setup

    def set_reward_mode(self, reward_mode: str) -> None:
        if reward_mode not in REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {REWARD_MODES}, got {reward_mode!r}"
            )
        self.reward_mode = reward_mode

    def copy(self, reward_mode: Optional[str] = None, seed: Optional[int] = None) -> "MazeEnv":
        return MazeEnv(
            self.grid,
            start=self.start,
            key=self.key,
            door=self.door,
            goal=self.goal,
            penalties=self.penalties,
            max_energy=self.max_energy,
            max_steps=self.max_steps,
            p_intended=self.p_intended,
            p_slip=self.p_slip,
            reward_mode=reward_mode or self.reward_mode,
            rewards=self.rewards,
            seed=self._seed if seed is None else seed,
            metadata=self.metadata,
        )

    # ------------------------------------------------------------- interaction

    def reset(self, seed: Optional[int] = None) -> State:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = seed
        self.agent_pos = self.start
        self.has_key = 0
        self.energy = self.max_energy
        self.steps = 0
        self.done = False
        self.outcome = None
        self.episode_reward = 0.0
        self.events = {key: 0 for key in EVENT_KEYS}
        return self.state

    @property
    def state(self) -> State:
        return (self.agent_pos[0], self.agent_pos[1], self.has_key, self.energy)

    def is_terminal(self, state: State) -> bool:
        r, c, has_key, energy = state
        if energy <= 0:
            return True
        return (r, c) == self.goal and has_key == 1

    def sample_move(self, action: int) -> Action:
        """Draw the realised move for an intended ``action``."""
        action = Action(action)
        left, right = PERPENDICULAR[action]
        moves = (action, left, right)
        probs = (self.p_intended, self.p_slip, self.p_slip)
        draw = self.rng.random()
        cumulative = 0.0
        for move, prob in zip(moves, probs):
            cumulative += prob
            if draw < cumulative:
                return move
        return moves[-1]

    def step(self, action: int) -> Tuple[State, float, bool, dict]:
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first")

        move = self.sample_move(action)
        next_state, reward, done, info = self.apply_move(self.state, move)

        self.agent_pos = (next_state[0], next_state[1])
        self.has_key = next_state[2]
        self.energy = next_state[3]
        self.steps += 1
        info["intended_action"] = int(action)
        info["realised_move"] = int(move)
        info["slipped"] = int(move) != int(action)

        self.episode_reward += reward
        self.events["wall_collisions"] += int(info["collision"])
        self.events["penalty_entries"] += int(info["penalty_cell"])
        self.events["door_blocked"] += int(info["locked_door"])
        self.events["door_crossings"] += int(info["door_crossed"])
        self.events["key_pickups"] += int(info["picked_key"])
        self.events["slips"] += int(info["slipped"])

        if done:
            self.outcome = "success" if info["success"] else "out_of_energy"
        elif self.steps >= self.max_steps:
            done = True
            info["truncated"] = True
            self.outcome = "max_steps"

        self.done = done
        info["outcome"] = self.outcome
        return next_state, reward, done, info

    # ----------------------------------------------------------------- dynamics

    def apply_move(self, state: State, move: int) -> Tuple[State, float, bool, dict]:
        """Deterministic outcome of a *realised* ``move`` (slip already resolved)."""
        r, c, has_key, energy = state
        dr, dc = DELTAS[Action(move)]
        nr, nc = r + dr, c + dc

        collision = False
        locked_door = False
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            nr, nc = r, c
            collision = True
        elif self.grid[nr, nc] == Cell.WALL:
            nr, nc = r, c
            collision = True
        elif self.grid[nr, nc] == Cell.DOOR and not has_key:
            nr, nc = r, c
            locked_door = True

        next_energy = energy - 1
        next_has_key = has_key
        picked_key = False
        if not next_has_key and (nr, nc) == self.key:
            next_has_key = 1
            picked_key = True

        success = (nr, nc) == self.goal and next_has_key == 1
        exhausted = (not success) and next_energy <= 0
        next_state: State = (nr, nc, next_has_key, max(next_energy, 0))

        info = {
            "collision": collision,
            "locked_door": locked_door,
            "door_crossed": (nr, nc) == self.door and (nr, nc) != (r, c),
            "picked_key": picked_key,
            "penalty_cell": (nr, nc) in self.penalty_set,
            "success": success,
            "energy_exhausted": exhausted,
            "truncated": False,
        }
        reward = self._reward(state, next_state, info)
        return next_state, reward, success or exhausted, info

    def transitions(self, state: State, action: int) -> Tuple[Tuple[float, State, float, bool], ...]:
        """Full stochastic model: ``(probability, next_state, reward, done)``."""
        if self.is_terminal(state):
            return ()
        action = Action(action)
        left, right = PERPENDICULAR[action]
        candidates = (
            (action, self.p_intended),
            (left, self.p_slip),
            (right, self.p_slip),
        )

        merged: Dict[Tuple[State, bool], List[float]] = {}
        for move, prob in candidates:
            if prob <= 0.0:
                continue
            next_state, reward, done, _ = self.apply_move(state, move)
            entry = merged.setdefault((next_state, done), [0.0, 0.0])
            entry[0] += prob
            entry[1] += prob * reward

        return tuple(
            (prob, next_state, reward_sum / prob, done)
            for (next_state, done), (prob, reward_sum) in merged.items()
        )

    def states(self) -> Iterator[State]:
        """Every reachable-in-principle state of the MDP."""
        for r, c in self.passable_cells:
            for has_key in (0, 1):
                for energy in range(self.max_energy + 1):
                    yield (r, c, has_key, energy)

    # ------------------------------------------------------------------ rewards

    def _reward(self, state: State, next_state: State, info: dict) -> float:
        reward = self.rewards["step_cost"]
        if info["picked_key"]:
            reward += self.rewards["key_reward"]
        if info["success"]:
            reward += self.rewards["goal_reward"]

        if self.reward_mode == "sparse":
            return reward

        if info["collision"]:
            reward += self.rewards["wall_collision"]
        if info["locked_door"]:
            reward += self.rewards["locked_door"]
        if info["penalty_cell"]:
            reward += self.rewards["penalty_cell"]
        if info["energy_exhausted"]:
            reward += self.rewards["energy_exhausted"]
        reward += self._shaping(state, next_state, info)
        return reward

    def _shaping(self, state: State, next_state: State, info: dict) -> float:
        """Progress shaping towards the current target (key, then goal)."""
        if info["picked_key"]:
            # The target switches on this very transition, so the two distance
            # maps are not comparable; ``key_reward`` already rewards it.
            return 0.0
        r, c, has_key, _ = state
        nr, nc, _, _ = next_state
        dist = self.dist_to_goal if has_key else self.dist_to_key
        d_prev, d_next = int(dist[r, c]), int(dist[nr, nc])
        if d_prev < 0 or d_next < 0:
            return 0.0
        return self.rewards["shaping_scale"] * float(d_prev - d_next)

    # ---------------------------------------------------------------- rendering

    def render(self, show_agent: bool = True) -> str:
        lines = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if show_agent and (r, c) == self.agent_pos:
                    row.append("A")
                elif (r, c) == self.door and self.has_key:
                    row.append("d")
                else:
                    row.append(RENDER_SYMBOLS.get(Cell(self.grid[r, c]), "?"))
            lines.append("".join(row))
        text = "\n".join(lines)
        print(text)
        return text

    def episode_summary(self) -> dict:
        """Everything the episode CSV logs need about the episode just finished."""
        return {
            "steps": self.steps,
            "episode_reward": self.episode_reward,
            "outcome": self.outcome or "running",
            "energy_left": self.energy,
            "has_key": int(self.has_key),
            "goal_success": int(self.outcome == "success"),
            "energy_exhausted": int(self.outcome == "out_of_energy"),
            "max_steps_reached": int(self.outcome == "max_steps"),
            **self.events,
        }

    def summary(self) -> str:
        wall_fraction = float(np.mean(self.grid == Cell.WALL))
        return (
            f"{self.rows}x{self.cols} maze | walls {wall_fraction:.1%} "
            f"({int((self.grid == Cell.WALL).sum())} cells) | passable {self.n_passable} | "
            f"penalties {len(self.penalties)}\n"
            f"start {self.start} -> key {self.key} -> door {self.door} -> goal {self.goal}\n"
            f"max_energy {self.max_energy} | max_steps {self.max_steps} | "
            f"p_intended {self.p_intended} | p_slip {self.p_slip} | "
            f"reward_mode {self.reward_mode}"
        )

    # -------------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        return {
            "grid": self.grid.tolist(),
            "start": list(self.start),
            "key": list(self.key),
            "door": list(self.door),
            "goal": list(self.goal),
            "penalties": [list(p) for p in self.penalties],
            "episode": {"max_energy": self.max_energy, "max_steps": self.max_steps},
            "dynamics": {"p_intended": self.p_intended, "p_slip": self.p_slip},
            "metadata": self.metadata,
        }

    def save_map(self, filepath) -> Path:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return filepath

    @classmethod
    def from_dict(
        cls,
        data: dict,
        reward_mode: str = "shaped",
        rewards: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 0,
        max_energy: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> "MazeEnv":
        episode = data.get("episode", {})
        dynamics = data.get("dynamics", {})
        return cls(
            np.array(data["grid"], dtype=np.int8),
            start=tuple(data["start"]),
            key=tuple(data["key"]),
            door=tuple(data["door"]),
            goal=tuple(data["goal"]),
            penalties=[tuple(p) for p in data["penalties"]],
            max_energy=episode["max_energy"] if max_energy is None else max_energy,
            max_steps=episode["max_steps"] if max_steps is None else max_steps,
            p_intended=dynamics.get("p_intended", 0.8),
            p_slip=dynamics.get("p_slip", 0.1),
            reward_mode=reward_mode,
            rewards=rewards,
            seed=seed,
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load_map(cls, filepath, **kwargs) -> "MazeEnv":
        with open(filepath, encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data, **kwargs)
