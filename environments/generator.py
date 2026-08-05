"""Deterministic, seeded generation of the project maze.

Seed rule from the assignment::

    student_id = "40305054"
    base_seed  = int(student_id[-2])      # -> 5
    maze_size  = 15 + (base_seed % 4)     # -> 16

``base_seed`` fixes the size; the layout itself is drawn from a NumPy
``default_rng`` seeded with the full student id, so the same id always produces
byte-identical maps.

Layout pipeline
---------------
1. Randomised depth-first carving over the odd-indexed cells produces a fully
   connected maze (roughly 45% walls, far above the 15% minimum).
2. A few extra openings are knocked through to create loops, so the maze is not
   a single tree and the stochastic dynamics matter.
3. The bottom-right ``vault_span x vault_span`` block is opened into a small
   room and then sealed: every cell adjacent to it becomes a wall except one,
   which becomes the locked door. The goal is placed in the corner of the room
   farthest from that door, so it is only reachable through the door.
4. The key is the reachable cell farthest from the start while the door is shut.
5. Penalty cells are sampled from the remaining reachable cells.
6. BFS validates start -> key (door shut) and key -> goal (door open). Invalid
   layouts are repaired by carving a corridor and, failing that, regenerated
   from the next seed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .maze import DELTAS, Cell, MazeEnv, bfs_distances

Coord = Tuple[int, int]


class LayoutError(RuntimeError):
    """Raised when a candidate layout cannot be built from the current seed."""


class MazeGenerator:
    def __init__(
        self,
        student_id: str = "40305054",
        seed: Optional[int] = None,
        size: Optional[int] = None,
        min_wall_fraction: float = 0.15,
        n_penalty_cells: int = 8,
        loop_openings: int = 12,
        vault_span: int = 3,
        energy_slack: float = 2.5,
        max_attempts: int = 20,
    ) -> None:
        self.student_id = str(student_id)
        self.base_seed = int(self.student_id[-2])
        self.size = int(size) if size is not None else 15 + (self.base_seed % 4)
        self.seed = int(seed) if seed is not None else int(self.student_id)
        self.min_wall_fraction = float(min_wall_fraction)
        self.n_penalty_cells = int(n_penalty_cells)
        self.loop_openings = int(loop_openings)
        self.vault_span = int(vault_span)
        self.energy_slack = float(energy_slack)
        self.max_attempts = int(max_attempts)

        if self.size < 2 * self.vault_span + 3:
            raise ValueError("maze is too small for the requested vault span")
        if self.n_penalty_cells < 5:
            raise ValueError("the assignment requires at least 5 penalty cells")

    # ------------------------------------------------------------------- public

    def generate(
        self,
        reward_mode: str = "shaped",
        rewards: Optional[Dict[str, float]] = None,
        max_energy: Optional[int] = None,
        max_steps: Optional[int] = None,
        env_seed: Optional[int] = 0,
    ) -> MazeEnv:
        """Build the maze, validating (and if needed repairing) reachability."""
        failures: List[str] = []
        for attempt in range(self.max_attempts):
            layout_seed = self.seed + attempt
            rng = np.random.default_rng(layout_seed)
            try:
                layout = self._build_layout(rng)
            except LayoutError as exc:
                failures.append(f"seed {layout_seed}: {exc}")
                continue

            ok, reason = self._validate(layout)
            if not ok:
                self._repair(layout)
                ok, reason = self._validate(layout)
            if not ok:
                failures.append(f"seed {layout_seed}: {reason}")
                continue

            return self._build_env(
                layout,
                layout_seed=layout_seed,
                attempts=attempt + 1,
                reward_mode=reward_mode,
                rewards=rewards,
                max_energy=max_energy,
                max_steps=max_steps,
                env_seed=env_seed,
            )

        raise RuntimeError(
            "could not generate a valid maze after "
            f"{self.max_attempts} attempts:\n  " + "\n  ".join(failures)
        )

    def info(self) -> str:
        return (
            f"student_id {self.student_id} | base_seed {self.base_seed} | "
            f"maze_size {self.size}x{self.size} | layout seed {self.seed}"
        )

    # -------------------------------------------------------------- layout parts

    def _build_layout(self, rng: np.random.Generator) -> dict:
        grid = self._carve_maze(rng)
        self._add_loops(grid, rng)
        vault, door, seal = self._seal_vault(grid, rng)
        goal = self._pick_goal(vault, door)
        start = (1, 1)
        key = self._pick_key(grid, start, vault, door)
        penalties = self._pick_penalties(grid, rng, start, key, goal, door, vault)

        return {
            "grid": grid,
            "start": start,
            "key": key,
            "door": door,
            "goal": goal,
            "penalties": penalties,
            "vault": vault,
            "seal": seal,
        }

    def _carve_maze(self, rng: np.random.Generator) -> np.ndarray:
        """Randomised DFS over odd-indexed cells; guarantees connectivity."""
        n = self.size
        grid = np.full((n, n), Cell.WALL, dtype=np.int8)
        cells = [(r, c) for r in range(1, n, 2) for c in range(1, n, 2)]
        cell_set = set(cells)
        for cell in cells:
            grid[cell] = Cell.PATH

        start_cell = (1, 1)
        visited: Set[Coord] = {start_cell}
        stack: List[Coord] = [start_cell]
        while stack:
            r, c = stack[-1]
            options = [
                (r + dr, c + dc)
                for dr, dc in ((-2, 0), (0, 2), (2, 0), (0, -2))
                if (r + dr, c + dc) in cell_set and (r + dr, c + dc) not in visited
            ]
            if not options:
                stack.pop()
                continue
            nr, nc = options[int(rng.integers(len(options)))]
            grid[(r + nr) // 2, (c + nc) // 2] = Cell.PATH
            visited.add((nr, nc))
            stack.append((nr, nc))
        return grid

    def _add_loops(self, grid: np.ndarray, rng: np.random.Generator) -> None:
        """Knock through walls that separate two carved cells, creating cycles."""
        n = self.size
        candidates: List[Coord] = []
        for r in range(1, n):
            for c in range(1, n):
                if grid[r, c] != Cell.WALL:
                    continue
                if r % 2 == 1 and c % 2 == 0:
                    sides = ((r, c - 1), (r, c + 1))
                elif r % 2 == 0 and c % 2 == 1:
                    sides = ((r - 1, c), (r + 1, c))
                else:
                    continue
                if any(not (0 <= x < n) for side in sides for x in side):
                    continue
                if all(grid[side] != Cell.WALL for side in sides):
                    candidates.append((r, c))

        if not candidates:
            return
        order = rng.permutation(len(candidates))
        for index in order[: self.loop_openings]:
            grid[candidates[int(index)]] = Cell.PATH

    def _seal_vault(
        self, grid: np.ndarray, rng: np.random.Generator
    ) -> Tuple[List[Coord], Coord, Set[Coord]]:
        """Open the corner room, then wall it off except for a single door."""
        n, span = self.size, self.vault_span
        vault = [(r, c) for r in range(n - span, n) for c in range(n - span, n)]
        vault_set = set(vault)
        for cell in vault:
            grid[cell] = Cell.PATH

        frontier: Set[Coord] = set()
        for r, c in vault:
            for dr, dc in DELTAS.values():
                neighbour = (r + dr, c + dc)
                if neighbour in vault_set:
                    continue
                if 0 <= neighbour[0] < n and 0 <= neighbour[1] < n:
                    frontier.add(neighbour)

        # A usable door must also connect outwards to the rest of the maze.
        boundary = n - span - 1
        candidates = []
        for cell in sorted(frontier):
            r, c = cell
            outward = (r - 1, c) if r == boundary else (r, c - 1)
            if not (0 <= outward[0] < n and 0 <= outward[1] < n):
                continue
            if grid[outward] != Cell.WALL:
                candidates.append(cell)
        if not candidates:
            raise LayoutError("no frontier cell can serve as a door")

        already_open = [cell for cell in candidates if grid[cell] != Cell.WALL]
        pool = already_open or candidates
        door = pool[int(rng.integers(len(pool)))]

        for cell in frontier:
            grid[cell] = Cell.WALL
        grid[door] = Cell.DOOR

        seal = {cell for cell in frontier if cell != door}
        return vault, door, seal

    def _pick_goal(self, vault: Sequence[Coord], door: Coord) -> Coord:
        """Deepest cell of the room measured from the door's entry cell."""
        boundary = self.size - self.vault_span - 1
        r, c = door
        entry = (r + 1, c) if r == boundary else (r, c + 1)
        return max(
            sorted(vault),
            key=lambda cell: abs(cell[0] - entry[0]) + abs(cell[1] - entry[1]),
        )

    def _pick_key(
        self, grid: np.ndarray, start: Coord, vault: Sequence[Coord], door: Coord
    ) -> Coord:
        """Farthest cell reachable from the start while the door is still shut."""
        blocked = set(vault) | {door}
        dist = bfs_distances(grid, [start], door_passable=False, blocked=blocked)
        candidates = [
            cell
            for cell in sorted(zip(*np.where(dist > 0)))
            if (int(cell[0]), int(cell[1])) not in blocked
        ]
        if not candidates:
            raise LayoutError("no reachable cell available for the key")
        best = max(candidates, key=lambda cell: (int(dist[cell]), cell))
        return (int(best[0]), int(best[1]))

    def _pick_penalties(
        self,
        grid: np.ndarray,
        rng: np.random.Generator,
        start: Coord,
        key: Coord,
        goal: Coord,
        door: Coord,
        vault: Sequence[Coord],
    ) -> List[Coord]:
        excluded = {start, key, goal, door} | set(vault)
        reachable = bfs_distances(grid, [start], door_passable=True)
        candidates = [
            (int(r), int(c))
            for r, c in sorted(zip(*np.where(reachable > 0)))
            if (int(r), int(c)) not in excluded
        ]
        if len(candidates) < self.n_penalty_cells:
            raise LayoutError("not enough free cells for the penalty cells")
        chosen = rng.choice(len(candidates), size=self.n_penalty_cells, replace=False)
        return sorted(candidates[int(i)] for i in chosen)

    # ------------------------------------------------------- validate and repair

    def _validate(self, layout: dict) -> Tuple[bool, str]:
        grid = layout["grid"]
        start, key, goal, door = (
            layout["start"],
            layout["key"],
            layout["goal"],
            layout["door"],
        )

        wall_fraction = float(np.mean(grid == Cell.WALL))
        if wall_fraction < self.min_wall_fraction:
            return False, f"wall fraction {wall_fraction:.3f} below minimum"
        if len(layout["penalties"]) < 5:
            return False, "fewer than 5 penalty cells"

        shut = bfs_distances(grid, [start], door_passable=False)
        if shut[key] < 0:
            return False, "key not reachable from start with the door shut"
        if shut[goal] >= 0:
            return False, "goal reachable without the door"

        opened = bfs_distances(grid, [key], door_passable=True)
        if opened[goal] < 0:
            return False, "goal not reachable from key with the door open"
        return True, "ok"

    def _repair(self, layout: dict) -> None:
        """Safety net: carve straight corridors to restore reachability."""
        grid = layout["grid"]
        protected = set(layout["seal"])
        self._carve_corridor(grid, layout["start"], layout["key"], protected)
        self._carve_corridor(grid, layout["key"], layout["door"], protected)

    def _carve_corridor(
        self, grid: np.ndarray, source: Coord, target: Coord, protected: Set[Coord]
    ) -> None:
        r, c = source
        tr, tc = target
        while r != tr:
            r += 1 if tr > r else -1
            if (r, c) not in protected and grid[r, c] == Cell.WALL:
                grid[r, c] = Cell.PATH
        while c != tc:
            c += 1 if tc > c else -1
            if (r, c) not in protected and grid[r, c] == Cell.WALL:
                grid[r, c] = Cell.PATH

    # --------------------------------------------------------------- environment

    def _build_env(
        self,
        layout: dict,
        layout_seed: int,
        attempts: int,
        reward_mode: str,
        rewards: Optional[Dict[str, float]],
        max_energy: Optional[int],
        max_steps: Optional[int],
        env_seed: Optional[int],
    ) -> MazeEnv:
        grid = layout["grid"].copy()
        start, key, door, goal = (
            layout["start"],
            layout["key"],
            layout["door"],
            layout["goal"],
        )
        penalties = layout["penalties"]

        d_start_key = int(bfs_distances(grid, [start], door_passable=False)[key])
        d_key_goal = int(bfs_distances(grid, [key], door_passable=True)[goal])
        optimal_path = d_start_key + d_key_goal

        n_passable = int((grid != Cell.WALL).sum())
        computed_max_steps = max(200, 3 * n_passable)
        computed_max_energy = min(
            computed_max_steps,
            max(math.ceil(self.energy_slack * optimal_path), optimal_path + 20),
        )
        final_max_steps = computed_max_steps if max_steps is None else int(max_steps)
        final_max_energy = min(
            final_max_steps,
            computed_max_energy if max_energy is None else int(max_energy),
        )

        grid[start] = Cell.START
        grid[key] = Cell.KEY
        grid[goal] = Cell.GOAL
        for cell in penalties:
            grid[cell] = Cell.PENALTY

        metadata = {
            "student_id": self.student_id,
            "base_seed": self.base_seed,
            "size": self.size,
            "layout_seed": layout_seed,
            "generation_attempts": attempts,
            "min_wall_fraction": self.min_wall_fraction,
            "wall_fraction": round(float(np.mean(grid == Cell.WALL)), 4),
            "n_wall_cells": int((grid == Cell.WALL).sum()),
            "n_passable_cells": n_passable,
            "n_penalty_cells": len(penalties),
            "vault_cells": [list(cell) for cell in sorted(layout["vault"])],
            "d_start_key": d_start_key,
            "d_key_goal": d_key_goal,
            "optimal_path_length": optimal_path,
            "energy_slack": self.energy_slack,
            "computed_max_steps": computed_max_steps,
            "computed_max_energy": computed_max_energy,
        }

        return MazeEnv(
            grid,
            start=start,
            key=key,
            door=door,
            goal=goal,
            penalties=penalties,
            max_energy=final_max_energy,
            max_steps=final_max_steps,
            reward_mode=reward_mode,
            rewards=rewards,
            seed=env_seed,
            metadata=metadata,
        )
