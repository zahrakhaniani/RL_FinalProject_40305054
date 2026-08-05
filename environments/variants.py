"""Perturbed copies of the source maze, used as transfer-learning targets.

Two targets are built from the same source layout:

``similar`` (Target A)
    15-20% of the cells flip between wall and open floor. Start, key and goal
    stay exactly where they were, so the task is the same and only the corridors
    change.

``different`` (Target B)
    about 35% of the cells flip, the key moves somewhere else, and extra penalty
    cells appear. The task itself changes, not just the route to it.

Both keep the sealed goal vault and its single locked door intact -- those cells
are never touched -- and both are BFS validated with the same three checks the
generator uses: the key must be reachable while the door is shut, the goal must
*not* be, and the goal must be reachable from the key once the door opens. Flips
are applied in balanced pairs (one wall opened for every floor cell closed) so
the wall density stays close to the source, and a layout that fails validation is
regenerated from the next seed.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .maze import DELTAS, Cell, MazeEnv, bfs_distances

Coord = Tuple[int, int]

VARIANTS = ("similar", "different")

VARIANT_SETTINGS = {
    # change_fraction, move_key, extra_penalties
    "similar": {"change_fraction": 0.18, "move_key": False, "extra_penalties": 0},
    "different": {"change_fraction": 0.35, "move_key": True, "extra_penalties": 4},
}


class VariantError(RuntimeError):
    """Raised when a perturbation cannot produce a solvable maze."""


def protected_cells(env: MazeEnv) -> Set[Coord]:
    """Cells whose type must never change, so the vault stays sealed."""
    vault = {tuple(cell) for cell in env.metadata.get("vault_cells", [])}
    if not vault:
        vault = set()
    protected = {env.start, env.key, env.goal, env.door} | vault

    # Everything bordering the vault has to stay a wall except the door itself.
    for row, col in vault:
        for dr, dc in DELTAS.values():
            neighbour = (row + dr, col + dc)
            if neighbour in vault:
                continue
            if 0 <= neighbour[0] < env.rows and 0 <= neighbour[1] < env.cols:
                protected.add(neighbour)
    return protected


def _validate(grid: np.ndarray, start: Coord, key: Coord, goal: Coord) -> Tuple[bool, str]:
    shut = bfs_distances(grid, [start], door_passable=False)
    if shut[key] < 0:
        return False, "key unreachable with the door shut"
    if shut[goal] >= 0:
        return False, "goal reachable without the door"
    opened = bfs_distances(grid, [key], door_passable=True)
    if opened[goal] < 0:
        return False, "goal unreachable from the key"
    return True, "ok"


def _apply_flips(
    grid: np.ndarray,
    rng: np.random.Generator,
    protected: Set[Coord],
    n_changes: int,
) -> int:
    """Flip cells in balanced wall/floor pairs; returns how many actually changed."""
    rows, cols = grid.shape
    walls: List[Coord] = []
    floors: List[Coord] = []
    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            if cell in protected:
                continue
            if grid[cell] == Cell.WALL:
                walls.append(cell)
            elif grid[cell] == Cell.PATH:
                floors.append(cell)

    half = min(n_changes // 2, len(walls), len(floors))
    if half == 0:
        raise VariantError("no cells available to change")

    open_up = [walls[int(i)] for i in rng.choice(len(walls), size=half, replace=False)]
    close_off = [floors[int(i)] for i in rng.choice(len(floors), size=half, replace=False)]
    for cell in open_up:
        grid[cell] = Cell.PATH
    for cell in close_off:
        grid[cell] = Cell.WALL
    return 2 * half


def _relocate_key(
    grid: np.ndarray, rng: np.random.Generator, start: Coord, protected: Set[Coord]
) -> Coord:
    """Move the key to a far-away cell that is still reachable with the door shut."""
    reachable = bfs_distances(grid, [start], door_passable=False)
    candidates = [
        (int(r), int(c))
        for r, c in sorted(zip(*np.where(reachable > 0)))
        if (int(r), int(c)) not in protected
    ]
    if not candidates:
        raise VariantError("nowhere to move the key")

    # Prefer the far half of the maze so the task really changes.
    candidates.sort(key=lambda cell: -int(reachable[cell]))
    pool = candidates[: max(1, len(candidates) // 3)]
    return pool[int(rng.integers(len(pool)))]


def _add_penalties(
    grid: np.ndarray,
    rng: np.random.Generator,
    count: int,
    start: Coord,
    key: Coord,
    goal: Coord,
    door: Coord,
) -> List[Coord]:
    excluded = {start, key, goal, door}
    reachable = bfs_distances(grid, [start], door_passable=True)
    candidates = [
        (int(r), int(c))
        for r, c in sorted(zip(*np.where(reachable > 0)))
        if (int(r), int(c)) not in excluded and grid[int(r), int(c)] == Cell.PATH
    ]
    if len(candidates) < count:
        return []
    chosen = rng.choice(len(candidates), size=count, replace=False)
    return sorted(candidates[int(i)] for i in chosen)


def make_variant(
    source: MazeEnv,
    variant: str,
    seed: int,
    reward_mode: str = "shaped",
    rewards: Optional[Dict[str, float]] = None,
    max_attempts: int = 60,
) -> MazeEnv:
    """Build ``similar`` or ``different`` target maze from ``source``."""
    if variant not in VARIANT_SETTINGS:
        raise KeyError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    settings = VARIANT_SETTINGS[variant]
    protected = protected_cells(source)
    n_cells = source.rows * source.cols
    target_changes = int(round(settings["change_fraction"] * n_cells))

    failures: List[str] = []
    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        grid = source.grid.copy()
        start, key, goal, door = source.start, source.key, source.door, source.goal

        # Strip decorations back to plain floor so flips work on a clean grid.
        grid[grid == Cell.PENALTY] = Cell.PATH
        grid[grid == Cell.KEY] = Cell.PATH
        grid[grid == Cell.START] = Cell.PATH

        try:
            changed = _apply_flips(grid, rng, protected, target_changes)
            penalties = list(source.penalties)

            if settings["move_key"]:
                grid[key] = Cell.PATH
                key = _relocate_key(grid, rng, start, protected | {goal, door})
                extra = _add_penalties(
                    grid, rng, settings["extra_penalties"], start, key, goal, door
                )
                penalties = sorted(set(penalties) | set(extra))

            # Penalty cells only survive where the cell is still open floor.
            penalties = [
                cell for cell in penalties
                if grid[cell] == Cell.PATH and cell not in {start, key, goal, door}
            ]
            if len(penalties) < 5:
                penalties = sorted(
                    set(penalties)
                    | set(_add_penalties(grid, rng, 5 - len(penalties), start, key, goal, door))
                )

            ok, reason = _validate(grid, start, key, goal)
            if not ok:
                failures.append(f"seed {seed + attempt}: {reason}")
                continue
        except VariantError as error:
            failures.append(f"seed {seed + attempt}: {error}")
            continue

        return _finalise(
            source, grid, variant, seed + attempt, attempt + 1, changed,
            start, key, door, goal, penalties, reward_mode, rewards,
        )

    raise VariantError(
        f"could not build the {variant!r} target after {max_attempts} attempts:\n  "
        + "\n  ".join(failures)
    )


def _finalise(
    source: MazeEnv,
    grid: np.ndarray,
    variant: str,
    layout_seed: int,
    attempts: int,
    changed: int,
    start: Coord,
    key: Coord,
    door: Coord,
    goal: Coord,
    penalties: Sequence[Coord],
    reward_mode: str,
    rewards: Optional[Dict[str, float]],
) -> MazeEnv:
    d_start_key = int(bfs_distances(grid, [start], door_passable=False)[key])
    d_key_goal = int(bfs_distances(grid, [key], door_passable=True)[goal])
    optimal_path = d_start_key + d_key_goal

    n_passable = int((grid != Cell.WALL).sum())
    max_steps = max(200, 3 * n_passable)
    energy_slack = float(source.metadata.get("energy_slack", 2.5))
    max_energy = min(
        max_steps, max(math.ceil(energy_slack * optimal_path), optimal_path + 20)
    )

    grid[start] = Cell.START
    grid[key] = Cell.KEY
    grid[goal] = Cell.GOAL
    for cell in penalties:
        grid[cell] = Cell.PENALTY

    cells_changed = int(
        ((source.grid == Cell.WALL) != (grid == Cell.WALL)).sum()
    )
    metadata = dict(source.metadata)
    metadata.update(
        {
            "variant": variant,
            "source_layout_seed": source.metadata.get("layout_seed"),
            "layout_seed": layout_seed,
            "generation_attempts": attempts,
            "requested_changes": changed,
            "obstacle_cells_changed": cells_changed,
            "obstacle_change_fraction": round(cells_changed / (source.rows * source.cols), 4),
            "key_moved": key != source.key,
            "goal_moved": goal != source.goal,
            "wall_fraction": round(float(np.mean(grid == Cell.WALL)), 4),
            "n_wall_cells": int((grid == Cell.WALL).sum()),
            "n_passable_cells": n_passable,
            "n_penalty_cells": len(penalties),
            "d_start_key": d_start_key,
            "d_key_goal": d_key_goal,
            "optimal_path_length": optimal_path,
            "computed_max_steps": max_steps,
            "computed_max_energy": max_energy,
        }
    )

    return MazeEnv(
        grid,
        start=start,
        key=key,
        door=door,
        goal=goal,
        penalties=list(penalties),
        max_energy=max_energy,
        max_steps=max_steps,
        p_intended=source.p_intended,
        p_slip=source.p_slip,
        reward_mode=reward_mode,
        rewards=rewards or source.rewards,
        seed=0,
        metadata=metadata,
    )


def obstacle_difference(source: MazeEnv, target: MazeEnv) -> np.ndarray:
    """+1 where a wall was opened, -1 where floor was walled off, 0 unchanged."""
    source_wall = source.grid == Cell.WALL
    target_wall = target.grid == Cell.WALL
    difference = np.zeros(source.grid.shape, dtype=int)
    difference[source_wall & ~target_wall] = 1
    difference[~source_wall & target_wall] = -1
    return difference


def unchanged_neighbourhood_mask(
    source: MazeEnv, target: MazeEnv, radius: int = 1
) -> np.ndarray:
    """True where the cell and its whole (2r+1)^2 neighbourhood are identical.

    This is the mask that selective transfer uses: a Q-value is only worth
    keeping if the local geometry that produced it still looks the same.
    """
    source_wall = source.grid == Cell.WALL
    target_wall = target.grid == Cell.WALL
    rows, cols = source_wall.shape
    mask = np.zeros((rows, cols), dtype=bool)

    for row in range(rows):
        for col in range(cols):
            r0, r1 = max(0, row - radius), min(rows, row + radius + 1)
            c0, c1 = max(0, col - radius), min(cols, col + radius + 1)
            if np.array_equal(source_wall[r0:r1, c0:c1], target_wall[r0:r1, c0:c1]):
                mask[row, col] = True
    return mask
