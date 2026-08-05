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

import heapq
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

    # The door's own neighbours are frozen too, otherwise a flip can wall off the
    # approach to the door and leave the vault unreachable from any direction.
    for dr, dc in DELTAS.values():
        neighbour = (env.door[0] + dr, env.door[1] + dc)
        if 0 <= neighbour[0] < env.rows and 0 <= neighbour[1] < env.cols:
            protected.add(neighbour)
    return protected


def sealed_cells(env: MazeEnv) -> Set[Coord]:
    """The vault and its wall frontier: cells a repair corridor must never touch."""
    vault = {tuple(cell) for cell in env.metadata.get("vault_cells", [])}
    sealed = set(vault)
    for row, col in vault:
        for dr, dc in DELTAS.values():
            neighbour = (row + dr, col + dc)
            if neighbour in vault or neighbour == env.door:
                continue
            if 0 <= neighbour[0] < env.rows and 0 <= neighbour[1] < env.cols:
                sealed.add(neighbour)
    return sealed


def door_approach(env: MazeEnv) -> Coord:
    """The open cell just outside the door, which the agent must be able to reach."""
    vault = {tuple(cell) for cell in env.metadata.get("vault_cells", [])}
    for dr, dc in DELTAS.values():
        neighbour = (env.door[0] + dr, env.door[1] + dc)
        if neighbour in vault:
            continue
        if not (0 <= neighbour[0] < env.rows and 0 <= neighbour[1] < env.cols):
            continue
        if env.grid[neighbour] != Cell.WALL:
            return neighbour
    raise VariantError("the source door has no open approach cell")


def _bridge(grid: np.ndarray, start: Coord, target: Coord, sealed: Set[Coord]) -> bool:
    """Reconnect ``target`` to ``start`` by opening as few walls as possible.

    A shortest-path search where stepping onto open floor is free and breaking
    through a wall costs one. The cheapest route is therefore the thinnest wall
    between the two components, which keeps the corridor structure of the maze
    intact instead of cutting a straight highway across it.
    """
    rows, cols = grid.shape
    reachable = bfs_distances(grid, [start], door_passable=False) >= 0
    cost = np.full((rows, cols), np.inf)
    previous: Dict[Coord, Optional[Coord]] = {}
    queue: List[Tuple[int, Coord]] = []

    for row, col in zip(*np.where(reachable)):
        cell = (int(row), int(col))
        cost[cell] = 0.0
        previous[cell] = None
        heapq.heappush(queue, (0, cell))

    while queue:
        spent, cell = heapq.heappop(queue)
        if spent > cost[cell]:
            continue
        if cell == target:
            break
        for dr, dc in DELTAS.values():
            neighbour = (cell[0] + dr, cell[1] + dc)
            if not (0 <= neighbour[0] < rows and 0 <= neighbour[1] < cols):
                continue
            if neighbour in sealed or grid[neighbour] == Cell.DOOR:
                continue
            step = 1 if grid[neighbour] == Cell.WALL else 0
            if spent + step < cost[neighbour]:
                cost[neighbour] = spent + step
                previous[neighbour] = cell
                heapq.heappush(queue, (spent + step, neighbour))

    if not np.isfinite(cost[target]):
        return False

    cell: Optional[Coord] = target
    while cell is not None:
        if grid[cell] == Cell.WALL:
            grid[cell] = Cell.PATH
        cell = previous.get(cell)
    return True


def _repair(
    grid: np.ndarray, start: Coord, targets: Sequence[Coord], sealed: Set[Coord]
) -> bool:
    """Reconnect anything the flips cut off.

    Closing a few dozen open cells at random disconnects the maze more often than
    not, so rejecting every such layout would need an unreasonable number of
    attempts. Punching through the thinnest dividing wall is cheaper and leaves
    the rest of the perturbation intact.
    """
    for target in targets:
        if bfs_distances(grid, [start], door_passable=False)[target] >= 0:
            continue
        if not _bridge(grid, start, target, sealed):
            return False
    return True


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

    # Prefer the far end of the maze, so the route to the key genuinely changes
    # rather than the key landing next to the door.
    candidates.sort(key=lambda cell: -int(reachable[cell]))
    pool = candidates[: max(1, len(candidates) // 6)]
    return pool[int(rng.integers(len(pool)))]


def _open_cells(grid: np.ndarray, protected: Set[Coord]) -> List[Coord]:
    return [
        (int(r), int(c))
        for r, c in sorted(zip(*np.where(grid == Cell.PATH)))
        if (int(r), int(c)) not in protected
    ]


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
    sealed = sealed_cells(source)
    approach = door_approach(source)
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

            # Reconnect first, so the key is then placed using real distances on
            # the layout the agent will actually see.
            if not _repair(grid, start, [approach], sealed):
                failures.append(f"seed {seed + attempt}: could not reconnect the door")
                continue

            if settings["move_key"]:
                grid[key] = Cell.PATH
                key = _relocate_key(grid, rng, start, protected | {goal, door})

            if not _repair(grid, start, [key], sealed):
                failures.append(f"seed {seed + attempt}: could not reconnect the key")
                continue

            if settings["move_key"]:
                penalties = sorted(
                    set(penalties)
                    | set(
                        _add_penalties(
                            grid, rng, settings["extra_penalties"], start, key, goal, door
                        )
                    )
                )

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
    """True where a cell and the cells it can move into are all unchanged.

    This is the mask selective transfer uses. The neighbourhood is the cross of
    cells within ``radius`` steps rather than the full square block, because that
    is exactly what determines the outcome of an action from this cell: diagonal
    walls never affect where the agent ends up, so counting them would discard
    Q-values that are still perfectly valid.
    """
    changed = (source.grid == Cell.WALL) != (target.grid == Cell.WALL)
    rows, cols = changed.shape
    dirty = changed.copy()

    for step in range(1, radius + 1):
        dirty[step:, :] |= changed[: rows - step, :]
        dirty[: rows - step, :] |= changed[step:, :]
        dirty[:, step:] |= changed[:, : cols - step]
        dirty[:, : cols - step] |= changed[:, step:]
    return ~dirty
