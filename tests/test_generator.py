"""Tests for the seeded maze generator and the assignment's layout constraints."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from environments.generator import MazeGenerator
from environments.maze import DELTAS, Cell, bfs_distances


class TestSeedRule(unittest.TestCase):
    def test_size_follows_the_assignment_formula(self):
        generator = MazeGenerator(student_id="40305054")
        self.assertEqual(generator.base_seed, 5)
        self.assertEqual(generator.size, 15 + (5 % 4))
        self.assertEqual(generator.size, 16)

    def test_generation_is_deterministic(self):
        first = MazeGenerator().generate()
        second = MazeGenerator().generate()
        np.testing.assert_array_equal(first.grid, second.grid)
        self.assertEqual(first.start, second.start)
        self.assertEqual(first.key, second.key)
        self.assertEqual(first.door, second.door)
        self.assertEqual(first.goal, second.goal)
        self.assertEqual(first.penalties, second.penalties)
        self.assertEqual(first.max_energy, second.max_energy)

    def test_different_seeds_give_different_mazes(self):
        first = MazeGenerator().generate()
        second = MazeGenerator(seed=987654).generate()
        self.assertFalse(np.array_equal(first.grid, second.grid))


class TestLayoutConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = MazeGenerator().generate()

    def test_at_least_15_percent_walls(self):
        fraction = float(np.mean(self.env.grid == Cell.WALL))
        self.assertGreaterEqual(fraction, 0.15)

    def test_at_least_five_penalty_cells(self):
        self.assertGreaterEqual(len(self.env.penalties), 5)
        for cell in self.env.penalties:
            self.assertEqual(self.env.grid[cell], Cell.PENALTY)

    def test_all_required_features_are_present(self):
        for cell_type in (Cell.WALL, Cell.START, Cell.KEY, Cell.DOOR, Cell.GOAL, Cell.PENALTY):
            self.assertTrue(
                (self.env.grid == cell_type).any(), f"missing {cell_type.name} cells"
            )
        self.assertEqual(int((self.env.grid == Cell.DOOR).sum()), 1)
        self.assertEqual(int((self.env.grid == Cell.GOAL).sum()), 1)
        self.assertEqual(int((self.env.grid == Cell.KEY).sum()), 1)

    def test_key_reachable_before_the_door_opens(self):
        distances = bfs_distances(self.env.grid, [self.env.start], door_passable=False)
        self.assertGreater(distances[self.env.key], 0)

    def test_goal_unreachable_until_the_door_opens(self):
        shut = bfs_distances(self.env.grid, [self.env.start], door_passable=False)
        self.assertEqual(shut[self.env.goal], -1)

        opened = bfs_distances(self.env.grid, [self.env.key], door_passable=True)
        self.assertGreater(opened[self.env.goal], 0)

    def test_the_door_is_the_only_way_into_the_vault(self):
        """Every non-wall cell bordering the goal area must be the door itself."""
        vault = {tuple(cell) for cell in self.env.metadata["vault_cells"]}
        self.assertIn(self.env.goal, vault)
        for r, c in vault:
            for dr, dc in DELTAS.values():
                neighbour = (r + dr, c + dc)
                if neighbour in vault:
                    continue
                if not (0 <= neighbour[0] < self.env.rows and 0 <= neighbour[1] < self.env.cols):
                    continue
                cell_type = self.env.grid[neighbour]
                if cell_type != Cell.WALL:
                    self.assertEqual(
                        neighbour,
                        self.env.door,
                        f"{neighbour} opens into the vault but is not the door",
                    )


class TestEpisodeBudgets(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = MazeGenerator().generate()

    def test_max_steps_matches_the_suggested_formula(self):
        passable = int((self.env.grid != Cell.WALL).sum())
        self.assertEqual(self.env.max_steps, max(200, 3 * passable))
        self.assertEqual(passable, self.env.metadata["n_passable_cells"])

    def test_energy_is_the_binding_constraint(self):
        self.assertLessEqual(self.env.max_energy, self.env.max_steps)

    def test_energy_budget_allows_the_optimal_route(self):
        optimal = self.env.metadata["optimal_path_length"]
        self.assertGreater(self.env.max_energy, optimal)
        self.assertEqual(
            optimal,
            self.env.metadata["d_start_key"] + self.env.metadata["d_key_goal"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
