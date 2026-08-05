"""Tests for the transfer target mazes.

The two targets are only useful if they are genuinely different from the source
*and* still solvable, so these tests check the change budget, the landmarks and
all three BFS conditions that define the task.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from environments.generator import MazeGenerator
from environments.maze import Cell, bfs_distances
from environments.variants import (
    VARIANT_SETTINGS,
    make_variant,
    obstacle_difference,
    unchanged_neighbourhood_mask,
)

SEED = 87_654_321


def build_source():
    return MazeGenerator(student_id="40305054").generate(reward_mode="shaped")


class TestVariantStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = build_source()
        cls.targets = {
            name: make_variant(cls.source, name, seed=SEED)
            for name in VARIANT_SETTINGS
        }

    def test_exactly_one_of_each_landmark(self):
        for name, target in self.targets.items():
            with self.subTest(variant=name):
                for cell_type in (Cell.START, Cell.KEY, Cell.DOOR, Cell.GOAL):
                    self.assertEqual(
                        int((target.grid == cell_type).sum()), 1,
                        f"{name} should have exactly one {cell_type.name} cell",
                    )
                self.assertEqual(target.grid[target.door], Cell.DOOR)
                self.assertEqual(target.grid[target.goal], Cell.GOAL)
                self.assertEqual(target.grid[target.key], Cell.KEY)

    def test_bfs_conditions_hold(self):
        for name, target in self.targets.items():
            with self.subTest(variant=name):
                shut = bfs_distances(target.grid, [target.start], door_passable=False)
                opened = bfs_distances(target.grid, [target.key], door_passable=True)
                self.assertGreaterEqual(shut[target.key], 0, "key must be reachable")
                self.assertLess(shut[target.goal], 0, "goal must stay sealed")
                self.assertGreaterEqual(opened[target.goal], 0, "goal must be reachable")

    def test_penalty_and_wall_budgets(self):
        for name, target in self.targets.items():
            with self.subTest(variant=name):
                self.assertGreaterEqual(len(target.penalties), 5)
                self.assertGreaterEqual(float(np.mean(target.grid == Cell.WALL)), 0.15)

    def test_similar_target_changes_15_to_20_percent(self):
        meta = self.targets["similar"].metadata
        self.assertGreaterEqual(meta["obstacle_change_fraction"], 0.15)
        self.assertLessEqual(meta["obstacle_change_fraction"], 0.20)

    def test_similar_target_keeps_start_key_and_goal(self):
        target = self.targets["similar"]
        self.assertEqual(target.start, self.source.start)
        self.assertEqual(target.key, self.source.key)
        self.assertEqual(target.goal, self.source.goal)

    def test_different_target_changes_about_35_percent_and_moves_the_key(self):
        target = self.targets["different"]
        meta = target.metadata
        self.assertGreaterEqual(meta["obstacle_change_fraction"], 0.30)
        self.assertLessEqual(meta["obstacle_change_fraction"], 0.40)
        self.assertNotEqual(target.key, self.source.key)
        self.assertGreater(len(target.penalties), len(self.source.penalties))

    def test_generation_is_deterministic(self):
        for name in VARIANT_SETTINGS:
            with self.subTest(variant=name):
                again = make_variant(self.source, name, seed=SEED)
                np.testing.assert_array_equal(again.grid, self.targets[name].grid)


class TestDifferenceHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = build_source()
        cls.target = make_variant(cls.source, "similar", seed=SEED)

    def test_obstacle_difference_counts_match_the_metadata(self):
        difference = obstacle_difference(self.source, self.target)
        self.assertEqual(
            int(np.abs(difference).sum()),
            self.target.metadata["obstacle_cells_changed"],
        )

    def test_reuse_mask_excludes_changed_cells_and_their_neighbours(self):
        mask = unchanged_neighbourhood_mask(self.source, self.target, radius=1)
        changed = (self.source.grid == Cell.WALL) != (self.target.grid == Cell.WALL)
        self.assertFalse(mask[changed].any(), "changed cells must never be reused")

        rows, cols = mask.shape
        for row, col in zip(*np.where(changed)):
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    self.assertFalse(
                        mask[nr, nc], "neighbours of a changed cell must not be reused"
                    )

    def test_identical_mazes_are_fully_reusable(self):
        mask = unchanged_neighbourhood_mask(self.source, self.source)
        self.assertTrue(mask.all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
