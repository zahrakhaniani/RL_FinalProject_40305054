"""Tests for the stochastic maze MDP."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from environments.generator import MazeGenerator
from environments.maze import DELTAS, Action, Cell, MazeEnv


def build_env(reward_mode: str = "shaped") -> MazeEnv:
    return MazeGenerator().generate(reward_mode=reward_mode)


class TestTransitionModel(unittest.TestCase):
    def setUp(self) -> None:
        self.env = build_env()

    def sample_states(self, count: int = 200):
        rng = np.random.default_rng(0)
        states = []
        while len(states) < count:
            cell = self.env.passable_cells[int(rng.integers(self.env.n_passable))]
            state = (
                cell[0],
                cell[1],
                int(rng.integers(2)),
                int(rng.integers(1, self.env.max_energy + 1)),
            )
            if not self.env.is_terminal(state):
                states.append(state)
        return states

    def test_probabilities_sum_to_one(self):
        for state in self.sample_states():
            for action in range(4):
                total = sum(p for p, _, _, _ in self.env.transitions(state, action))
                self.assertAlmostEqual(total, 1.0, places=12)

    def test_intended_and_slip_probabilities(self):
        # An open crossroads keeps the three outcomes distinct so the individual
        # probabilities are directly observable.
        state = None
        for cell in self.env.passable_cells:
            r, c = cell
            neighbours = [
                (r + dr, c + dc)
                for dr, dc in DELTAS.values()
                if 0 <= r + dr < self.env.rows and 0 <= c + dc < self.env.cols
                and self.env.grid[r + dr, c + dc] != Cell.WALL
            ]
            if len(neighbours) == 4:
                state = (r, c, 1, self.env.max_energy)
                break
        self.assertIsNotNone(state, "expected at least one fully open cell")

        outcomes = self.env.transitions(state, Action.UP)
        probabilities = sorted(p for p, _, _, _ in outcomes)
        self.assertEqual(len(outcomes), 3)
        np.testing.assert_allclose(probabilities, [0.1, 0.1, 0.8])

        # The agent never moves backwards.
        reached = {(next_state[0], next_state[1]) for _, next_state, _, _ in outcomes}
        self.assertNotIn((state[0] + 1, state[1]), reached)

    def test_no_backward_move_in_any_direction(self):
        for state in self.sample_states(50):
            for action in Action:
                back_dr, back_dc = DELTAS[action]
                backward = (state[0] - back_dr, state[1] - back_dc)
                for _, next_state, _, _ in self.env.transitions(state, action):
                    moved = (next_state[0], next_state[1])
                    if moved != (state[0], state[1]):
                        self.assertNotEqual(moved, backward)

    def test_sampled_moves_match_the_model(self):
        """step() must draw from exactly the distribution transitions() reports."""
        state = (1, 1, 0, self.env.max_energy)
        expected = {}
        for probability, next_state, _, _ in self.env.transitions(state, Action.DOWN):
            expected[next_state[:2]] = probability

        counts = {cell: 0 for cell in expected}
        draws = 20000
        env = self.env.copy(seed=12345)
        for _ in range(draws):
            env.reset()
            env.agent_pos = (state[0], state[1])
            env.has_key = state[2]
            env.energy = state[3]
            next_state, _, _, _ = env.step(Action.DOWN)
            counts[next_state[:2]] += 1

        for cell, probability in expected.items():
            self.assertAlmostEqual(counts[cell] / draws, probability, delta=0.02)


class TestMovementRules(unittest.TestCase):
    def setUp(self) -> None:
        self.env = build_env()

    def test_wall_collision_keeps_position(self):
        state = (1, 1, 0, self.env.max_energy)
        self.assertEqual(self.env.grid[1, 2], Cell.WALL)
        next_state, reward, done, info = self.env.apply_move(state, Action.RIGHT)
        self.assertEqual(next_state[:2], (1, 1))
        self.assertTrue(info["collision"])
        self.assertFalse(done)

    def test_boundary_collision_keeps_position(self):
        cell = next(c for c in self.env.passable_cells if c[0] == 1)
        state = (cell[0], cell[1], 0, self.env.max_energy)
        next_state, _, _, info = self.env.apply_move(state, Action.UP)
        self.assertEqual(next_state[:2], cell)
        self.assertTrue(info["collision"])

    def test_locked_door_blocks_until_key_is_held(self):
        door = self.env.door
        approach = next(
            (door[0] - dr, door[1] - dc)
            for dr, dc in DELTAS.values()
            if 0 <= door[0] - dr < self.env.rows
            and 0 <= door[1] - dc < self.env.cols
            and self.env.grid[door[0] - dr, door[1] - dc] != Cell.WALL
        )
        move = next(
            action
            for action, (dr, dc) in DELTAS.items()
            if (approach[0] + dr, approach[1] + dc) == door
        )

        blocked, _, _, info = self.env.apply_move(
            (approach[0], approach[1], 0, self.env.max_energy), move
        )
        self.assertEqual(blocked[:2], approach)
        self.assertTrue(info["locked_door"])

        opened, _, _, info = self.env.apply_move(
            (approach[0], approach[1], 1, self.env.max_energy), move
        )
        self.assertEqual(opened[:2], door)
        self.assertFalse(info["locked_door"])

    def test_every_action_costs_one_energy(self):
        state = (1, 1, 0, self.env.max_energy)
        for action in Action:
            next_state, _, _, _ = self.env.apply_move(state, action)
            self.assertEqual(next_state[3], self.env.max_energy - 1)

    def test_energy_exhaustion_ends_the_episode(self):
        state = (1, 1, 0, 1)
        next_state, _, done, info = self.env.apply_move(state, Action.DOWN)
        self.assertEqual(next_state[3], 0)
        self.assertTrue(done)
        self.assertTrue(info["energy_exhausted"])
        self.assertTrue(self.env.is_terminal(next_state))

    def test_goal_only_terminal_with_the_key(self):
        goal = self.env.goal
        self.assertTrue(self.env.is_terminal((goal[0], goal[1], 1, 50)))
        self.assertFalse(self.env.is_terminal((goal[0], goal[1], 0, 50)))

    def test_key_is_collected_once(self):
        key = self.env.key
        approach = next(
            (key[0] - dr, key[1] - dc)
            for dr, dc in DELTAS.values()
            if 0 <= key[0] - dr < self.env.rows
            and 0 <= key[1] - dc < self.env.cols
            and self.env.grid[key[0] - dr, key[1] - dc] != Cell.WALL
        )
        move = next(
            action
            for action, (dr, dc) in DELTAS.items()
            if (approach[0] + dr, approach[1] + dc) == key
        )
        state = (approach[0], approach[1], 0, self.env.max_energy)
        next_state, reward, _, info = self.env.apply_move(state, move)
        self.assertEqual(next_state[2], 1)
        self.assertTrue(info["picked_key"])

        # Standing on the key again with has_key already set pays nothing extra.
        again, _, _, info = self.env.apply_move(next_state, move)
        self.assertFalse(info["picked_key"])

    def test_max_steps_truncates_the_episode(self):
        env = build_env()
        env.max_steps = 5
        env.reset(seed=3)
        for _ in range(5):
            _, _, done, info = env.step(Action.DOWN)
        self.assertTrue(done)
        self.assertEqual(env.outcome, "max_steps")
        self.assertTrue(info["truncated"])


class TestRewardModes(unittest.TestCase):
    def test_sparse_mode_uses_only_the_base_rewards(self):
        env = build_env("sparse")
        state = (1, 1, 0, env.max_energy)
        _, reward, _, info = env.apply_move(state, Action.RIGHT)
        self.assertTrue(info["collision"])
        self.assertAlmostEqual(reward, env.rewards["step_cost"])

    def test_shaped_mode_adds_the_collision_penalty(self):
        env = build_env("shaped")
        state = (1, 1, 0, env.max_energy)
        _, reward, _, _ = env.apply_move(state, Action.RIGHT)
        self.assertAlmostEqual(
            reward, env.rewards["step_cost"] + env.rewards["wall_collision"]
        )

    def test_shaping_rewards_progress_and_penalises_regress(self):
        env = build_env("shaped")
        distances = env.dist_to_key
        closer = farther = same = 0

        for r, c in env.passable_cells:
            if distances[r, c] < 0:
                continue
            for action in Action:
                dr, dc = DELTAS[action]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < env.rows and 0 <= nc < env.cols):
                    continue
                if distances[nr, nc] < 0:
                    continue
                shaping = env._shaping(
                    (r, c, 0, 50), (nr, nc, 0, 49), {"picked_key": False}
                )
                if distances[nr, nc] < distances[r, c]:
                    self.assertGreater(shaping, 0)
                    closer += 1
                elif distances[nr, nc] > distances[r, c]:
                    self.assertLess(shaping, 0)
                    farther += 1
                else:
                    self.assertEqual(shaping, 0.0)
                    same += 1

        self.assertGreater(closer, 0)
        self.assertGreater(farther, 0)

    def test_shaping_is_skipped_on_the_key_transition(self):
        env = build_env("shaped")
        shaping = env._shaping((1, 1, 0, 50), (1, 2, 1, 49), {"picked_key": True})
        self.assertEqual(shaping, 0.0)

    def test_sparse_and_shaped_agree_on_the_goal_reward(self):
        for mode in ("sparse", "shaped"):
            env = build_env(mode)
            goal, door = env.goal, env.door
            neighbour = next(
                (goal[0] - dr, goal[1] - dc)
                for dr, dc in DELTAS.values()
                if 0 <= goal[0] - dr < env.rows
                and 0 <= goal[1] - dc < env.cols
                and env.grid[goal[0] - dr, goal[1] - dc] != Cell.WALL
            )
            move = next(
                action
                for action, (dr, dc) in DELTAS.items()
                if (neighbour[0] + dr, neighbour[1] + dc) == goal
            )
            _, reward, done, info = env.apply_move(
                (neighbour[0], neighbour[1], 1, env.max_energy), move
            )
            self.assertTrue(done)
            self.assertTrue(info["success"])
            self.assertGreater(reward, env.rewards["goal_reward"] - 1.0)


class TestPersistence(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        env = build_env()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "map.json"
            env.save_map(path)
            restored = MazeEnv.load_map(path, reward_mode="shaped")
        np.testing.assert_array_equal(env.grid, restored.grid)
        self.assertEqual(env.start, restored.start)
        self.assertEqual(env.key, restored.key)
        self.assertEqual(env.door, restored.door)
        self.assertEqual(env.goal, restored.goal)
        self.assertEqual(env.penalties, restored.penalties)
        self.assertEqual(env.max_energy, restored.max_energy)
        self.assertEqual(env.max_steps, restored.max_steps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
