"""Tests for the three agents and the machinery they rely on."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents.base import energy_bin, evaluate_policy
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import EligibilityTraces, SarsaLambdaAgent
from agents.value_iteration import ValueIterationAgent
from environments.generator import MazeGenerator
from environments.maze import N_ACTIONS


def build_env(reward_mode: str = "shaped"):
    return MazeGenerator().generate(reward_mode=reward_mode)


class TestStateAbstraction(unittest.TestCase):
    def test_energy_bins_stay_in_range_and_are_monotone(self):
        max_energy, bins = 225, 8
        previous = -1
        for energy in range(max_energy + 1):
            index = energy_bin(energy, max_energy, bins)
            self.assertGreaterEqual(index, 0)
            self.assertLess(index, bins)
            self.assertGreaterEqual(index, previous)
            previous = index
        self.assertEqual(energy_bin(0, max_energy, bins), 0)
        self.assertEqual(energy_bin(max_energy, max_energy, bins), bins - 1)

    def test_flat_index_matches_numpy_layout(self):
        agent = QLearningAgent(build_env(), seed=0, energy_bins=8)
        rng = np.random.default_rng(0)
        for _ in range(200):
            feature = (
                int(rng.integers(agent.env.rows)),
                int(rng.integers(agent.env.cols)),
                int(rng.integers(2)),
                int(rng.integers(agent.energy_bins)),
                int(rng.integers(N_ACTIONS)),
            )
            self.assertEqual(
                agent.flat_index(feature),
                int(np.ravel_multi_index(feature, agent.q.shape)),
            )


class TestEligibilityTraces(unittest.TestCase):
    """The sparse trace store must behave exactly like a dense trace vector."""

    def run_against_dense(self, threshold: float, atol: float) -> None:
        size, decay, alpha = 400, 0.891, 0.1
        rng = np.random.default_rng(7)
        # Deliberately tiny capacity and prune limit to exercise grow + prune.
        sparse = EligibilityTraces(
            decay=decay, threshold=threshold, replacing=True, capacity=8, prune_at=16
        )
        dense = np.zeros(size)
        q_sparse = np.zeros(size)
        q_dense = np.zeros(size)

        for step in range(3000):
            if step % 250 == 0:
                sparse.clear()
                dense[:] = 0.0
            index = int(rng.integers(size))
            delta = float(rng.normal())

            sparse.bump(index)
            dense[index] = 1.0

            sparse.apply(q_sparse, alpha * delta)
            q_dense += alpha * delta * dense

            sparse.decay_all()
            dense *= decay

        np.testing.assert_allclose(q_sparse, q_dense, atol=atol)

    def test_matches_dense_reference_without_pruning_loss(self):
        self.run_against_dense(threshold=0.0, atol=1e-12)

    def test_matches_dense_reference_with_realistic_threshold(self):
        self.run_against_dense(threshold=1e-4, atol=1e-3)

    def test_traces_never_hold_duplicate_indices(self):
        traces = EligibilityTraces(decay=0.9, capacity=4, prune_at=8)
        for index in [3, 3, 5, 3, 7, 5]:
            traces.bump(index)
            traces.decay_all()
        active = traces.index[: traces.size]
        self.assertEqual(len(set(active.tolist())), traces.size)

    def test_replacing_and_accumulating_traces_differ(self):
        replacing = EligibilityTraces(decay=0.9, replacing=True)
        accumulating = EligibilityTraces(decay=0.9, replacing=False)
        for traces in (replacing, accumulating):
            traces.bump(1)
            traces.bump(1)
        self.assertAlmostEqual(replacing.value[0], 1.0)
        self.assertAlmostEqual(accumulating.value[0], 2.0)


class TestValueIteration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = build_env("shaped")
        cls.agent = ValueIterationAgent(cls.env, gamma=0.99)
        cls.stats = cls.agent.train()

    def test_cached_model_matches_the_environment_model(self):
        """The vectorised cache must reproduce env.transitions() exactly."""
        rng = np.random.default_rng(1)
        exhaustion = self.env.rewards["energy_exhausted"]

        for _ in range(400):
            cell = self.env.passable_cells[int(rng.integers(self.env.n_passable))]
            has_key = int(rng.integers(2))
            # Include energy 1 and 2, where the exhaustion term switches on.
            energy = int(rng.choice([1, 2, 3, self.env.max_energy]))
            state = (cell[0], cell[1], has_key, energy)
            if self.env.is_terminal(state):
                continue

            for action in range(N_ACTIONS):
                expected = sum(
                    probability * reward
                    for probability, _, reward, _ in self.env.transitions(state, action)
                )
                index = self.agent.cell_index[cell]
                cached = 0.0
                for probability, _, _, reward, success in self.agent._model[
                    (has_key, action)
                ]:
                    step_reward = reward[index]
                    if energy == 1 and not success[index]:
                        step_reward += exhaustion
                    cached += probability * step_reward
                self.assertAlmostEqual(cached, expected, places=10)

    def test_bellman_residual_is_numerically_zero(self):
        self.assertLess(self.stats["bellman_residual"], 1e-8)

    def test_optimal_policy_always_solves_the_maze(self):
        metrics = evaluate_policy(self.env, self.agent, episodes=100, seed=4242)
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertGreater(metrics["mean_energy_left"], 0.0)

    def test_optimal_policy_beats_the_random_policy(self):
        class RandomAgent:
            def __init__(self, seed):
                self.rng = np.random.default_rng(seed)

            def get_action(self, state, greedy=True):
                return int(self.rng.integers(N_ACTIONS))

        random_metrics = evaluate_policy(self.env, RandomAgent(0), episodes=30, seed=11)
        optimal_metrics = evaluate_policy(self.env, self.agent, episodes=30, seed=11)
        self.assertGreater(
            optimal_metrics["mean_return"], random_metrics["mean_return"]
        )

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vi.npz"
            self.agent.save(path)
            restored = ValueIterationAgent(self.env, gamma=0.99)
            restored.load(path)
        np.testing.assert_allclose(restored.values, self.agent.values)
        np.testing.assert_array_equal(restored.policy, self.agent.policy)


class TestLearners(unittest.TestCase):
    def test_q_learning_improves_with_training(self):
        env = build_env("shaped")
        agent = QLearningAgent(
            env, alpha=0.15, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
            epsilon_decay=0.99, energy_bins=8, seed=1,
        )
        before = evaluate_policy(env, agent, episodes=20, seed=99)
        agent.train(episodes=400, eval_every=0)
        after = evaluate_policy(env, agent, episodes=20, seed=99)
        self.assertGreater(after["mean_return"], before["mean_return"])

    def test_sarsa_lambda_improves_with_training(self):
        env = build_env("shaped")
        agent = SarsaLambdaAgent(
            env, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
            epsilon_decay=0.99, energy_bins=8, seed=1, lam=0.7,
        )
        before = evaluate_policy(env, agent, episodes=20, seed=99)
        agent.train(episodes=400, eval_every=0)
        after = evaluate_policy(env, agent, episodes=20, seed=99)
        self.assertGreater(after["mean_return"], before["mean_return"])

    def test_training_is_reproducible_for_a_fixed_seed(self):
        first = QLearningAgent(build_env("shaped"), seed=3, energy_bins=8)
        first.train(episodes=120, eval_every=0)
        second = QLearningAgent(build_env("shaped"), seed=3, energy_bins=8)
        second.train(episodes=120, eval_every=0)
        np.testing.assert_allclose(first.q, second.q)

    def test_tabular_save_and_load_round_trip(self):
        env = build_env("shaped")
        agent = QLearningAgent(env, seed=2, energy_bins=8)
        agent.train(episodes=50, eval_every=0)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "q.npz"
            agent.save(path)
            restored = QLearningAgent(env, seed=2, energy_bins=8)
            restored.load(path)
        np.testing.assert_allclose(restored.q, agent.q)

    def test_learners_respect_the_energy_limit(self):
        env = build_env("shaped")
        agent = QLearningAgent(env, seed=1, energy_bins=8)
        agent.train(episodes=20, eval_every=0)
        self.assertLessEqual(env.steps, env.max_steps)
        self.assertGreaterEqual(env.energy, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
