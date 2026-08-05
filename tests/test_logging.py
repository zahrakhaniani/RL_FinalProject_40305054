"""Tests for the per-episode event counters and the logging built on them.

The CSV logs are only as trustworthy as these counters, so each one is checked
against a hand-driven episode rather than against another counter.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agents.base import EPISODE_LOG_COLUMNS, TrainingLog, log_episode
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from environments.generator import MazeGenerator
from environments.maze import Action, Cell


def build_env(reward_mode: str = "shaped"):
    return MazeGenerator().generate(reward_mode=reward_mode)


def walk(env, action: int, times: int = 1):
    """Force a deterministic move by removing the slip, for exact counting."""
    env.p_intended, env.p_slip = 1.0, 0.0
    for _ in range(times):
        if env.done:
            break
        env.step(action)


class TestEventCounters(unittest.TestCase):
    def setUp(self):
        self.env = build_env()
        self.env.reset(seed=0)

    def test_counters_start_at_zero_and_reset(self):
        walk(self.env, Action.DOWN, 3)
        self.env.reset(seed=1)
        self.assertEqual(sum(self.env.events.values()), 0)
        self.assertEqual(self.env.episode_reward, 0.0)

    def test_wall_collisions_are_counted(self):
        # The start corner is enclosed, so walking up is always a collision.
        walk(self.env, Action.UP, 4)
        self.assertEqual(self.env.events["wall_collisions"], 4)

    def test_penalty_entries_are_counted(self):
        env = build_env()
        env.reset(seed=0)
        approaches = {(-1, 0): Action.DOWN, (1, 0): Action.UP,
                      (0, -1): Action.RIGHT, (0, 1): Action.LEFT}

        for penalty in env.penalties:
            for (dr, dc), action in approaches.items():
                neighbour = (penalty[0] + dr, penalty[1] + dc)
                if not (0 <= neighbour[0] < env.rows and 0 <= neighbour[1] < env.cols):
                    continue
                if env.grid[neighbour] == Cell.WALL:
                    continue
                env.agent_pos = neighbour
                walk(env, action, 1)
                self.assertEqual(env.agent_pos, penalty)
                self.assertEqual(env.events["penalty_entries"], 1)
                return
        self.fail("no penalty cell has an open neighbour to step in from")

    def test_door_blocked_then_crossed(self):
        env = build_env()
        env.reset(seed=0)
        approach = None
        for delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (env.door[0] + delta[0], env.door[1] + delta[1])
            if 0 <= candidate[0] < env.rows and 0 <= candidate[1] < env.cols:
                if env.grid[candidate] not in (Cell.WALL, Cell.GOAL):
                    approach = candidate
                    towards = {(-1, 0): Action.DOWN, (1, 0): Action.UP,
                               (0, -1): Action.RIGHT, (0, 1): Action.LEFT}[delta]
                    break
        self.assertIsNotNone(approach, "the door must have an open approach cell")

        env.agent_pos = approach
        walk(env, towards, 1)
        self.assertEqual(env.events["door_blocked"], 1)
        self.assertEqual(env.events["door_crossings"], 0)
        self.assertEqual(env.agent_pos, approach, "a locked door must not be entered")

        env.has_key = 1
        walk(env, towards, 1)
        self.assertEqual(env.agent_pos, env.door)
        self.assertEqual(env.events["door_crossings"], 1)

    def test_key_pickup_is_counted_once(self):
        env = build_env()
        env.reset(seed=0)
        env.agent_pos = env.key
        env.has_key = 0
        # Step off the key and back on: the pickup happens on the first entry only.
        env.agent_pos = (env.key[0], env.key[1])
        walk(env, Action.UP, 1)
        walk(env, Action.DOWN, 1)
        self.assertLessEqual(env.events["key_pickups"], 1)
        self.assertEqual(env.has_key, 1)

    def test_episode_summary_reports_the_termination_reason(self):
        env = build_env()
        env.reset(seed=0)
        env.energy = 1
        walk(env, Action.DOWN, 1)
        summary = env.episode_summary()
        self.assertEqual(summary["outcome"], "out_of_energy")
        self.assertEqual(summary["energy_exhausted"], 1)
        self.assertEqual(summary["goal_success"], 0)
        self.assertEqual(summary["max_steps_reached"], 0)


class TestTrainingLog(unittest.TestCase):
    def test_rows_have_every_required_column(self):
        env = build_env()
        env.reset(seed=0)
        walk(env, Action.DOWN, 3)
        log = TrainingLog()
        log_episode(log, env, episode=1, epsilon=0.5, alpha=0.1, td_errors=[1.0, -3.0])
        self.assertEqual(tuple(log.rows[0]), EPISODE_LOG_COLUMNS)

    def test_td_error_and_trace_statistics_are_summarised(self):
        env = build_env()
        env.reset(seed=0)
        log = TrainingLog()
        log_episode(
            log, env, episode=1, epsilon=0.1, alpha=0.1,
            td_errors=[1.0, -4.0, 2.0], trace_counts=[1, 7, 3],
        )
        row = log.rows[0]
        self.assertAlmostEqual(row["mean_abs_td_error"], 7 / 3)
        self.assertAlmostEqual(row["max_abs_td_error"], 4.0)
        self.assertAlmostEqual(row["max_active_traces"], 7.0)

    def test_mean_return_is_a_rolling_average(self):
        env = build_env()
        log = TrainingLog()
        for value in (10.0, 20.0, 30.0):
            env.reset(seed=0)
            env.episode_reward = value
            log_episode(log, env, episode=1, epsilon=0.0, alpha=0.1)
        self.assertAlmostEqual(log.rows[0]["mean_return_100"], 10.0)
        self.assertAlmostEqual(log.rows[2]["mean_return_100"], 20.0)


class TestEpsilonSchedules(unittest.TestCase):
    def test_exponential_decay_multiplies_and_stops_at_the_floor(self):
        agent = QLearningAgent(
            build_env(), epsilon_start=1.0, epsilon_end=0.1,
            epsilon_decay=0.5, epsilon_schedule="exponential",
        )
        agent.update_epsilon(1, 100)
        self.assertAlmostEqual(agent.epsilon, 0.5)
        agent.update_epsilon(2, 100)
        self.assertAlmostEqual(agent.epsilon, 0.25)
        for episode in range(3, 40):
            agent.update_epsilon(episode, 100)
        self.assertAlmostEqual(agent.epsilon, 0.1)

    def test_linear_decay_reaches_the_floor_at_the_end_of_the_span(self):
        agent = QLearningAgent(
            build_env(), epsilon_start=1.0, epsilon_end=0.0,
            epsilon_schedule="linear", epsilon_decay_episodes=100,
        )
        agent.update_epsilon(50, 200)
        self.assertAlmostEqual(agent.epsilon, 0.5)
        agent.update_epsilon(100, 200)
        self.assertAlmostEqual(agent.epsilon, 0.0)
        agent.update_epsilon(150, 200)
        self.assertAlmostEqual(agent.epsilon, 0.0, msg="epsilon must not go negative")

    def test_linear_span_defaults_to_most_of_training(self):
        agent = QLearningAgent(
            build_env(), epsilon_start=1.0, epsilon_end=0.0, epsilon_schedule="linear"
        )
        agent.update_epsilon(700, 1000)
        self.assertAlmostEqual(agent.epsilon, 0.0)

    def test_unknown_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            QLearningAgent(build_env(), epsilon_schedule="cosine")


class TestVisitCounts(unittest.TestCase):
    def test_visits_are_recorded_while_training(self):
        for factory in (QLearningAgent, SarsaLambdaAgent):
            with self.subTest(agent=factory.__name__):
                agent = factory(build_env(), seed=1)
                self.assertEqual(agent.visit_counts.sum(), 0)
                log = agent.train(episodes=3, eval_every=0)
                visits = agent.visit_grid()
                self.assertGreater(visits.sum(), 0)
                self.assertGreater(visits[agent.env.start], 0)
                # One visit is recorded per step, plus the initial state per episode.
                self.assertEqual(
                    int(agent.visit_counts.sum()), sum(log.column("steps")) + 3
                )

    def test_visits_never_land_on_a_wall(self):
        agent = QLearningAgent(build_env(), seed=2)
        agent.train(episodes=3, eval_every=0)
        walls = agent.env.grid == Cell.WALL
        self.assertEqual(int(agent.visit_grid()[walls].sum()), 0)


class TestQUpdateSamples(unittest.TestCase):
    def test_samples_record_the_before_and_after_values(self):
        agent = QLearningAgent(build_env(), seed=3, q_log_every=1, q_log_max=5)
        agent.train(episodes=1, eval_every=0)
        self.assertEqual(len(agent.q_update_samples), 5)
        for sample in agent.q_update_samples:
            self.assertAlmostEqual(
                sample["q_new"],
                sample["q_old"] + agent.alpha * sample["td_error"],
                places=5,
            )
            self.assertIn("state", sample)
            self.assertIn("next_state", sample)
            self.assertIn(sample["done"], (0, 1))

    def test_sampling_can_be_switched_off(self):
        agent = QLearningAgent(build_env(), seed=3, q_log_max=0)
        agent.train(episodes=1, eval_every=0)
        self.assertEqual(agent.q_update_samples, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
