import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.generator import MazeGenerator
from environments.maze import MazeEnv, Action
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent


def test_maze_env():
    gen = MazeGenerator(rows=5, cols=5, wall_density=0.0, seed=42)
    env = gen.generate_random()
    state = env.reset()
    assert state == (0, 0), "Reset should return start position"
    assert not env.done, "Should not be done at start"
    print("PASS: test_maze_env")


def test_value_iteration():
    gen = MazeGenerator(rows=5, cols=5, wall_density=0.0, seed=42)
    env = gen.generate_random()
    agent = ValueIterationAgent(env)
    iters = agent.train()
    assert iters > 0, "Should converge in > 0 iterations"
    print("PASS: test_value_iteration")


def test_q_learning():
    gen = MazeGenerator(rows=5, cols=5, wall_density=0.0, seed=42)
    env = gen.generate_random()
    agent = QLearningAgent(env)
    results = agent.train(episodes=100)
    assert len(results["rewards"]) == 100
    print("PASS: test_q_learning")


def test_sarsa_lambda():
    gen = MazeGenerator(rows=5, cols=5, wall_density=0.0, seed=42)
    env = gen.generate_random()
    agent = SarsaLambdaAgent(env)
    results = agent.train(episodes=100)
    assert len(results["rewards"]) == 100
    print("PASS: test_sarsa_lambda")


def test_maze_generator():
    gen = MazeGenerator(rows=10, cols=10, seed=42)
    env = gen.generate_random()
    assert env.rows == 10
    assert env.cols == 10
    print("PASS: test_maze_generator")


if __name__ == "__main__":
    test_maze_env()
    test_value_iteration()
    test_q_learning()
    test_sarsa_lambda()
    test_maze_generator()
    print("\nAll tests passed!")
