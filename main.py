import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environments.generator import MazeGenerator
from environments.maze import MazeEnv, Action
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent

STUDENT_ID = "40305054"


def print_maze(env):
    env.render()


def run_agent_path(env, agent, label):
    env.reset()
    state = env._get_state()
    steps = 0
    path = [state[:2]]

    while not env.done and steps < 500:
        action = agent.get_action(state)
        state, reward, done, _ = env.step(action)
        path.append(state[:2])
        steps += 1

    print(f"\n=== {label} ===")
    print(f"Path ({steps} steps): {path}")
    print(f"Has key: {env.has_key}, Door open: {env.door_open}")
    print_maze(env)
    return steps, env.agent_pos == env.goal


def demo_value_iteration(env):
    agent = ValueIterationAgent(env.copy(), gamma=0.99)
    iters = agent.train()
    print(f"Converged in {iters} iterations")
    return run_agent_path(env.copy(), agent, "Value Iteration Demo")


def demo_q_learning(env):
    agent = QLearningAgent(env.copy(), alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995)
    results = agent.train(episodes=1000)
    agent.epsilon = 0

    avg_reward = np.mean(results["rewards"][-100:])
    print(f"Training complete. Avg reward (last 100): {avg_reward:.2f}")
    return run_agent_path(env.copy(), agent, "Q-Learning Demo")


def demo_sarsa_lambda(env):
    agent = SarsaLambdaAgent(env.copy(), alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, lam=0.9)
    results = agent.train(episodes=1000)
    agent.epsilon = 0

    avg_reward = np.mean(results["rewards"][-100:])
    print(f"Training complete. Avg reward (last 100): {avg_reward:.2f}")
    return run_agent_path(env.copy(), agent, "SARSA(lambda) Demo")


def main():
    print("=" * 50)
    print("  Reinforcement Learning Maze Solver")
    print("=" * 50)

    gen = MazeGenerator(student_id=STUDENT_ID)
    gen.print_info()
    print()

    env = gen.generate()
    print("Maze (shared by all algorithm runs):")
    print_maze(env)
    print()

    print("Options:")
    print("1. Value Iteration (model-based)")
    print("2. Q-Learning (model-free)")
    print("3. SARSA(lambda) (model-free)")
    print("4. Run all demos")
    print("5. Generate and save maze only")

    choice = input("\nSelect option (1-5): ").strip()

    if choice == "1":
        demo_value_iteration(env)
    elif choice == "2":
        demo_q_learning(env)
    elif choice == "3":
        demo_sarsa_lambda(env)
    elif choice == "4":
        vi_steps, vi_ok = demo_value_iteration(env)
        ql_steps, ql_ok = demo_q_learning(env)
        sa_steps, sa_ok = demo_sarsa_lambda(env)
        print("\n=== Comparison Summary (same map) ===")
        print(f"Value Iteration: {vi_steps} steps, success={vi_ok}")
        print(f"Q-Learning:      {ql_steps} steps, success={ql_ok}")
        print(f"SARSA(lambda):   {sa_steps} steps, success={sa_ok}")
    elif choice == "5":
        print("Generated maze:")
        print_maze(env)
        map_path = os.path.join("results", "raw_data", "maze_map.json")
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        env.save_map(map_path)
        print(f"Map saved to {map_path}")
    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
