import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.generator import MazeGenerator
from environments.maze import MazeEnv, Action
from agents.value_iteration import ValueIterationAgent
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent


def run_experiment(config):
    gen = MazeGenerator(
        rows=config["rows"],
        cols=config["cols"],
        wall_density=config["wall_density"],
        seed=config.get("seed", 42),
    )

    results = {}

    # Value Iteration
    env = gen.generate_random()
    vi_agent = ValueIterationAgent(env, gamma=config["gamma"])
    vi_iters = vi_agent.train()
    env.reset()
    steps = 0
    state = env.agent_pos
    while not env.done and steps < 200:
        action = vi_agent.get_action(state)
        state, _, done, _ = env.step(action)
        steps += 1
    results["value_iteration"] = {"iterations": vi_iters, "steps": steps}

    # Q-Learning
    env_ql = gen.generate_random()
    ql_agent = QLearningAgent(
        env_ql, alpha=config["alpha"], gamma=config["gamma"],
        epsilon=config["epsilon"], epsilon_decay=config["epsilon_decay"],
    )
    ql_results = ql_agent.train(episodes=config["episodes"])
    results["q_learning"] = {
        "avg_reward": float(np.mean(ql_results["rewards"][-50:])),
        "avg_steps": float(np.mean(ql_results["steps"][-50:])),
        "final_epsilon": ql_results["epsilon"][-1],
    }

    # SARSA(lambda)
    env_sa = gen.generate_random()
    sa_agent = SarsaLambdaAgent(
        env_sa, alpha=config["alpha"], gamma=config["gamma"],
        epsilon=config["epsilon"], epsilon_decay=config["epsilon_decay"],
        lam=config["lambda"],
    )
    sa_results = sa_agent.train(episodes=config["episodes"])
    results["sarsa_lambda"] = {
        "avg_reward": float(np.mean(sa_results["rewards"][-50:])),
        "avg_steps": float(np.mean(sa_results["steps"][-50:])),
        "final_epsilon": sa_results["epsilon"][-1],
    }

    return results, {"ql": ql_results, "sarsa": sa_results}


def save_results(results, raw_data_dir):
    os.makedirs(raw_data_dir, exist_ok=True)
    with open(os.path.join(raw_data_dir, "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {raw_data_dir}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, "experiments", "configs", "default.json")

    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {
            "rows": 10, "cols": 10, "wall_density": 0.25, "seed": 42,
            "gamma": 0.99, "alpha": 0.1, "epsilon": 1.0,
            "epsilon_decay": 0.995, "lambda": 0.9, "episodes": 500,
        }

    print("Running experiment with config:", json.dumps(config, indent=2))
    results, training_data = run_experiment(config)
    print("\nResults:")
    print(json.dumps(results, indent=2))

    raw_dir = os.path.join(base_dir, "results", "raw_data")
    save_results(results, raw_dir)
