import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(filepath):
    with open(filepath) as f:
        return json.load(f)


def compare_agents(results):
    print("\n=== Agent Comparison ===")
    print(f"{'Agent':<20} {'Metric':<20} {'Value':<10}")
    print("-" * 50)

    for agent_name, metrics in results.items():
        for metric, value in metrics.items():
            print(f"{agent_name:<20} {metric:<20} {value:<10.4f}")


def analyze_training(training_data, save_dir=None):
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if "ql" in training_data:
            ql = training_data["ql"]
            axes[0].plot(ql["rewards"], alpha=0.3, label="Q-Learning (raw)")
            window = min(50, len(ql["rewards"]))
            smoothed = [np.mean(ql["rewards"][i:i+window]) for i in range(len(ql["rewards"])-window+1)]
            axes[0].plot(range(window-1, len(ql["rewards"])), smoothed, label="Q-Learning (smoothed)")

        if "sarsa" in training_data:
            sa = training_data["sarsa"]
            axes[0].plot(sa["rewards"], alpha=0.3, label="SARSA(lambda) (raw)")
            window = min(50, len(sa["rewards"]))
            smoothed = [np.mean(sa["rewards"][i:i+window]) for i in range(len(sa["rewards"])-window+1)]
            axes[0].plot(range(window-1, len(sa["rewards"])), smoothed, label="SARSA(lambda) (smoothed)")

        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Total Reward")
        axes[0].set_title("Training Reward")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if "ql" in training_data:
            axes[1].plot(training_data["ql"]["steps"], alpha=0.3, label="Q-Learning")
        if "sarsa" in training_data:
            axes[1].plot(training_data["sarsa"]["steps"], alpha=0.3, label="SARSA(lambda)")

        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Steps")
        axes[1].set_title("Steps per Episode")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
            print(f"Plot saved to {save_dir}/training_curves.png")

        plt.show()
    except ImportError:
        print("matplotlib not available for plotting")


if __name__ == "__main__":
    import numpy as np
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_dir, "results", "raw_data", "experiment_results.json")

    if os.path.exists(results_path):
        results = load_results(results_path)
        compare_agents(results)
    else:
        print("No results found. Run run_experiments.py first.")
