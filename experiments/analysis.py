"""Turn the saved raw results into figures and comparison tables.

Reads everything from ``results/raw_data/<algorithm>/`` and writes figures to
``results/figures/<algorithm>/``, so it can be re-run at any time without
retraining anything.

    python experiments/analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
from agents.base import energy_bin
from environments.maze import Cell
from experiments import common

MODE_COLORS = {"sparse": "#d1495b", "shaped": "#00798c"}
ALGO_COLORS = {
    "value_iteration": "#2e4057",
    "q_learning": "#00798c",
    "sarsa_lambda": "#edae49",
    "transfer": "#8f2d56",
}
CELL_COLORS = {
    Cell.PATH: "#f7f7f5",
    Cell.WALL: "#2f3640",
    Cell.START: "#7bc043",
    Cell.KEY: "#f2b134",
    Cell.DOOR: "#b5651d",
    Cell.GOAL: "#0392cf",
    Cell.PENALTY: "#ee4035",
}
ARROWS = {0: (0, -0.32), 1: (0.32, 0), 2: (0, 0.32), 3: (-0.32, 0)}


# ------------------------------------------------------------------- utilities


def moving_average(values: Sequence[float], window: int = 50) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    window = max(1, min(int(window), values.size))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_records(algorithm: str, filename: Optional[str] = None) -> List[dict]:
    path = common.raw_path(algorithm, filename or f"{algorithm}_results.json")
    if not path.exists():
        print(f"  (skipping {algorithm}: {paths.rel(path)} not found)")
        return []
    return common.load_json(path)["records"]


def group_by(records: Sequence[dict], field: str) -> Dict[object, List[dict]]:
    grouped: Dict[object, List[dict]] = {}
    for record in records:
        grouped.setdefault(record.get(field), []).append(record)
    return grouped


def stack_curves(records: Sequence[dict], key: str) -> Optional[np.ndarray]:
    """Stack a per-episode or per-eval curve across seeds, trimmed to equal length."""
    curves = [np.asarray(record["log"][key], dtype=float) for record in records]
    curves = [curve for curve in curves if curve.size]
    if not curves:
        return None
    length = min(curve.size for curve in curves)
    return np.vstack([curve[:length] for curve in curves])


def band(axis, x, stack: np.ndarray, color: str, label: str) -> None:
    mean = stack.mean(axis=0)
    axis.plot(x[: mean.size], mean, color=color, label=label, linewidth=1.8)
    if stack.shape[0] > 1:
        spread = stack.std(axis=0)
        axis.fill_between(
            x[: mean.size], mean - spread, mean + spread, color=color, alpha=0.18
        )


def save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
    print(f"  wrote {paths.rel(path)}")


# ---------------------------------------------------------------- maze figures


def plot_maze(config: dict, trajectory: Optional[Sequence[Sequence[int]]] = None,
              title: str = "Maze layout", filename: str = "maze_layout.png") -> None:
    env = common.build_env(config, reward_mode="shaped")
    figure, axis = plt.subplots(figsize=(6.4, 6.4))
    canvas = np.zeros((env.rows, env.cols, 3))
    for r in range(env.rows):
        for c in range(env.cols):
            colour = CELL_COLORS[Cell(env.grid[r, c])]
            canvas[r, c] = matplotlib.colors.to_rgb(colour)
    axis.imshow(canvas, interpolation="nearest")

    for label, cell in (("S", env.start), ("K", env.key), ("D", env.door), ("G", env.goal)):
        axis.text(
            cell[1], cell[0], label, ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
        )

    if trajectory:
        rows = [state[0] for state in trajectory]
        cols = [state[1] for state in trajectory]
        axis.plot(cols, rows, color="#ffffff", linewidth=1.2, alpha=0.55)
        axis.plot(cols, rows, color="#111111", linewidth=0.7, linestyle=":")

    axis.set_title(f"{title}\n{env.rows}x{env.cols}, walls "
                   f"{float(np.mean(env.grid == Cell.WALL)):.1%}, "
                   f"{len(env.penalties)} penalty cells")
    axis.set_xticks(range(env.cols))
    axis.set_yticks(range(env.rows))
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.grid(color="#c8c8c8", linewidth=0.4)
    axis.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    handles = [
        mpatches.Patch(color=CELL_COLORS[Cell.START], label="start"),
        mpatches.Patch(color=CELL_COLORS[Cell.KEY], label="key"),
        mpatches.Patch(color=CELL_COLORS[Cell.DOOR], label="locked door"),
        mpatches.Patch(color=CELL_COLORS[Cell.GOAL], label="goal (in vault)"),
        mpatches.Patch(color=CELL_COLORS[Cell.PENALTY], label="penalty cell"),
        mpatches.Patch(color=CELL_COLORS[Cell.WALL], label="wall"),
    ]
    axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                ncol=3, frameon=False, fontsize=8)
    save(figure, paths.FIGURES / filename)


# -------------------------------------------------------- value iteration plots


def plot_value_iteration(config: dict, records: Sequence[dict]) -> None:
    if not records:
        return

    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for record in records:
        deltas = record["train_stats"]["max_delta_per_sweep"]
        axis.semilogy(
            range(1, len(deltas) + 1),
            np.maximum(deltas, 1e-16),
            color=MODE_COLORS[record["reward_mode"]],
            label=f"{record['reward_mode']} (residual "
                  f"{record['train_stats']['bellman_residual']:.1e})",
        )
    axis.set_xlabel("energy level solved (backward induction sweep)")
    axis.set_ylabel("max change in V vs. previous energy level")
    axis.set_title("Value Iteration convergence")
    axis.grid(alpha=0.3)
    axis.legend()
    save(figure, common.figure_path("value_iteration", "convergence.png"))

    # Optimal value landscape before and after picking up the key.
    env = common.build_env(config, reward_mode="shaped")
    agent = common.make_planner(env, config)
    agent.train()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, has_key in zip(axes, (0, 1)):
        grid = agent.value_grid(has_key=has_key, energy=env.max_energy)
        masked = np.ma.masked_invalid(grid)
        image = axis.imshow(masked, cmap="viridis")
        figure.colorbar(image, ax=axis, fraction=0.046)
        axis.set_title(f"V*(s) with has_key={has_key}, full battery")
        axis.set_xticks([])
        axis.set_yticks([])
        for label, cell in (("K", env.key), ("D", env.door), ("G", env.goal)):
            axis.text(cell[1], cell[0], label, ha="center", va="center",
                      color="white", fontsize=9, fontweight="bold")
    save(figure, common.figure_path("value_iteration", "value_landscape.png"))

    # Greedy policy arrows.
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for axis, has_key in zip(axes, (0, 1)):
        policy = agent.greedy_policy_grid(has_key=has_key, energy=env.max_energy)
        axis.imshow(env.grid == Cell.WALL, cmap="Greys", interpolation="nearest")
        for r, c in env.passable_cells:
            dx, dy = ARROWS[int(policy[r, c])]
            axis.arrow(c, r, dx, dy, head_width=0.18, head_length=0.18,
                       color="#00798c", linewidth=0.6)
        axis.set_title(f"Optimal policy, has_key={has_key}")
        axis.set_xticks([])
        axis.set_yticks([])
    save(figure, common.figure_path("value_iteration", "policy_arrows.png"))


# ----------------------------------------------------------------learner plots


def plot_learner(algorithm: str, records: Sequence[dict]) -> None:
    if not records:
        return
    by_mode = group_by(records, "reward_mode")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for reward_mode, group in sorted(by_mode.items()):
        colour = MODE_COLORS.get(reward_mode, "#555555")

        rewards = stack_curves(group, "reward")
        if rewards is not None:
            smoothed = np.vstack([moving_average(row, 50) for row in rewards])
            band(axes[0], np.arange(1, smoothed.shape[1] + 1), smoothed, colour, reward_mode)

        success = stack_curves(group, "eval_success_rate")
        episodes = np.asarray(group[0]["log"]["eval_episode"], dtype=float)
        if success is not None:
            band(axes[1], episodes, success, colour, reward_mode)

        returns = stack_curves(group, "eval_return")
        if returns is not None:
            band(axes[2], episodes, returns, colour, reward_mode)

    axes[0].set_title("training return (50-episode moving average)")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("return")
    axes[1].set_title("greedy success rate")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_title("greedy return")
    axes[2].set_xlabel("episode")
    axes[2].set_ylabel("return")
    for axis in axes:
        axis.grid(alpha=0.3)
        axis.legend(title="reward mode")
    figure.suptitle(f"{algorithm} learning curves (mean +/- std over seeds)")
    save(figure, common.figure_path(algorithm, "learning_curves.png"))


def plot_lambda_sweep(sweeps: Dict[str, Sequence[dict]]) -> None:
    """One line per reward mode, because lambda matters far more when rewards are sparse."""
    sweeps = {mode: records for mode, records in sweeps.items() if records}
    if not sweeps:
        return

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for reward_mode, records in sorted(sweeps.items()):
        frame = pd.DataFrame(
            {
                "lam": [record["lam"] for record in records],
                "success_rate": [r["final_eval"]["success_rate"] for r in records],
                "mean_return": [r["final_eval"]["mean_return"] for r in records],
            }
        )
        grouped = frame.groupby("lam").agg(["mean", "std"])
        lams = grouped.index.to_numpy()
        colour = MODE_COLORS.get(reward_mode, "#777777")

        for axis, column in ((axes[0], "success_rate"), (axes[1], "mean_return")):
            mean = grouped[(column, "mean")].to_numpy()
            spread = np.nan_to_num(grouped[(column, "std")].to_numpy())
            axis.errorbar(
                lams, mean, yerr=spread, marker="o", capsize=4,
                color=colour, label=reward_mode,
            )

    axes[0].set_ylabel("greedy success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("greedy mean return")
    for axis in axes:
        axis.set_xlabel("lambda")
        axis.grid(alpha=0.3)
        axis.legend(title="reward mode")
    figure.suptitle("SARSA(lambda): effect of the trace decay parameter")
    save(figure, common.figure_path("sarsa_lambda", "lambda_sweep.png"))


# ------------------------------------------------------------- transfer plots


def plot_transfer(records: Sequence[dict]) -> None:
    if not records:
        return
    by_strategy = group_by(records, "strategy")
    palette = {"scratch": "#8d99ae", "warm_start": "#00798c", "policy_reuse": "#8f2d56"}

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    strategies = sorted(by_strategy)
    for strategy in strategies:
        group = by_strategy[strategy]
        colour = palette.get(strategy, "#555555")
        episodes = np.asarray(group[0]["log"]["eval_episode"], dtype=float)
        success = stack_curves(group, "eval_success_rate")
        if success is not None:
            band(axes[0], episodes, success, colour, strategy)
        returns = stack_curves(group, "eval_return")
        if returns is not None:
            band(axes[1], episodes, returns, colour, strategy)

    axes[0].set_title("greedy success rate on the new maze")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_title("greedy return on the new maze")
    axes[1].set_ylabel("return")
    for axis in axes[:2]:
        axis.set_xlabel("episode on the target maze")
        axis.grid(alpha=0.3)
        axis.legend(title="strategy")

    # Zero-shot (before any target training) next to the converged result.
    positions = np.arange(len(strategies))
    width = 0.36
    for offset, (key, label) in enumerate(
        ((("transfer_metrics", "zero_shot_success"), "zero-shot"),
         (("final_eval", "success_rate"), "after training"))
    ):
        values = [
            float(np.mean([record[key[0]][key[1]] for record in by_strategy[s]]))
            for s in strategies
        ]
        axes[2].bar(
            positions + offset * width - width / 2,
            values,
            width,
            label=label,
            color="#8d99ae" if offset == 0 else "#00798c",
        )
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels([s.replace("_", "\n") for s in strategies], fontsize=9)
    axes[2].set_ylabel("success rate")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("zero-shot vs. converged performance")
    axes[2].grid(alpha=0.3, axis="y")
    axes[2].legend()
    figure.suptitle("Transfer learning: warm start and policy reuse vs. learning from scratch")
    save(figure, common.figure_path("transfer", "transfer_curves.png"))


# --------------------------------------------------------------- comparison


def build_comparison(config: dict) -> pd.DataFrame:
    frames = []
    for algorithm in ("value_iteration", "q_learning", "sarsa_lambda"):
        path = common.raw_path(algorithm, f"{algorithm}_summary.csv")
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby(["algorithm", "reward_mode"])
        .agg(
            runs=("success_rate", "size"),
            success_rate=("success_rate", "mean"),
            success_rate_std=("success_rate", "std"),
            mean_return=("mean_return", "mean"),
            return_std=("mean_return", "std"),
            mean_steps=("mean_steps", "mean"),
            mean_energy_left=("mean_energy_left", "mean"),
            train_seconds=("train_seconds", "mean"),
        )
        .reset_index()
        .fillna(0.0)
    )

    grouped.to_csv(paths.RAW_DATA / "comparison_summary.csv", index=False)
    write_markdown_table(grouped, paths.RAW_DATA / "comparison_summary.md")
    print(f"  wrote {paths.rel(paths.RAW_DATA / 'comparison_summary.csv')}")
    return grouped


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    """Markdown table without pulling in an extra dependency for it."""

    def cell(value) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    header = f"| {' | '.join(frame.columns)} |"
    divider = f"| {' | '.join('---' for _ in frame.columns)} |"
    rows = [
        f"| {' | '.join(cell(value) for value in record)} |"
        for record in frame.itertuples(index=False, name=None)
    ]
    path.write_text("\n".join([header, divider, *rows]) + "\n", encoding="utf-8")
    print(f"  wrote {paths.rel(path)}")


def plot_comparison(comparison: pd.DataFrame) -> None:
    if comparison.empty:
        return
    metrics = (
        ("success_rate", "greedy success rate", "success_rate_std"),
        ("mean_return", "greedy mean return", "return_std"),
        ("mean_steps", "steps per episode", None),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    algorithms = sorted(comparison["algorithm"].unique())
    modes = sorted(comparison["reward_mode"].unique())
    width = 0.36

    for axis, (column, label, error_column) in zip(axes, metrics):
        positions = np.arange(len(algorithms))
        for offset, mode in enumerate(modes):
            subset = comparison[comparison["reward_mode"] == mode].set_index("algorithm")
            values = [subset.loc[a, column] if a in subset.index else np.nan for a in algorithms]
            errors = None
            if error_column:
                errors = [
                    subset.loc[a, error_column] if a in subset.index else 0.0
                    for a in algorithms
                ]
            axis.bar(
                positions + offset * width - width / 2,
                values,
                width,
                yerr=errors,
                capsize=3,
                label=mode,
                color=MODE_COLORS.get(mode, "#777777"),
            )
        axis.set_xticks(positions)
        axis.set_xticklabels([a.replace("_", "\n") for a in algorithms], fontsize=9)
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(alpha=0.3, axis="y")
        axis.legend(title="reward mode")
    axes[0].set_ylim(0, 1.05)
    figure.suptitle("Algorithm comparison on the shared maze")
    save(figure, common.figure_path("comparison", "algorithm_comparison.png"))


def plot_combined_curves() -> None:
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    plotted = False
    for algorithm in ("q_learning", "sarsa_lambda"):
        records = [
            record
            for record in load_records(algorithm)
            if record["reward_mode"] == "shaped"
        ]
        if not records:
            continue
        success = stack_curves(records, "eval_success_rate")
        episodes = np.asarray(records[0]["log"]["eval_episode"], dtype=float)
        if success is not None:
            band(axis, episodes, success, ALGO_COLORS[algorithm], algorithm)
            plotted = True

    vi = [r for r in load_records("value_iteration") if r["reward_mode"] == "shaped"]
    if vi:
        axis.axhline(
            vi[0]["final_eval"]["success_rate"],
            color=ALGO_COLORS["value_iteration"],
            linestyle="--",
            label="value iteration (optimal)",
        )
        plotted = True

    if not plotted:
        plt.close(figure)
        return
    axis.set_xlabel("episode")
    axis.set_ylabel("greedy success rate")
    axis.set_ylim(-0.05, 1.05)
    axis.set_title("Model-free learners vs. the model-based optimum (shaped rewards)")
    axis.grid(alpha=0.3)
    axis.legend()
    save(figure, common.figure_path("comparison", "learning_vs_optimal.png"))


# --------------------------------------------------------------------- driver


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build figures and comparison tables")
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)

    vi_records = load_records("value_iteration")
    trajectory = None
    for record in vi_records:
        if record["reward_mode"] == "shaped":
            trajectory = record["example_trajectory"]["states"]
    plot_maze(config, trajectory=trajectory,
              title="Shared maze with the optimal policy's trajectory")

    plot_value_iteration(config, vi_records)
    plot_learner("q_learning", load_records("q_learning"))
    plot_learner("sarsa_lambda", load_records("sarsa_lambda"))
    plot_lambda_sweep(
        {
            reward_mode: load_records(
                "sarsa_lambda", f"sarsa_lambda_lambda_sweep_{reward_mode}.json"
            )
            for reward_mode in config["reward_modes"]
        }
    )
    plot_transfer(load_records("transfer"))

    comparison = build_comparison(config)
    plot_comparison(comparison)
    plot_combined_curves()

    if not comparison.empty:
        print()
        print("Algorithm comparison")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
