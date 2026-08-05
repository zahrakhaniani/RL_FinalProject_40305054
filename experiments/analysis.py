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
SCENARIO_COLORS = {
    "scratch": "#8d99ae",
    "full": "#8f2d56",
    "scaled_0.25": "#f4a261",
    "scaled_0.5": "#e76f51",
    "scaled_0.75": "#b5651d",
    "selective": "#00798c",
}


# ------------------------------------------------------------------- utilities


def moving_average(values: Sequence[float], window: int = 50) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    window = max(1, min(int(window), values.size))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_payload(algorithm: str, filename: str) -> Optional[dict]:
    path = common.raw_path(algorithm, filename)
    if not path.exists():
        print(f"  (skipping {paths.rel(path)}: not found)")
        return None
    return common.load_json(path)


def load_records(algorithm: str, filename: Optional[str] = None) -> List[dict]:
    payload = load_payload(algorithm, filename or f"{algorithm}_results.json")
    return payload["records"] if payload else []


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


def bare_axes(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])


def mark_landmarks(axis, env, colour: str = "white") -> None:
    for label, cell in (("S", env.start), ("K", env.key), ("D", env.door), ("G", env.goal)):
        axis.text(
            cell[1], cell[0], label, ha="center", va="center",
            fontsize=8, fontweight="bold", color=colour,
        )


def wall_overlay(axis, env) -> None:
    """Grey out the walls so a heat map reads as a maze."""
    walls = np.ma.masked_where(env.grid != Cell.WALL, np.ones_like(env.grid, dtype=float))
    axis.imshow(walls, cmap=matplotlib.colors.ListedColormap(["#2f3640"]), vmin=0, vmax=1)


def heatmap(figure, axis, env, grid: np.ndarray, title: str, cmap: str = "viridis") -> None:
    masked = np.ma.masked_where(env.grid == Cell.WALL, grid)
    image = axis.imshow(masked, cmap=cmap, interpolation="nearest")
    wall_overlay(axis, env)
    figure.colorbar(image, ax=axis, fraction=0.046)
    axis.set_title(title, fontsize=10)
    bare_axes(axis)
    mark_landmarks(axis, env)


def policy_panel(axis, env, policy: np.ndarray, title: str, colour: str = "#00798c") -> None:
    axis.imshow(env.grid == Cell.WALL, cmap="Greys", interpolation="nearest")
    for row, col in env.passable_cells:
        dx, dy = ARROWS[int(policy[row, col])]
        axis.arrow(col, row, dx, dy, head_width=0.18, head_length=0.18,
                   color=colour, linewidth=0.6)
    axis.set_title(title, fontsize=10)
    bare_axes(axis)
    mark_landmarks(axis, env, colour="#d1495b")


# ---------------------------------------------------------------- maze figures


def plot_maze(config: dict, trajectory=None, title: str = "Maze layout",
              filename: str = "maze_layout.png", env=None) -> None:
    env = env or common.build_env(config, reward_mode="shaped")
    figure, axis = plt.subplots(figsize=(6.4, 6.4))
    canvas = np.zeros((env.rows, env.cols, 3))
    for r in range(env.rows):
        for c in range(env.cols):
            canvas[r, c] = matplotlib.colors.to_rgb(CELL_COLORS[Cell(env.grid[r, c])])
    axis.imshow(canvas, interpolation="nearest")
    mark_landmarks(axis, env)

    if trajectory:
        rows = [state[0] for state in trajectory]
        cols = [state[1] for state in trajectory]
        axis.plot(cols, rows, color="#ffffff", linewidth=1.2, alpha=0.55)
        axis.plot(cols, rows, color="#111111", linewidth=0.7, linestyle=":")

    axis.set_title(
        f"{title}\n{env.rows}x{env.cols}, walls "
        f"{float(np.mean(env.grid == Cell.WALL)):.1%}, "
        f"{len(env.penalties)} penalty cells"
    )
    bare_axes(axis)
    axis.set_xticks(np.arange(-0.5, env.cols, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, env.rows, 1), minor=True)
    axis.grid(which="minor", color="#c8c8c8", linewidth=0.4)
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

    agent, env = common.load_trained_planner(config, "shaped")
    if agent is None:
        return

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, has_key in zip(axes, (0, 1)):
        heatmap(figure, axis, env, agent.value_grid(has_key, env.max_energy),
                f"V*(s), has_key={has_key}, full battery")
    figure.suptitle("Value Iteration: optimal value heat map")
    save(figure, common.figure_path("value_iteration", "value_heatmap.png"))

    figure, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for axis, has_key in zip(axes, (0, 1)):
        policy_panel(axis, env, agent.greedy_policy_grid(has_key, env.max_energy),
                     f"optimal policy, has_key={has_key}")
    figure.suptitle("Value Iteration: final policy map")
    save(figure, common.figure_path("value_iteration", "policy_map.png"))


def plot_gamma_sweep(payload: Optional[dict]) -> None:
    """What the discount factor costs in this maze."""
    if not payload:
        return
    frame = pd.DataFrame(
        [
            {
                "gamma": record["gamma"],
                "reward_mode": record["reward_mode"],
                "success_rate": record["final_eval"]["success_rate"],
                "mean_return": record["final_eval"]["mean_return"],
                "mean_steps": record["final_eval"]["mean_steps"],
                "train_seconds": record["train_seconds"],
            }
            for record in payload["records"]
        ]
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for reward_mode, group in frame.groupby("reward_mode"):
        colour = MODE_COLORS.get(reward_mode, "#777777")
        group = group.sort_values("gamma")
        for axis, column in zip(axes, ("success_rate", "mean_return", "mean_steps")):
            axis.plot(group["gamma"], group[column], marker="o", color=colour, label=reward_mode)

    for axis, label in zip(
        axes, ("greedy success rate", "greedy mean return", "steps per episode")
    ):
        axis.set_xlabel("gamma")
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(alpha=0.3)
        axis.legend(title="reward mode")
    axes[0].set_ylim(-0.05, 1.05)
    figure.suptitle("Value Iteration: effect of the discount factor")
    save(figure, common.figure_path("value_iteration", "gamma_sweep.png"))


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

        episodes = np.asarray(group[0]["log"]["eval_episode"], dtype=float)
        success = stack_curves(group, "eval_success_rate")
        if success is not None:
            band(axes[1], episodes, success, colour, reward_mode)
        returns = stack_curves(group, "eval_return")
        if returns is not None:
            band(axes[2], episodes, returns, colour, reward_mode)

    axes[0].set_title("training return (50-episode moving average)")
    axes[0].set_ylabel("return")
    axes[1].set_title("greedy success rate")
    axes[1].set_ylabel("success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_title("greedy return")
    axes[2].set_ylabel("return")
    for axis in axes:
        axis.set_xlabel("episode")
        axis.grid(alpha=0.3)
        axis.legend(title="reward mode")
    figure.suptitle(f"{algorithm} learning curves (mean +/- std over seeds)")
    save(figure, common.figure_path(algorithm, "learning_curves.png"))


def plot_learner_maps(
    config: dict, algorithm: str, records: Sequence[dict], reward_mode: str = "shaped"
) -> None:
    """Value heat map, final policy map and visit counts for one learner."""
    group = [record for record in records if record["reward_mode"] == reward_mode]
    if not group:
        return
    seed = group[0]["seed"]
    agent, env = common.load_trained_learner(config, algorithm, reward_mode, seed)
    if agent is None:
        print(f"  (skipping {algorithm} maps: no saved model for seed {seed})")
        return

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, has_key in zip(axes, (0, 1)):
        heatmap(figure, axis, env, agent.value_grid(has_key, env.max_energy),
                f"max_a Q(s,a), has_key={has_key}, full battery")
    figure.suptitle(f"{algorithm}: learned value heat map ({reward_mode} rewards, seed {seed})")
    save(figure, common.figure_path(algorithm, "value_heatmap.png"))

    figure, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for axis, has_key in zip(axes, (0, 1)):
        policy_panel(axis, env, agent.greedy_policy_grid(has_key, env.max_energy),
                     f"greedy policy, has_key={has_key}",
                     colour=ALGO_COLORS.get(algorithm, "#00798c"))
    figure.suptitle(f"{algorithm}: final policy map ({reward_mode} rewards, seed {seed})")
    save(figure, common.figure_path(algorithm, "policy_map.png"))

    # Visit counts summed over seeds: where the exploration actually went.
    visits = np.sum(
        [np.asarray(record["visit_counts"], dtype=float) for record in group], axis=0
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    heatmap(figure, axes[0], env, visits, "state visits (all seeds)", cmap="magma")
    heatmap(figure, axes[1], env, np.log10(visits + 1.0),
            "log10(1 + visits)", cmap="magma")
    figure.suptitle(f"{algorithm}: visit-count heat map ({reward_mode} rewards)")
    save(figure, common.figure_path(algorithm, "visit_counts.png"))


def plot_epsilon_study(algorithm: str, payloads: Dict[str, Optional[dict]]) -> None:
    """Linear vs exponential epsilon decay."""
    payloads = {mode: payload for mode, payload in payloads.items() if payload}
    if not payloads:
        return
    palette = {"exponential": "#00798c", "linear": "#d1495b"}

    figure, axes = plt.subplots(len(payloads), 3, figsize=(15, 4.2 * len(payloads)),
                                squeeze=False)
    for row, (reward_mode, payload) in enumerate(sorted(payloads.items())):
        by_schedule = group_by(payload["records"], "epsilon_schedule")
        for schedule, group in sorted(by_schedule.items()):
            colour = palette.get(schedule, "#777777")
            epsilon = stack_curves(group, "epsilon")
            if epsilon is not None:
                axes[row][0].plot(
                    np.arange(1, epsilon.shape[1] + 1), epsilon.mean(axis=0),
                    color=colour, label=schedule,
                )
            episodes = np.asarray(group[0]["log"]["eval_episode"], dtype=float)
            success = stack_curves(group, "eval_success_rate")
            if success is not None:
                band(axes[row][1], episodes, success, colour, schedule)
            rewards = stack_curves(group, "reward")
            if rewards is not None:
                smoothed = np.vstack([moving_average(curve, 50) for curve in rewards])
                band(axes[row][2], np.arange(1, smoothed.shape[1] + 1),
                     smoothed, colour, schedule)

        axes[row][0].set_title(f"epsilon schedule ({reward_mode})")
        axes[row][0].set_ylabel("epsilon")
        axes[row][1].set_title(f"greedy success rate ({reward_mode})")
        axes[row][1].set_ylabel("success rate")
        axes[row][1].set_ylim(-0.05, 1.05)
        axes[row][2].set_title(f"training return ({reward_mode})")
        axes[row][2].set_ylabel("return")
        for axis in axes[row]:
            axis.set_xlabel("episode")
            axis.grid(alpha=0.3)
            axis.legend(title="decay")

    figure.suptitle(f"{algorithm}: linear vs exponential epsilon decay")
    save(figure, common.figure_path(algorithm, "epsilon_schedules.png"))


def plot_lambda_sweep(sweeps: Dict[str, Sequence[dict]]) -> None:
    """One line per reward mode, because lambda matters far more when rewards are sparse."""
    sweeps = {mode: records for mode, records in sweeps.items() if records}
    if not sweeps:
        return

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for reward_mode, records in sorted(sweeps.items()):
        frame = pd.DataFrame(
            {
                "lam": [record["lam"] for record in records],
                "success_rate": [r["final_eval"]["success_rate"] for r in records],
                "mean_return": [r["final_eval"]["mean_return"] for r in records],
                "late_return_std": [
                    r.get("stability", {}).get("late_return_std", np.nan) for r in records
                ],
            }
        )
        grouped = frame.groupby("lam").agg(["mean", "std"])
        lams = grouped.index.to_numpy()
        colour = MODE_COLORS.get(reward_mode, "#777777")

        for axis, column in zip(axes, ("success_rate", "mean_return", "late_return_std")):
            mean = grouped[(column, "mean")].to_numpy()
            spread = np.nan_to_num(grouped[(column, "std")].to_numpy())
            axis.errorbar(lams, mean, yerr=spread, marker="o", capsize=4,
                          color=colour, label=reward_mode)

    axes[0].set_ylabel("greedy success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_title("final performance")
    axes[1].set_ylabel("greedy mean return")
    axes[1].set_title("final return")
    axes[2].set_ylabel("std of return, last 200 episodes")
    axes[2].set_title("stability (lower is steadier)")
    for axis in axes:
        axis.set_xlabel("lambda")
        axis.grid(alpha=0.3)
        axis.legend(title="reward mode")
    figure.suptitle("SARSA(lambda): effect of the trace decay parameter")
    save(figure, common.figure_path("sarsa_lambda", "lambda_sweep.png"))


# ---------------------------------------------------------------- comparison


def plot_policy_difference(config: dict, payload: Optional[dict]) -> None:
    """Where the learners disagree with the proven-optimal action."""
    if not payload:
        return
    map_file = paths.ROOT / payload["difference_maps_file"]
    if not map_file.exists():
        return

    frame = pd.DataFrame(payload["rows"])
    with np.load(map_file) as data:
        keys = [key for key in data.files if not key.endswith("__visits")]
        if not keys:
            return
        panels = []
        for key in sorted(keys):
            reward_mode, algorithm = key.split("__")
            panels.append((reward_mode, algorithm, data[key]))

        figure, axes = plt.subplots(len(panels), 2, figsize=(10.5, 4.6 * len(panels)),
                                   squeeze=False)
        colours = matplotlib.colors.ListedColormap(["#d1495b", "#7bc043"])
        for row, (reward_mode, algorithm, grid) in enumerate(panels):
            subset = frame[
                (frame["algorithm"] == algorithm) & (frame["reward_mode"] == reward_mode)
            ]
            agreement = subset["policy_agreement"].mean()
            weighted = subset["weighted_policy_agreement"].mean()
            env = common.build_env(config, reward_mode=reward_mode)
            for column, has_key in enumerate((0, 1)):
                axis = axes[row][column]
                axis.imshow(np.ma.masked_invalid(grid[has_key]), cmap=colours,
                            vmin=0, vmax=1, interpolation="nearest")
                wall_overlay(axis, env)
                axis.set_title(
                    f"{algorithm} vs VI, {reward_mode}, has_key={has_key}, full battery",
                    fontsize=10,
                )
                bare_axes(axis)
                mark_landmarks(axis, env)
            axes[row][0].set_ylabel(
                f"agreement {agreement:.0%}\nvisit-weighted {weighted:.0%}", fontsize=9
            )

    handles = [
        mpatches.Patch(color="#7bc043", label="same action as Value Iteration"),
        mpatches.Patch(color="#d1495b", label="different action"),
        mpatches.Patch(color="#2f3640", label="wall"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    figure.suptitle(
        "Policy difference map: model-free greedy action vs Value Iteration\n"
        "maps show the full-battery slice; the quoted percentages cover every energy bin",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    figure.savefig(common.figure_path("comparison", "policy_difference.png"), dpi=130)
    plt.close(figure)
    print(f"  wrote {paths.rel(common.figure_path('comparison', 'policy_difference.png'))}")


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
            mean_penalty_entries=("mean_penalty_entries", "mean"),
            memory_kilobytes=("memory_kilobytes", "mean"),
            train_seconds=("train_seconds", "mean"),
        )
        .reset_index()
    )

    agreement_path = common.raw_path("comparison", "algorithm_comparison.csv")
    if agreement_path.exists():
        agreement = (
            pd.read_csv(agreement_path)
            .groupby(["algorithm", "reward_mode"])[
                ["policy_agreement", "weighted_policy_agreement"]
            ]
            .mean()
            .reset_index()
        )
        grouped = grouped.merge(agreement, on=["algorithm", "reward_mode"], how="left")

    grouped = grouped.fillna(0.0)
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
    metrics = [
        ("success_rate", "greedy success rate", "success_rate_std"),
        ("mean_return", "greedy mean return", "return_std"),
        ("mean_steps", "steps per episode", None),
    ]
    if "policy_agreement" in comparison.columns:
        metrics.append(("policy_agreement", "policy agreement with VI", None))

    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.4))
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
            axis.bar(positions + offset * width - width / 2, values, width,
                     yerr=errors, capsize=3, label=mode,
                     color=MODE_COLORS.get(mode, "#777777"))
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
            record for record in load_records(algorithm)
            if record["reward_mode"] == "shaped"
        ]
        if not records:
            continue
        success = stack_curves(records, "eval_success_rate")
        episodes = np.asarray(records[0]["log"]["eval_episode"], dtype=float)
        if success is not None:
            band(axis, episodes, success, ALGO_COLORS[algorithm], algorithm)
            plotted = True

    planner = [r for r in load_records("value_iteration") if r["reward_mode"] == "shaped"]
    if planner:
        axis.axhline(planner[0]["final_eval"]["success_rate"],
                     color=ALGO_COLORS["value_iteration"], linestyle="--",
                     label="value iteration (optimal)")
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


# ------------------------------------------------------------- transfer plots


def plot_transfer(config: dict, payload: Optional[dict]) -> None:
    if not payload:
        return
    records = payload["records"]
    variants = sorted({record["variant"] for record in records})
    scenarios = payload["scenarios"]
    # Expand the scaled family back out to the individual betas.
    scenario_names = sorted({record["scenario"] for record in records},
                            key=lambda name: (name != "scratch", name))

    figure, axes = plt.subplots(len(variants), 3, figsize=(16, 4.6 * len(variants)),
                               squeeze=False)
    for row, variant in enumerate(variants):
        subset = [record for record in records if record["variant"] == variant]
        by_scenario = group_by(subset, "scenario")

        for scenario in scenario_names:
            group = by_scenario.get(scenario)
            if not group:
                continue
            colour = SCENARIO_COLORS.get(scenario, "#555555")
            episodes = np.asarray(group[0]["log"]["eval_episode"], dtype=float)
            success = stack_curves(group, "eval_success_rate")
            if success is not None:
                band(axes[row][0], episodes, success, colour, scenario)
            returns = stack_curves(group, "eval_return")
            if returns is not None:
                band(axes[row][1], episodes, returns, colour, scenario)

        positions = np.arange(len(scenario_names))
        width = 0.27
        for offset, (field, label) in enumerate(
            (("zero_shot_success", "zero-shot"),
             ("early_success", "early (<=300 ep)"),
             ("final_success", "final"))
        ):
            values = [
                float(np.mean([record[field] for record in by_scenario.get(name, [])]))
                if by_scenario.get(name) else np.nan
                for name in scenario_names
            ]
            axes[row][2].bar(positions + (offset - 1) * width, values, width,
                             label=label,
                             color=("#8d99ae", "#edae49", "#00798c")[offset])
        axes[row][2].set_xticks(positions)
        axes[row][2].set_xticklabels(
            [name.replace("_", "\n") for name in scenario_names], fontsize=8
        )
        axes[row][2].set_ylim(0, 1.05)
        axes[row][2].set_ylabel("success rate")
        axes[row][2].set_title(f"{variant}: initial vs early vs final")
        axes[row][2].grid(alpha=0.3, axis="y")
        axes[row][2].legend(fontsize=8)

        axes[row][0].set_title(f"{variant} target: greedy success rate")
        axes[row][0].set_ylabel("success rate")
        axes[row][0].set_ylim(-0.05, 1.05)
        axes[row][1].set_title(f"{variant} target: greedy return")
        axes[row][1].set_ylabel("return")
        for axis in axes[row][:2]:
            axis.set_xlabel("episode on the target maze")
            axis.grid(alpha=0.3)
            axis.legend(title="scenario", fontsize=8)

    figure.suptitle("Transfer learning: four ways of reusing the source Q-table")
    save(figure, common.figure_path("transfer", "transfer_curves.png"))


def plot_transfer_maps(config: dict, payload: Optional[dict]) -> None:
    """Q-difference maps before and after target training, plus what changed."""
    if not payload:
        return
    map_file = paths.ROOT / payload["q_maps_file"]
    if not map_file.exists():
        return

    source_env = common.build_env(
        config, reward_mode=config["transfer"].get("reward_mode", "shaped")
    )
    with np.load(map_file) as data:
        # Environment differences and the selective-transfer mask.
        variants = sorted(payload["targets"])
        figure, axes = plt.subplots(len(variants), 2, figsize=(10.5, 4.6 * len(variants)),
                                   squeeze=False)
        change_colours = matplotlib.colors.ListedColormap(["#d1495b", "#f7f7f5", "#7bc043"])
        for row, variant in enumerate(variants):
            target_env = common.build_env(
                config,
                reward_mode=config["transfer"].get("reward_mode", "shaped"),
                map_path=paths.map_file(config["student_id"], variant=variant),
            )
            meta = payload["targets"][variant]["metadata"]
            axes[row][0].imshow(data[f"{variant}__obstacle_diff"], cmap=change_colours,
                                vmin=-1, vmax=1, interpolation="nearest")
            axes[row][0].set_title(
                f"{variant}: {meta['obstacle_change_fraction']:.0%} of cells changed",
                fontsize=10,
            )
            bare_axes(axes[row][0])
            mark_landmarks(axes[row][0], target_env, colour="#2f3640")

            axes[row][1].imshow(data[f"{variant}__reuse_mask"], cmap="Greens",
                                vmin=0, vmax=1, interpolation="nearest")
            axes[row][1].set_title(
                f"{variant}: selective-transfer mask "
                f"({payload['targets'][variant]['reuse_fraction']:.0%} reused)",
                fontsize=10,
            )
            bare_axes(axes[row][1])
            mark_landmarks(axes[row][1], target_env, colour="#2f3640")

        handles = [
            mpatches.Patch(color="#7bc043", label="wall removed"),
            mpatches.Patch(color="#d1495b", label="wall added"),
            mpatches.Patch(color="#f7f7f5", label="unchanged"),
        ]
        figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
        figure.suptitle("Transfer targets: how each maze differs from the source")
        figure.tight_layout(rect=(0, 0.04, 1, 0.97))
        figure.savefig(common.figure_path("transfer", "environment_changes.png"), dpi=130)
        plt.close(figure)
        print(f"  wrote {paths.rel(common.figure_path('transfer', 'environment_changes.png'))}")

        # Q-difference maps: transferred prior, trained result, and the change.
        shown = [
            name for name in ("scratch", "full", "selective")
            if f"{variants[0]}__{name}__q_before" in data.files
        ]
        for variant in variants:
            figure, axes = plt.subplots(len(shown), 3, figsize=(13.5, 4.3 * len(shown)),
                                       squeeze=False)
            target_env = common.build_env(
                config,
                reward_mode=config["transfer"].get("reward_mode", "shaped"),
                map_path=paths.map_file(config["student_id"], variant=variant),
            )
            for row, scenario in enumerate(shown):
                before = data[f"{variant}__{scenario}__q_before"][0]
                after = data[f"{variant}__{scenario}__q_after"][0]
                heatmap(figure, axes[row][0], target_env, before,
                        f"{scenario}: max Q before target training")
                heatmap(figure, axes[row][1], target_env, after,
                        f"{scenario}: max Q after target training")
                heatmap(figure, axes[row][2], target_env, after - before,
                        f"{scenario}: difference (after - before)", cmap="coolwarm")
            figure.suptitle(
                f"Transfer Q-difference maps on the {variant} target (has_key=0)"
            )
            save(figure, common.figure_path("transfer", f"q_difference_{variant}.png"))

    # Source value map for reference.
    plot_maze(config, title="Source maze", filename="maze_layout.png", env=source_env)
    for variant in payload["targets"]:
        target_env = common.build_env(
            config,
            reward_mode=config["transfer"].get("reward_mode", "shaped"),
            map_path=paths.map_file(config["student_id"], variant=variant),
        )
        plot_maze(config, title=f"Transfer target: {variant}",
                  filename=f"maze_target_{variant}.png", env=target_env)


def write_transfer_table(payload: Optional[dict]) -> None:
    if not payload:
        return
    table = common.raw_path("transfer", "transfer_by_scenario.csv")
    if not table.exists():
        return
    frame = pd.read_csv(table)
    keep = [
        "variant", "scenario", "zero_shot_success", "early_success",
        "final_success", "final_success_std", "episodes_to_threshold",
        "final_delta", "speed_delta", "verdict",
    ]
    write_markdown_table(
        frame[[column for column in keep if column in frame.columns]],
        paths.RAW_DATA / "transfer_summary.md",
    )


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
    plot_gamma_sweep(load_payload("value_iteration", "value_iteration_gamma_sweep.json"))

    for algorithm in ("q_learning", "sarsa_lambda"):
        records = load_records(algorithm)
        plot_learner(algorithm, records)
        plot_learner_maps(config, algorithm, records)

    plot_epsilon_study(
        "q_learning",
        {
            reward_mode: load_payload(
                "q_learning", f"q_learning_epsilon_study_{reward_mode}.json"
            )
            for reward_mode in config["reward_modes"]
        },
    )
    plot_lambda_sweep(
        {
            reward_mode: load_records(
                "sarsa_lambda", f"sarsa_lambda_lambda_sweep_{reward_mode}.json"
            )
            for reward_mode in config["reward_modes"]
        }
    )

    transfer = load_payload("transfer", "transfer_results.json")
    plot_transfer(config, transfer)
    plot_transfer_maps(config, transfer)
    write_transfer_table(transfer)

    plot_policy_difference(
        config, load_payload("comparison", "algorithm_comparison.json")
    )
    comparison = build_comparison(config)
    plot_comparison(comparison)
    plot_combined_curves()

    if not comparison.empty:
        print()
        print("Algorithm comparison")
        print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
