"""Standalone pygame GUI for the maze agents.

Three things can be switched independently while it runs:

* **algorithm** -- Value Iteration, Q-Learning or SARSA(lambda);
* **environment** -- the source maze or either transfer target;
* **mode** -- ``eval`` replays the greedy policy, ``train`` learns live.

In ``eval`` mode a trained model is loaded from ``results/models/`` if one exists,
so the window shows the policy the experiments actually produced. In ``train``
mode the agent learns episode by episode through the same ``run_episode`` code the
batch experiments use, and each episode is replayed step by step, so exploration
visibly narrows as epsilon decays. Pressing ``f`` trains without animating, which
is much faster when you want to watch the success rate climb.

    python gui/app.py
    python gui/app.py --algorithm q_learning --environment similar --mode train
    python gui/app.py --record          # write PNG frames to results/videos/
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pygame

import paths
from agents.base import evaluate_policy
from environments.variants import make_variant
from experiments import common
from gui.renderer import MazeRenderer

ALGORITHMS = ("value_iteration", "q_learning", "sarsa_lambda")
ALGORITHM_LABELS = {
    "value_iteration": "Value Iteration (model-based)",
    "q_learning": "Q-Learning (off-policy TD)",
    "sarsa_lambda": "SARSA(lambda) (on-policy TD)",
}
ENVIRONMENTS = ("source", "similar", "different")
ENVIRONMENT_LABELS = {
    "source": "source maze",
    "similar": "target A (similar)",
    "different": "target B (different)",
}
MODES = ("eval", "train")

CONTROLS = (
    "space  play / pause",
    "n      single step",
    "r      reset episode",
    "1 2 3  algorithm",
    "e      environment",
    "t      train / eval",
    "m      sparse <-> shaped",
    "v      value heat map",
    "p      policy arrows",
    "b      trail",
    "f      fast training",
    "+ -    speed",
    "g      saved figures",
    "c      record frames",
    "esc    quit",
)


def build_environment(config: dict, environment: str, reward_mode: str):
    """Load the source maze, or a transfer target (generating it if needed)."""
    if environment == "source":
        return common.build_env(config, reward_mode=reward_mode, seed=0)

    map_path = paths.map_file(config["student_id"], variant=environment)
    if map_path.exists():
        return common.build_env(config, reward_mode=reward_mode, seed=0, map_path=map_path)

    source = common.build_env(config, reward_mode=reward_mode, seed=0)
    target = make_variant(
        source,
        environment,
        seed=int(config["transfer"]["variant_seed"]),
        reward_mode=reward_mode,
        rewards=config["rewards"],
    )
    target.save_map(map_path)
    return target


class MazeApp:
    def __init__(
        self,
        config: dict,
        algorithm: str = "value_iteration",
        environment: str = "source",
        reward_mode: str = "shaped",
        mode: str = "eval",
        cell_size: int = 32,
        steps_per_second: float = 10.0,
        record: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.algorithm = algorithm
        self.environment = environment
        self.reward_mode = reward_mode
        self.mode = mode
        self.cell_size = cell_size
        self.steps_per_second = float(steps_per_second)
        self.base_seed = seed if seed is not None else 0

        self.show_values = False
        self.show_policy = False
        self.show_trail = True
        self.playing = True
        self.fast_training = False
        self.gallery_open = False
        self.recording = False
        self.frame_index = 0
        self.record_root: Optional[Path] = None

        self.episode_index = 0
        self.total_episodes = 0
        self.total_reward = 0.0
        self.slips = 0
        self.bump_cell: Optional[Tuple[int, int]] = None
        self.trail: List[Tuple[int, int]] = []
        self.notice = ""
        self.recent_outcomes: deque = deque(maxlen=50)
        self.replay: List[Tuple[tuple, Optional[dict]]] = []
        self.replay_position = 0
        self.eval_success: Optional[float] = None

        self.figures: List[Path] = []
        self.figure_index = 0
        self._figure_cache: Dict[Path, pygame.Surface] = {}

        self.env = None
        self.agent = None
        self._load()
        self.renderer = MazeRenderer(self.env, cell_size=cell_size)
        if record:
            self._toggle_recording()

    # -------------------------------------------------------------- agent setup

    def _load(self) -> None:
        """Rebuild the environment and agent for the current selection."""
        self.env = build_environment(self.config, self.environment, self.reward_mode)
        self.episode_index = 0
        self.recent_outcomes.clear()
        self.eval_success = None

        if self.algorithm == "value_iteration":
            self.agent = self._load_planner()
            self.total_episodes = 0
        else:
            self.total_episodes = int(self.config[self.algorithm]["episodes"])
            if self.mode == "train":
                # Training starts from an empty table so the progress bar and the
                # success rate mean what they say.
                self.agent = common.make_learner(
                    self.algorithm, self.env, self.config,
                    seed=self.config["training"]["seeds"][0],
                )
                self.notice = f"training from scratch, {self.total_episodes} episodes"
            else:
                self.agent, loaded = self._load_learner()
                if not loaded:
                    self.notice = (
                        f"nothing trained on the {self.environment} maze"
                        " -- press t to train live"
                    )

        self.reset_episode()

    def _load_planner(self):
        model = common.model_path(
            "value_iteration", f"value_iteration_{self.reward_mode}.npz"
        )
        agent = common.make_planner(
            self.env, self.config, gamma=self.config["value_iteration"]["gamma"]
        )
        # A saved table only matches the source maze; targets have to be re-solved.
        if self.environment == "source" and model.exists() and self.mode == "eval":
            agent.load(model)
            self.notice = f"loaded {paths.rel(model)}"
        else:
            print(f"solving value iteration on the {self.environment} maze ...")
            stats = agent.train()
            self.notice = (
                f"solved in {stats['train_seconds']:.1f}s, "
                f"{stats['iterations']} sweeps, residual {stats['bellman_residual']:.1e}"
            )
        return agent

    def _load_learner(self):
        seeds = self.config["training"]["seeds"]
        agent = common.make_learner(self.algorithm, self.env, self.config, seed=seeds[0])
        for candidate in self._model_candidates(seeds):
            if candidate.exists():
                try:
                    agent.load(candidate)
                except ValueError:
                    continue
                self.notice = f"loaded {paths.rel(candidate)}"
                return agent, True
        return agent, False

    def _model_candidates(self, seeds) -> List[Path]:
        """Models that were actually trained on the maze currently selected.

        A table trained elsewhere is not offered as a fallback: energy bins are
        cut relative to each maze's own budget, so the same table means
        different things on different mazes and the policy it draws would be
        meaningless.
        """
        if self.environment != "source":
            if self.algorithm != "q_learning":
                return []
            return [
                common.model_path(
                    "transfer", f"{self.environment}_{scenario}_seed{seeds[0]}.npz"
                )
                for scenario in ("full", "selective", "scratch")
            ]
        return [
            common.model_path(
                self.algorithm, f"{self.algorithm}_{self.reward_mode}_seed{seed}.npz"
            )
            for seed in seeds
        ]

    # ---------------------------------------------------------------- episodes

    def reset_episode(self) -> None:
        self.env.reset(seed=self.base_seed + self.episode_index)
        self.trail = [self.env.agent_pos]
        self.total_reward = 0.0
        self.slips = 0
        self.bump_cell = None
        self.replay = []
        self.replay_position = 0

    def next_episode(self) -> None:
        if self.env.outcome:
            self.recent_outcomes.append(1.0 if self.env.outcome == "success" else 0.0)
        self.episode_index += 1
        if self.mode == "train" and self.training_finished:
            self.playing = False
            self.notice = f"training finished after {self.agent.training_episodes} episodes"
            return
        self.reset_episode()
        if self.mode == "train":
            self._start_training_episode()

    @property
    def training_finished(self) -> bool:
        return (
            self.mode == "train"
            and self.total_episodes > 0
            and self.agent is not None
            and getattr(self.agent, "training_episodes", 0) >= self.total_episodes
        )

    def _start_training_episode(self) -> None:
        """Learn one episode, then replay it so the behaviour is visible."""
        episode = int(getattr(self.agent, "training_episodes", 0)) + 1
        result = self.agent.run_episode(
            episode, self.total_episodes, self.base_seed + self.episode_index, record=True
        )
        self.replay = result["path"]
        self.replay_position = 0
        # Rewind the environment so the recorded path can be animated from the start.
        self.env.reset(seed=self.base_seed + self.episode_index)
        self.trail = [self.env.agent_pos]
        self.total_reward = 0.0
        self.slips = 0

        if episode % 100 == 0:
            metrics = evaluate_policy(self.env, self.agent, episodes=20, seed=555_000)
            self.eval_success = metrics["success_rate"]

    def step_once(self) -> None:
        if self.mode == "train" and self.replay:
            self._advance_replay()
            return
        if self.env.done:
            return
        state = self.env.state
        action = self.agent.get_action(state, greedy=True)
        _, reward, done, info = self.env.step(action)
        self._record_step(reward, info)

    def _advance_replay(self) -> None:
        """Walk through the states the training episode actually visited."""
        self.replay_position += 1
        if self.replay_position >= len(self.replay):
            self.env.done = True
            return
        state, info = self.replay[self.replay_position]
        self.env.agent_pos = (state[0], state[1])
        self.env.has_key = state[2]
        self.env.energy = state[3]
        self.env.steps = self.replay_position
        self.trail.append(self.env.agent_pos)
        if info:
            self.slips += int(info.get("slipped", False))
            self.bump_cell = self.env.agent_pos if info.get("collision") else None
        if self.replay_position == len(self.replay) - 1:
            self.env.done = True
            self.env.outcome = self._replay_outcome()

    def _replay_outcome(self) -> str:
        final = self.replay[-1][0]
        if (final[0], final[1]) == self.env.goal and final[2] == 1:
            return "success"
        if final[3] <= 0:
            return "out_of_energy"
        return "max_steps"

    def _record_step(self, reward: float, info: dict) -> None:
        self.total_reward += reward
        self.trail.append(self.env.agent_pos)
        self.slips += int(info.get("slipped", False))
        self.bump_cell = self.env.agent_pos if info.get("collision") else None

    def _train_without_animation(self, budget: int = 40) -> None:
        """Crunch episodes between frames, for when watching each step is too slow."""
        for _ in range(budget):
            if self.training_finished:
                self.playing = False
                self.notice = f"training finished after {self.agent.training_episodes} episodes"
                return
            episode = int(self.agent.training_episodes) + 1
            self.agent.run_episode(
                episode, self.total_episodes, self.base_seed + self.episode_index
            )
            self.recent_outcomes.append(1.0 if self.env.outcome == "success" else 0.0)
            self.episode_index += 1
            if episode % 100 == 0:
                metrics = evaluate_policy(self.env, self.agent, episodes=20, seed=555_000)
                self.eval_success = metrics["success_rate"]
        self.trail = [self.env.agent_pos]

    # ------------------------------------------------------------------ display

    def _overlays(self):
        value_grid = policy_grid = None
        if self.show_values or self.show_policy:
            has_key = int(self.env.has_key)
            energy = max(1, int(self.env.energy))
            if self.show_values and hasattr(self.agent, "value_grid"):
                value_grid = self.agent.value_grid(has_key=has_key, energy=energy)
            if self.show_policy and hasattr(self.agent, "greedy_policy_grid"):
                policy_grid = self.agent.greedy_policy_grid(has_key=has_key, energy=energy)
        return value_grid, policy_grid

    def _sections(self):
        recent = (
            f"{sum(self.recent_outcomes) / len(self.recent_outcomes):.0%}"
            if self.recent_outcomes else "-"
        )
        events = self.env.events
        live = [
            ("episode", self.episode_index + 1),
            ("step", f"{self.env.steps} / {self.env.max_steps}"),
            ("return", f"{self.env.episode_reward:+.2f}"),
            ("has key", "yes" if self.env.has_key else "no"),
            ("outcome", self.env.outcome or ("running" if not self.env.done else "-")),
            (f"success rate (last {self.recent_outcomes.maxlen})", recent),
            ("wall hits", events["wall_collisions"]),
            ("penalty cells", events["penalty_entries"]),
            ("door blocked", events["door_blocked"]),
            ("slips", self.slips),
        ]
        if self.eval_success is not None:
            live.append(("greedy eval success", f"{self.eval_success:.0%}"))

        setup = [
            ("algorithm", self.algorithm),
            ("environment", ENVIRONMENT_LABELS[self.environment]),
            ("reward mode", self.reward_mode),
            ("mode", self.mode + (" (fast)" if self.fast_training else "")),
            ("speed", f"{self.steps_per_second:.0f} steps/s"),
            ("recording", "on" if self.recording else "off"),
        ]

        hyper = [("gamma", f"{getattr(self.agent, 'gamma', float('nan')):.3f}")]
        if self.algorithm == "value_iteration":
            hyper.append(("theta", f"{self.agent.theta:.0e}"))
            hyper.append(("energy sweeps", self.agent.max_energy))
        else:
            hyper.extend(
                [
                    ("alpha", f"{self.agent.alpha:.3f}"),
                    ("epsilon", f"{self.agent.epsilon:.3f}"),
                    ("epsilon decay", self.agent.epsilon_schedule),
                    ("energy bins", self.agent.energy_bins),
                    ("trained episodes", self.agent.training_episodes),
                ]
            )
            if hasattr(self.agent, "lam"):
                hyper.append(("lambda", f"{self.agent.lam:.2f}"))
                hyper.append(
                    ("traces", "replacing" if self.agent.replacing_traces else "accumulating")
                )

        return (("live episode", live), ("setup", setup), ("hyperparameters", hyper))

    def _progress(self):
        if self.mode != "train" or not self.total_episodes:
            return None
        done = int(getattr(self.agent, "training_episodes", 0))
        return (
            done / self.total_episodes,
            f"training progress  {done} / {self.total_episodes} episodes",
        )

    def _status(self) -> str:
        lines = [ALGORITHM_LABELS[self.algorithm]]
        if self.notice:
            lines.append(self.notice[:52])
        lines.append("playing" if self.playing else "paused")
        return "\n".join(lines)

    # ---------------------------------------------------------------- gallery

    def _refresh_figures(self) -> None:
        self.figures = sorted(paths.FIGURES.rglob("*.png"))
        self.figure_index = min(self.figure_index, max(0, len(self.figures) - 1))

    def _current_figure(self) -> Tuple[Optional[pygame.Surface], str]:
        if not self.figures:
            return None, "run experiments/analysis.py to generate the figures"
        path = self.figures[self.figure_index]
        if path not in self._figure_cache:
            try:
                self._figure_cache[path] = pygame.image.load(str(path))
            except pygame.error:
                return None, f"could not load {paths.rel(path)}"
        return self._figure_cache[path], paths.rel(path)

    # ---------------------------------------------------------------- recording

    def _toggle_recording(self) -> None:
        self.recording = not self.recording
        if self.recording:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.record_root = paths.subdir(paths.VIDEOS, self.algorithm) / f"run_{stamp}"
            self.record_root.mkdir(parents=True, exist_ok=True)
            self.frame_index = 0
            print(f"recording frames to {paths.rel(self.record_root)}")
        else:
            print(f"stopped recording ({self.frame_index} frames)")

    def _save_frame(self, surface) -> None:
        if not self.recording or self.record_root is None:
            return
        pygame.image.save(surface, str(self.record_root / f"frame_{self.frame_index:05d}.png"))
        self.frame_index += 1

    # -------------------------------------------------------------------- loop

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("RL Maze Solver -- 40305054")
        surface = pygame.display.set_mode(self.renderer.window_size)
        clock = pygame.time.Clock()
        accumulator = 0.0
        finished_at: Optional[int] = None
        running = True

        if self.mode == "train" and self.algorithm != "value_iteration":
            self._start_training_episode()

        while running:
            delta = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event)

            if self.gallery_open:
                image, caption = self._current_figure()
                self.renderer.draw_gallery(
                    surface, image, caption, self.figure_index, len(self.figures),
                    "left / right to browse, g or esc to go back",
                )
                pygame.display.flip()
                continue

            if self.playing and self.mode == "train" and self.fast_training:
                self._train_without_animation()
            elif self.playing and not self.env.done:
                accumulator += delta
                interval = 1.0 / max(self.steps_per_second, 0.5)
                while accumulator >= interval and not self.env.done:
                    self.step_once()
                    accumulator -= interval
                if self.env.done:
                    finished_at = pygame.time.get_ticks()
            elif self.playing and self.env.done:
                if finished_at is None:
                    finished_at = pygame.time.get_ticks()
                elif pygame.time.get_ticks() - finished_at > 1200:
                    finished_at = None
                    self.next_episode()

            value_grid, policy_grid = self._overlays()
            self.renderer.draw(
                surface,
                self.env,
                trail=self.trail if self.show_trail else None,
                value_grid=value_grid,
                policy_grid=policy_grid,
                sections=self._sections(),
                status=self._status(),
                progress=self._progress(),
                bump_cell=self.bump_cell,
                controls=CONTROLS,
            )
            self._save_frame(surface)
            pygame.display.flip()

        pygame.quit()

    def _handle_key(self, event) -> bool:
        key = event.key
        if self.gallery_open:
            if key in (pygame.K_g, pygame.K_ESCAPE):
                self.gallery_open = False
            elif key in (pygame.K_RIGHT, pygame.K_n) and self.figures:
                self.figure_index = (self.figure_index + 1) % len(self.figures)
            elif key in (pygame.K_LEFT, pygame.K_p) and self.figures:
                self.figure_index = (self.figure_index - 1) % len(self.figures)
            return True

        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key == pygame.K_SPACE:
            self.playing = not self.playing
        elif key == pygame.K_n:
            self.playing = False
            self.step_once()
        elif key == pygame.K_r:
            self.reset_episode()
            if self.mode == "train":
                self._start_training_episode()
        elif key == pygame.K_v:
            self.show_values = not self.show_values
        elif key == pygame.K_p:
            self.show_policy = not self.show_policy
        elif key == pygame.K_b:
            self.show_trail = not self.show_trail
        elif key == pygame.K_f:
            self.fast_training = not self.fast_training
        elif key == pygame.K_c:
            self._toggle_recording()
        elif key == pygame.K_g:
            self._refresh_figures()
            self.gallery_open = True
        elif key == pygame.K_m:
            self.reward_mode = "sparse" if self.reward_mode == "shaped" else "shaped"
            self._load()
        elif key == pygame.K_e:
            self.environment = ENVIRONMENTS[
                (ENVIRONMENTS.index(self.environment) + 1) % len(ENVIRONMENTS)
            ]
            self._load()
        elif key == pygame.K_t:
            self.mode = "train" if self.mode == "eval" else "eval"
            self._load()
            if self.mode == "train" and self.algorithm != "value_iteration":
                self._start_training_episode()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_UP):
            self.steps_per_second = min(240.0, self.steps_per_second * 1.5)
        elif key in (pygame.K_MINUS, pygame.K_DOWN):
            self.steps_per_second = max(1.0, self.steps_per_second / 1.5)
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3):
            self.algorithm = ALGORITHMS[key - pygame.K_1]
            self._load()
            if self.mode == "train" and self.algorithm != "value_iteration":
                self._start_training_episode()
        return True


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Interactive maze viewer")
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    parser.add_argument("--algorithm", default="value_iteration", choices=ALGORITHMS)
    parser.add_argument("--environment", default="source", choices=ENVIRONMENTS)
    parser.add_argument("--reward-mode", default="shaped", choices=("sparse", "shaped"))
    parser.add_argument("--mode", default="eval", choices=MODES)
    parser.add_argument("--cell-size", type=int, default=32)
    parser.add_argument("--speed", type=float, default=10.0, help="steps per second")
    parser.add_argument("--seed", type=int, default=None, help="first episode seed")
    parser.add_argument("--record", action="store_true", help="save PNG frames")
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    common.ensure_map(config, verbose=False)

    MazeApp(
        config,
        algorithm=args.algorithm,
        environment=args.environment,
        reward_mode=args.reward_mode,
        mode=args.mode,
        cell_size=args.cell_size,
        steps_per_second=args.speed,
        record=args.record,
        seed=args.seed,
    ).run()


if __name__ == "__main__":
    main()
