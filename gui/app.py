"""Interactive pygame viewer for the trained policies.

Loads the shared maze and whichever agent you ask for, then animates greedy
episodes under the real stochastic dynamics, so the slips are visible. Trained
models are picked up from ``results/models/`` when they exist; otherwise the
agent is trained on the spot.

    python gui/app.py
    python gui/app.py --algorithm q_learning --reward-mode sparse
    python gui/app.py --record          # write PNG frames to results/videos/
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pygame

import paths
from experiments import common
from gui.renderer import MazeRenderer

ALGORITHMS = ("value_iteration", "q_learning", "sarsa_lambda")
ALGORITHM_LABELS = {
    "value_iteration": "Value Iteration (model-based)",
    "q_learning": "Q-Learning (model-free)",
    "sarsa_lambda": "SARSA(lambda) (model-free)",
}


class MazeApp:
    def __init__(
        self,
        config: dict,
        algorithm: str = "value_iteration",
        reward_mode: str = "shaped",
        cell_size: int = 34,
        steps_per_second: float = 8.0,
        record: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config
        self.algorithm = algorithm
        self.reward_mode = reward_mode
        self.cell_size = cell_size
        self.steps_per_second = float(steps_per_second)
        self.episode_seed = seed if seed is not None else 0

        self.show_values = False
        self.show_policy = False
        self.show_trail = True
        self.playing = True
        self.recording = False
        self.frame_index = 0
        self.record_root: Optional[Path] = None
        self.episodes_finished = 0
        self.finished_at: Optional[int] = None
        self.total_reward = 0.0
        self.slips = 0
        self.collisions = 0
        self.trail: List[Tuple[int, int]] = []
        self.notice = ""

        self.env = None
        self.agent = None
        self._load(algorithm, reward_mode)
        self.renderer = MazeRenderer(self.env, cell_size=cell_size)
        if record:
            self._toggle_recording()

    # -------------------------------------------------------------- agent setup

    def _load(self, algorithm: str, reward_mode: str) -> None:
        self.algorithm = algorithm
        self.reward_mode = reward_mode
        self.env = common.build_env(self.config, reward_mode=reward_mode, seed=0)

        if algorithm == "value_iteration":
            agent = common.make_planner(self.env, self.config)
            model = common.model_path(algorithm, f"{algorithm}_{reward_mode}.npz")
            if model.exists():
                agent.load(model)
                self.notice = f"loaded {paths.rel(model)}"
            else:
                print(f"training Value Iteration ({reward_mode}) ...")
                agent.train()
                self.notice = "trained value iteration in-session"
        else:
            seeds = self.config["training"]["seeds"]
            agent = common.make_learner(algorithm, self.env, self.config, seed=seeds[0])
            model = next(
                (
                    path
                    for path in (
                        common.model_path(algorithm, f"{algorithm}_{reward_mode}_seed{s}.npz")
                        for s in seeds
                    )
                    if path.exists()
                ),
                None,
            )
            if model is not None:
                agent.load(model)
                self.notice = f"loaded {paths.rel(model)}"
            else:
                episodes = self.config[algorithm]["episodes"]
                print(
                    f"no saved model for {algorithm} ({reward_mode}); "
                    f"training {episodes} episodes ..."
                )
                agent.train(episodes=episodes, eval_every=0)
                self.notice = f"trained {episodes} episodes in-session"

        self.agent = agent
        self.reset_episode()

    # ---------------------------------------------------------------- episodes

    def reset_episode(self) -> None:
        self.env.reset(seed=self.episode_seed)
        self.trail = [self.env.agent_pos]
        self.total_reward = 0.0
        self.slips = 0
        self.collisions = 0
        self.finished_at = None

    def next_episode(self) -> None:
        self.episode_seed += 1
        self.episodes_finished += 1
        self.reset_episode()

    def step_once(self) -> None:
        if self.env.done:
            return
        state = self.env.state
        action = self.agent.get_action(state, greedy=True)
        _, reward, done, info = self.env.step(action)
        self.total_reward += reward
        self.trail.append(self.env.agent_pos)
        if info.get("slipped"):
            self.slips += 1
        if info.get("collision"):
            self.collisions += 1
        if done:
            self.finished_at = pygame.time.get_ticks()

    # ------------------------------------------------------------------ display

    def _overlays(self):
        value_grid = policy_grid = None
        if self.show_values or self.show_policy:
            has_key = int(self.env.has_key)
            energy = int(self.env.energy)
            if self.show_values and hasattr(self.agent, "value_grid"):
                value_grid = self.agent.value_grid(has_key=has_key, energy=energy)
            if self.show_policy and hasattr(self.agent, "greedy_policy_grid"):
                policy_grid = self.agent.greedy_policy_grid(has_key=has_key, energy=energy)
        return value_grid, policy_grid

    def _hud(self):
        outcome = self.env.outcome or ("running" if not self.env.done else "-")
        return [
            ("algorithm", self.algorithm),
            ("reward mode", self.reward_mode),
            ("episode seed", self.episode_seed),
            ("step", f"{self.env.steps} / {self.env.max_steps}"),
            ("return", f"{self.total_reward:+.2f}"),
            ("has key", "yes" if self.env.has_key else "no"),
            ("slips", self.slips),
            ("wall bumps", self.collisions),
            ("outcome", outcome),
            ("speed", f"{self.steps_per_second:.0f} steps/s"),
            ("recording", "on" if self.recording else "off"),
        ]

    def _status(self) -> str:
        lines = [ALGORITHM_LABELS[self.algorithm]]
        if self.notice:
            lines.append(self.notice)
        lines.append("paused" if not self.playing else "playing")
        return "\n".join(lines)

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
        running = True

        while running:
            delta = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event)

            if self.playing and not self.env.done:
                accumulator += delta
                interval = 1.0 / max(self.steps_per_second, 0.5)
                while accumulator >= interval and not self.env.done:
                    self.step_once()
                    accumulator -= interval
            elif self.env.done and self.playing and self.finished_at is not None:
                if pygame.time.get_ticks() - self.finished_at > 1400:
                    self.next_episode()

            value_grid, policy_grid = self._overlays()
            self.renderer.draw(
                surface,
                self.env,
                trail=self.trail if self.show_trail else None,
                value_grid=value_grid,
                policy_grid=policy_grid,
                hud_lines=self._hud(),
                status=self._status(),
            )
            self._save_frame(surface)
            pygame.display.flip()

        pygame.quit()

    def _handle_key(self, event) -> bool:
        key = event.key
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key == pygame.K_SPACE:
            self.playing = not self.playing
        elif key == pygame.K_n:
            self.playing = False
            self.step_once()
        elif key == pygame.K_r:
            self.reset_episode()
        elif key == pygame.K_v:
            self.show_values = not self.show_values
        elif key == pygame.K_p:
            self.show_policy = not self.show_policy
        elif key == pygame.K_t:
            self.show_trail = not self.show_trail
        elif key == pygame.K_c:
            self._toggle_recording()
        elif key == pygame.K_m:
            self._load(self.algorithm, "sparse" if self.reward_mode == "shaped" else "shaped")
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_UP):
            self.steps_per_second = min(120.0, self.steps_per_second * 1.5)
        elif key in (pygame.K_MINUS, pygame.K_DOWN):
            self.steps_per_second = max(1.0, self.steps_per_second / 1.5)
        elif key in (pygame.K_1, pygame.K_2, pygame.K_3):
            self._load(ALGORITHMS[key - pygame.K_1], self.reward_mode)
        return True


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Interactive maze viewer")
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    parser.add_argument("--algorithm", default="value_iteration", choices=ALGORITHMS)
    parser.add_argument("--reward-mode", default="shaped", choices=("sparse", "shaped"))
    parser.add_argument("--cell-size", type=int, default=34)
    parser.add_argument("--speed", type=float, default=8.0, help="steps per second")
    parser.add_argument("--seed", type=int, default=None, help="first episode seed")
    parser.add_argument("--record", action="store_true", help="save PNG frames")
    args = parser.parse_args(argv)

    paths.ensure_dirs()
    config = common.load_config(args.config)
    common.ensure_map(config, verbose=False)

    app = MazeApp(
        config,
        algorithm=args.algorithm,
        reward_mode=args.reward_mode,
        cell_size=args.cell_size,
        steps_per_second=args.speed,
        record=args.record,
        seed=args.seed,
    )
    app.run()


if __name__ == "__main__":
    main()
