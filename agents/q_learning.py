"""Model-free Q-Learning (off-policy temporal-difference control).

The learner never touches ``env.transitions()``; it only samples from
``env.step()``, so the 0.8 / 0.1 / 0.1 slip model is respected implicitly
through experience. Episodes cut short by ``max_steps`` are bootstrapped
normally instead of being treated as terminal, because truncation is an
experiment-level limit rather than a real absorbing state of the MDP.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from agents.base import TabularAgent, TrainingLog, evaluate_policy
from environments.maze import State


class QLearningAgent(TabularAgent):
    name = "q_learning"

    def train(
        self,
        episodes: int = 3000,
        eval_every: int = 100,
        eval_episodes: int = 30,
        eval_seed: int = 555_000,
        episode_seed_base: Optional[int] = None,
        verbose: bool = False,
    ) -> TrainingLog:
        log = TrainingLog()
        base = self.seed * 1_000_003 if episode_seed_base is None else episode_seed_base
        started = time.perf_counter()

        for episode in range(1, episodes + 1):
            state = self.env.reset(seed=base + episode)
            total_reward = 0.0
            done = False

            while not done:
                action = self.get_action(state, greedy=False)
                next_state, reward, done, info = self.env.step(action)
                self._update(state, action, reward, next_state, info)
                total_reward += reward
                state = next_state

            self.decay_epsilon()
            self.training_episodes += 1
            log.add_episode(
                episode=episode,
                reward=total_reward,
                steps=self.env.steps,
                success=self.env.outcome == "success",
                energy_left=self.env.energy,
                epsilon=self.epsilon,
                outcome=self.env.outcome or "unknown",
            )

            if eval_every and episode % eval_every == 0:
                metrics = evaluate_policy(self.env, self, eval_episodes, eval_seed)
                log.add_eval(episode, metrics)
                if verbose:
                    print(
                        f"    episode {episode:5d} | eps {self.epsilon:.3f} | "
                        f"greedy success {metrics['success_rate']:.2f} | "
                        f"greedy return {metrics['mean_return']:8.2f}"
                    )

        self.train_seconds = time.perf_counter() - started
        return log

    def _update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        info: dict,
    ) -> None:
        index = self.feature(state) + (action,)
        terminal = info["success"] or info["energy_exhausted"]
        target = reward
        if not terminal:
            target += self.gamma * float(self.q[self.feature(next_state)].max())
        self.q[index] += self.alpha * (target - self.q[index])
