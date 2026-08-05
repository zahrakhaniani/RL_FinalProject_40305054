"""Model-free Q-Learning (off-policy temporal-difference control).

The learner never touches ``env.transitions()``; it only samples from
``env.step()``, so the 0.8 / 0.1 / 0.1 slip model is respected implicitly through
experience. Episodes cut short by ``max_steps`` are bootstrapped normally instead
of being treated as terminal, because truncation is an experiment-level limit
rather than a real absorbing state of the MDP.

A thinned sample of the individual Q-updates is kept so the report can show
concrete ``(state, action, reward, next_state, Q_old, Q_new)`` rows rather than
only aggregate curves.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from agents.base import TabularAgent, TrainingLog, evaluate_policy, log_episode
from environments.maze import State


class QLearningAgent(TabularAgent):
    name = "q_learning"

    def __init__(
        self,
        *args,
        q_log_every: int = 500,
        q_log_max: int = 500,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.q_log_every = int(q_log_every)
        self.q_log_max = int(q_log_max)
        self.q_update_samples: List[dict] = []
        self._updates = 0

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
            result = self.run_episode(episode, episodes, base + episode)
            log_episode(
                log, self.env, episode, self.epsilon, self.alpha, result["td_errors"]
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

    def run_episode(
        self,
        episode: int,
        total_episodes: int,
        seed: int,
        record: bool = False,
    ) -> dict:
        """One episode of learning. ``record`` also returns the path walked.

        Split out from ``train`` so the GUI can drive training one episode at a
        time and animate what the agent actually did, using the same code path as
        the batch experiments.
        """
        state = self.env.reset(seed=seed)
        self.record_visit(state)
        td_errors: List[float] = []
        path: List[Tuple[State, Optional[dict]]] = [(state, None)] if record else []
        done = False

        while not done:
            action = self.get_action(state, greedy=False)
            next_state, reward, done, info = self.env.step(action)
            td_errors.append(self._update(state, action, reward, next_state, info, episode))
            if record:
                path.append((next_state, info))
            state = next_state
            self.record_visit(state)

        self.update_epsilon(episode, total_episodes)
        self.training_episodes += 1
        return {"td_errors": td_errors, "path": path, "outcome": self.env.outcome}

    def _update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        info: dict,
        episode: int,
    ) -> float:
        index = self.feature(state) + (action,)
        terminal = info["success"] or info["energy_exhausted"]
        target = reward
        if not terminal:
            target += self.gamma * float(self.q[self.feature(next_state)].max())

        old_value = float(self.q[index])
        td_error = target - old_value
        new_value = old_value + self.alpha * td_error
        self.q[index] = new_value

        self._updates += 1
        if (
            len(self.q_update_samples) < self.q_log_max
            and self._updates % self.q_log_every == 0
        ):
            self.q_update_samples.append(
                {
                    "episode": episode,
                    "update": self._updates,
                    "state": str(tuple(int(x) for x in state)),
                    "action": int(action),
                    "reward": round(float(reward), 6),
                    "next_state": str(tuple(int(x) for x in next_state)),
                    "done": int(bool(terminal)),
                    "q_old": round(old_value, 6),
                    "q_new": round(new_value, 6),
                    "td_error": round(td_error, 6),
                    "alpha": self.alpha,
                    "gamma": self.gamma,
                    "epsilon": round(self.epsilon, 6),
                }
            )
        return td_error
