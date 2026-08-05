"""Model-free SARSA(lambda) with backward-view eligibility traces.

Traces decay by ``gamma * lambda`` every step, so only a short window of recent
state-action pairs carries meaningful credit. Instead of sweeping the whole
table on every step, the active traces are held in compact NumPy arrays and
pruned once they fall below a threshold, which keeps a trace update to a couple
of vector operations regardless of table size.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from agents.base import TabularAgent, TrainingLog, evaluate_policy, log_episode
from environments.maze import State


class EligibilityTraces:
    """Sparse eligibility traces over flat Q-table indices."""

    def __init__(
        self,
        decay: float,
        threshold: float = 1e-4,
        replacing: bool = True,
        capacity: int = 256,
        prune_at: int = 192,
    ) -> None:
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.replacing = bool(replacing)
        self.prune_at = int(prune_at)
        self.index = np.zeros(capacity, dtype=np.int64)
        self.value = np.zeros(capacity, dtype=np.float64)
        self.position: Dict[int, int] = {}
        self.size = 0

    def clear(self) -> None:
        self.position.clear()
        self.size = 0

    def bump(self, flat_index: int) -> None:
        position = self.position.get(flat_index)
        if position is None:
            if self.size == self.index.size:
                self._grow()
            self.index[self.size] = flat_index
            self.value[self.size] = 1.0
            self.position[flat_index] = self.size
            self.size += 1
        elif self.replacing:
            self.value[position] = 1.0
        else:
            self.value[position] += 1.0

    def apply(self, q_flat: np.ndarray, scale: float) -> None:
        if self.size:
            active = self.index[: self.size]
            q_flat[active] += scale * self.value[: self.size]

    def decay_all(self) -> None:
        if not self.size:
            return
        self.value[: self.size] *= self.decay
        if self.size >= self.prune_at:
            self._prune()

    def _prune(self) -> None:
        keep = self.value[: self.size] >= self.threshold
        indices = self.index[: self.size][keep]
        values = self.value[: self.size][keep]
        self.size = int(indices.size)
        self.index[: self.size] = indices
        self.value[: self.size] = values
        self.position = {int(key): i for i, key in enumerate(indices)}

    def _grow(self) -> None:
        self.index = np.concatenate([self.index, np.zeros_like(self.index)])
        self.value = np.concatenate([self.value, np.zeros_like(self.value)])


class SarsaLambdaAgent(TabularAgent):
    name = "sarsa_lambda"

    def __init__(self, *args, lam: float = 0.9, replacing_traces: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lam = float(lam)
        self.replacing_traces = bool(replacing_traces)
        self._traces: Optional[EligibilityTraces] = None

    def hyperparameters(self) -> Dict[str, float]:
        params = super().hyperparameters()
        params.update({"lambda": self.lam, "replacing_traces": self.replacing_traces})
        return params

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
                log,
                self.env,
                episode,
                self.epsilon,
                self.alpha,
                td_errors=result["td_errors"],
                trace_counts=result["trace_counts"],
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

    @property
    def traces(self) -> EligibilityTraces:
        if self._traces is None:
            self._traces = EligibilityTraces(
                decay=self.gamma * self.lam, replacing=self.replacing_traces
            )
        return self._traces

    def run_episode(
        self,
        episode: int,
        total_episodes: int,
        seed: int,
        record: bool = False,
    ) -> dict:
        """One episode of learning, traces included.

        Split out from ``train`` so the GUI can drive training one episode at a
        time and animate what the agent actually did, using the same code path as
        the batch experiments.
        """
        q_flat = self.q.ravel()
        traces = self.traces
        traces.clear()

        state = self.env.reset(seed=seed)
        self.record_visit(state)
        action = self.get_action(state, greedy=False)
        deltas: List[float] = []
        trace_counts: List[int] = []
        path: List[Tuple[State, Optional[dict]]] = [(state, None)] if record else []
        done = False

        while not done:
            next_state, reward, done, info = self.env.step(action)
            terminal = info["success"] or info["energy_exhausted"]

            if terminal:
                next_action = None
                next_q = 0.0
            else:
                next_action = self.get_action(next_state, greedy=False)
                next_q = float(self.q[self.feature(next_state) + (next_action,)])

            current = self.feature(state) + (action,)
            delta = reward + self.gamma * next_q - float(self.q[current])

            traces.bump(self.flat_index(current))
            traces.apply(q_flat, self.alpha * delta)
            traces.decay_all()

            deltas.append(delta)
            trace_counts.append(traces.size)
            if record:
                path.append((next_state, info))
            state = next_state
            self.record_visit(state)
            if next_action is None:
                break
            action = next_action

        self.update_epsilon(episode, total_episodes)
        self.training_episodes += 1
        return {
            "td_errors": deltas,
            "trace_counts": trace_counts,
            "path": path,
            "outcome": self.env.outcome,
        }
