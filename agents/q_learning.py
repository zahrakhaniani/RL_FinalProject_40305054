import numpy as np
import random
from collections import defaultdict
from environments.maze import MazeEnv, Action


class QLearningAgent:
    def __init__(
        self,
        env,
        alpha=0.15,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.998,
        optimistic_init=1.0,
    ):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.optimistic_init = optimistic_init
        self.Q = defaultdict(lambda: np.full(len(Action), optimistic_init))

    def get_action(self, state, greedy=False):
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return Action.UP

        if not greedy and random.random() < self.epsilon:
            return random.choice(valid_actions)

        q_values = self.Q[state]
        max_q = max(q_values[a] for a in valid_actions)
        best_actions = [a for a in valid_actions if q_values[a] == max_q]
        return Action(random.choice(best_actions))

    def _max_valid_q(self, state):
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return 0.0
        q_values = self.Q[state]
        return max(q_values[a] for a in valid_actions)

    def update(self, state, action, reward, next_state, done):
        current_q = self.Q[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * self._max_valid_q(next_state)
        self.Q[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, episodes=3000, max_steps=500):
        results = {"episodes": [], "rewards": [], "steps": [], "epsilon": []}

        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            step = 0

            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

            self.decay_epsilon()
            results["episodes"].append(episode)
            results["rewards"].append(total_reward)
            results["steps"].append(step + 1)
            results["epsilon"].append(self.epsilon)

        return results

    def get_q_table(self):
        return dict(self.Q)
