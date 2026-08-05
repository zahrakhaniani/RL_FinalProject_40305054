import numpy as np
import random
from collections import defaultdict
from environments.maze import MazeEnv, Action


class SarsaLambdaAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, lam=0.9):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lam = lam
        self.Q = defaultdict(lambda: np.zeros(len(Action)))
        self.E = defaultdict(lambda: np.zeros(len(Action)))

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

    def reset_eligibility(self):
        self.E = defaultdict(lambda: np.zeros(len(Action)))

    def update(self, state, action, reward, next_state, next_action, done):
        delta = reward - self.Q[state][action]
        if not done:
            delta += self.gamma * self.Q[next_state][next_action]

        self.E[state][action] += 1

        states = list(self.E.keys())
        for s in states:
            for a in Action:
                self.Q[s][a] += self.alpha * delta * self.E[s][a]
                self.E[s][a] *= self.gamma * self.lam

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, episodes=1000, max_steps=500):
        results = {"episodes": [], "rewards": [], "steps": [], "epsilon": []}

        for episode in range(episodes):
            state = self.env.reset()
            action = self.get_action(state)
            self.reset_eligibility()
            total_reward = 0

            for step in range(max_steps):
                next_state, reward, done, _ = self.env.step(action)
                next_action = self.get_action(next_state)
                self.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
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
