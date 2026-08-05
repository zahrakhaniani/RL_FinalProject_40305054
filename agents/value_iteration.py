import numpy as np
from environments.maze import MazeEnv, Action


class ValueIterationAgent:
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = {}
        self.policy = {}
        self._initialize()

    def _initialize(self):
        for state in self.env.get_state_space():
            self.V[state] = 0.0
            self.policy[state] = Action.UP

    def train(self):
        iterations = 0
        while True:
            delta = 0
            iterations += 1
            for state in self.env.get_state_space():
                r, c, hk, do = state
                if (r, c) == self.env.goal:
                    continue

                old_value = self.V[state]
                action_values = []

                for action in Action:
                    self.env.agent_pos = (r, c)
                    self.env.has_key = bool(hk)
                    self.env.door_open = bool(do)
                    self.env.done = False
                    next_state, reward, done, _ = self.env.step(action)
                    value = reward + (0 if done else self.gamma * self.V.get(next_state, 0))
                    action_values.append(value)

                self.V[state] = max(action_values)
                self.policy[state] = Action(list(Action)[np.argmax(action_values)])
                delta = max(delta, abs(old_value - self.V[state]))

            if delta < self.theta:
                break

        return iterations

    def get_action(self, state):
        return self.policy.get(state, Action.UP)

    def get_values(self):
        return self.V.copy()
