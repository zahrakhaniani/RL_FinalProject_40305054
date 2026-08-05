import numpy as np
from copy import deepcopy
from agents.q_learning import QLearningAgent
from agents.sarsa_lambda import SarsaLambdaAgent
from environments.maze import MazeEnv


class TransferLearning:
    def __init__(self, source_agent, target_env, alpha_transfer=0.5):
        self.source_agent = source_agent
        self.target_env = target_env
        self.alpha_transfer = alpha_transfer

    def create_transferred_agent(self):
        target_agent = QLearningAgent(
            env=self.target_env,
            alpha=0.1,
            gamma=0.99,
            epsilon=0.5,
        )
        for state, q_values in self.source_agent.get_q_table().items():
            target_agent.Q[state] = q_values.copy()
        return target_agent

    def create_frozen_agent(self):
        transferred = self.create_transferred_agent()
        for state in transferred.Q:
            transferred.Q[state] = transferred.Q[state].copy()
        transferred.alpha = 0.01
        return transferred

    def train_transferred(self, episodes=500, max_steps=200):
        agent = self.create_transferred_agent()
        results = {"episodes": [], "rewards": [], "steps": []}

        for episode in range(episodes):
            state = self.target_env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, done, _ = self.target_env.step(action)
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

            results["episodes"].append(episode)
            results["rewards"].append(total_reward)
            results["steps"].append(step + 1)

        return results, agent

    def train_from_scratch(self, episodes=500, max_steps=200):
        agent = QLearningAgent(env=self.target_env, alpha=0.1, gamma=0.99, epsilon=1.0)
        results = {"episodes": [], "rewards": [], "steps": []}

        for episode in range(episodes):
            state = self.target_env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = agent.get_action(state)
                next_state, reward, done, _ = self.target_env.step(action)
                agent.update(state, action, reward, next_state, done)
                agent.decay_epsilon()
                state = next_state
                total_reward += reward
                if done:
                    break

            results["episodes"].append(episode)
            results["rewards"].append(total_reward)
            results["steps"].append(step + 1)

        return results, agent
