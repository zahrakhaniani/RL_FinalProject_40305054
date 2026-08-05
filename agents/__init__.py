"""Value Iteration, Q-Learning and SARSA(lambda) agents."""

from .base import (
    BaseAgent,
    TabularAgent,
    TrainingLog,
    energy_bin,
    evaluate_policy,
    rollout,
)
from .q_learning import QLearningAgent
from .sarsa_lambda import EligibilityTraces, SarsaLambdaAgent
from .value_iteration import ValueIterationAgent

AGENT_REGISTRY = {
    "value_iteration": ValueIterationAgent,
    "q_learning": QLearningAgent,
    "sarsa_lambda": SarsaLambdaAgent,
}

__all__ = [
    "AGENT_REGISTRY",
    "BaseAgent",
    "EligibilityTraces",
    "QLearningAgent",
    "SarsaLambdaAgent",
    "TabularAgent",
    "TrainingLog",
    "ValueIterationAgent",
    "energy_bin",
    "evaluate_policy",
    "rollout",
]
