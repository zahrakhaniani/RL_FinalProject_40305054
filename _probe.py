import numpy as np

from agents.base import rollout
from environments.maze import bfs_distances
from experiments import common

config = common.load_config()

for reward_mode in ("shaped",):
    for gamma in (0.90, 0.95, 0.99):
        env = common.build_env(config, reward_mode=reward_mode, seed=0)
        agent = common.make_planner(env, config, gamma=gamma)
        stats = agent.train()
        trace = rollout(env, agent, seed=config["training"]["eval_seed"])
        cells = [(s[0], s[1]) for s in trace["states"]]
        d_key = bfs_distances(env.grid, [env.key], door_passable=False)
        print(f"gamma {gamma}: V(start)={stats['start_state_value']:.4f} "
              f"outcome={trace['outcome']} steps={trace['steps']} "
              f"return={trace['total_reward']:.2f}")
        print(f"   first 14 cells: {cells[:14]}")
        print(f"   distinct cells visited: {len(set(cells))}, "
              f"min distance to key reached: {min(int(d_key[c]) for c in cells)}")
        print(f"   events: {trace['events']}")
