# RL_FinalProject_40305054

Reinforcement learning on a 16x16 stochastic maze with a key, a locked door,
penalty cells and a limited energy budget. Three algorithms are implemented from
scratch on top of NumPy only — no RL libraries:


| algorithm       | family                                           | uses the transition model?           |
| --------------- | ------------------------------------------------ | ------------------------------------ |
| Value Iteration | model-based, dynamic programming                 | yes, the exact 0.8 / 0.1 / 0.1 model |
| Q-Learning      | model-free, off-policy TD                        | no, samples only                     |
| SARSA(lambda)   | model-free, on-policy TD with eligibility traces | no, samples only                     |


A fourth study reuses the learned Q-table on two perturbed mazes under four
transfer schemes, and measures where transfer helps, does nothing, and hurts.

## Setup

**Windows**

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Only NumPy, pandas, Matplotlib and pygame are used. Run every command from the
repository root; all paths in the code are relative to it.

## Reproducing every result

The commands below regenerate everything in `results/` from scratch. Each writes
into its own per-algorithm subfolder, so the stages are independent and can be
re-run individually.

```bash
python main.py info                        # config, maze stats, energy budgets
python main.py generate                    # write environments/maps/maze_40305054.json
python main.py test                        # unit tests

python experiments/run_value_iteration.py  # optimal reference + gamma sweep
python experiments/run_q_learning.py       # 2 reward modes x 5 seeds + epsilon-decay study
python experiments/run_sarsa_lambda.py     # same, plus a lambda sweep per reward mode
python experiments/run_comparison.py       # policy agreement vs Value Iteration
python experiments/run_transfer.py         # 2 targets x 4 transfer scenarios
python experiments/analysis.py             # all figures and comparison tables

python experiments/run_experiments.py      # everything above in one go
python experiments/run_experiments.py --quick        # tiny smoke run (~1 min)
python experiments/run_experiments.py --skip-sweeps  # main runs only, no studies
```

`run_comparison.py` and `analysis.py` read saved models and raw data, so run them
after the three algorithm stages. Every run is seeded, so repeating any command
reproduces the same numbers. The maze is generated once and saved; all algorithms
then load that same file.

## Interactive viewer

```bash
python main.py gui                                  # or: python gui/app.py
python gui/app.py --algorithm q_learning --environment similar
python gui/app.py --algorithm sarsa_lambda --mode train
python gui/app.py --record                          # PNG frames to results/videos/
```

The viewer draws the walls, floor, penalty cells, start, key, locked door, goal
and agent, with the remaining energy as a bar, the key status, a flash on the
cell where the agent bumped a wall, and a banner naming the outcome when the
episode ends. Three selectors can be changed while it runs:

- **algorithm** — Value Iteration, Q-Learning or SARSA(lambda);
- **environment** — the source maze, target A (similar) or target B (different);
- **mode** — `eval` replays a greedy policy, `train` learns live.

In `eval` mode a trained model is loaded from `results/models/` if one exists, so
the window shows the policy the experiments actually produced. Only a model
trained on the maze currently selected is used: on a transfer target that means
the saved transfer Q-tables, and Value Iteration simply re-solves the target on
the spot (it takes ~20 ms). Where nothing applicable exists — SARSA(lambda) on a
target, say — the panel says so and invites you to press `t` and train it live
instead of quietly showing a policy from a different maze. In `train` mode the
agent learns from an empty table through the same `run_episode` code the batch
experiments use, and each episode is replayed step by step, so exploration
visibly narrows as epsilon decays. A progress bar tracks episodes completed, and
`f` trains without animating when you would rather just watch the success rate
climb.


| key         | action                                              |
| ----------- | --------------------------------------------------- |
| `space`     | start / pause / continue                            |
| `n`         | single step                                         |
| `r`         | reset the episode                                   |
| `1` `2` `3` | Value Iteration / Q-Learning / SARSA(lambda)        |
| `e`         | cycle environment: source -> similar -> different   |
| `t`         | switch between train and eval                       |
| `m`         | switch sparse <-> shaped rewards                    |
| `v`         | value heat map                                      |
| `p`         | policy arrows                                       |
| `b`         | trail                                               |
| `f`         | train without animating                             |
| `+` `-`     | animation speed                                     |
| `g`         | browse the saved figures, arrow keys to page through |
| `c`         | start / stop recording frames                       |
| `esc` / `q` | quit                                                |


The live panel shows the episode and step counters, the running return, key
status, the success rate over the last 50 episodes, the event counters (wall hits,
penalty cells, blocked door attempts, slips), and every hyperparameter in use:
gamma, alpha, epsilon and its schedule, energy bins, lambda and the trace type
for SARSA, theta and the sweep count for Value Iteration.

Pressing `g` opens a gallery of everything in `results/figures/`, so the value
heat maps, policy maps, visit counts, policy-difference maps and transfer
Q-difference maps are all reachable from inside the GUI.


Recorded frames land in `results/videos/<algorithm>/run_<timestamp>/`. To turn
them into a video with an external tool:

```bash
ffmpeg -framerate 30 -i frame_%05d.png -pix_fmt yuv420p run.mp4
```



## The environment



### Seed rule

```python
student_id = "40305054"
base_seed  = int(student_id[-2])       # 5
maze_size  = 15 + (base_seed % 4)      # 16
```

`base_seed` fixes the size; the layout is drawn from a NumPy `default_rng` seeded
with the full student id, so the map is byte-identical on every machine.

### Generated maze


| property       | value                                         |
| -------------- | --------------------------------------------- |
| size           | 16 x 16                                       |
| walls          | 118 / 256 = **46.1%** (minimum required: 15%) |
| passable cells | 138                                           |
| penalty cells  | **8** (minimum required: 5)                   |
| start          | `(1, 1)`                                      |
| key            | `(11, 15)`                                    |
| locked door    | `(15, 12)`                                    |
| goal           | `(13, 15)`, inside a sealed 3x3 vault         |


The goal sits in a corner room whose every other border cell is a wall, so the
door is the only way in. BFS validates that the key is reachable from the start
while the door is shut, that the goal is **not** reachable while it is shut, and
that the goal is reachable from the key once it opens. A unit test asserts all
three properties, plus that no non-wall cell other than the door touches the
vault.

### State, actions, dynamics

```
state = (row, col, has_key, energy_remaining)
```

Actions are up / right / down / left. The intended move happens with probability
**0.8**, and each of the two perpendicular slips with probability **0.1**; the
agent never moves backwards. Walking into a wall, into the outer boundary, or
into a still-locked door leaves the agent in place — and still costs energy.

`MazeEnv.transitions(state, action)` exposes this model exactly; Value Iteration
plans with it, while the two learners only ever call `step()` and so respect it
implicitly. A test checks that 20,000 sampled steps match the analytic
distribution.

### Episode budgets


| quantity          | value        | where it comes from                     |
| ----------------- | ------------ | --------------------------------------- |
| `d(start -> key)` | 36           | BFS with the door shut                  |
| `d(key -> goal)`  | 54           | BFS with the door open                  |
| optimal path      | **90** steps | 36 + 54                                 |
| `max_steps`       | **414**      | `max(200, 3 x 138 passable cells)`      |
| `max_energy`      | **225**      | `ceil(2.5 x 90)`, capped at `max_steps` |


Both are stored in `experiments/configs/default.json` (as `null`, meaning
"derive from the maze") and in the saved map file. `max_energy <= max_steps` is
enforced so energy is always the binding constraint, which keeps the MDP that
Value Iteration solves identical to the episodes the learners experience.

An episode ends on success (goal reached with the key), on energy exhaustion, or
on `max_steps`. Under the optimal policy the agent finishes in ~117 steps and
still has ~108 energy left, so the budget is a real constraint but not a
crippling one.

### Reward modes

Both modes are implemented and every experiment runs on both.

**Sparse** — only the three base terms:


| term      | value |
| --------- | ----- |
| step cost | -0.05 |
| key       | +10   |
| goal      | +100  |


**Shaped** — the same base terms plus:


| term                      | value                                      |
| ------------------------- | ------------------------------------------ |
| progress shaping          | `0.5 x (distance_before - distance_after)` |
| wall / boundary collision | -1                                         |
| locked-door attempt       | -1                                         |
| penalty cell              | -5                                         |
| energy exhausted          | -20                                        |


The shaping target is the key while `has_key == 0` and the goal afterwards,
measured as BFS distance, so moving closer is rewarded and moving away is
penalised. On the single transition that picks up the key the target switches, so
shaping is suppressed there and the `+10` covers it.

Note that in sparse mode a wall bump earns only the step cost. That follows the
assignment's reward table, which lists the collision, penalty-cell and
locked-door penalties as shaped-mode additions; bumping a wall is still harmful
because it wastes a unit of energy.


## Project layout

```
RL_FinalProject_40305054/
├── environments/
│   ├── maze.py                 # the MDP: state, dynamics, both reward modes
│   ├── generator.py            # seeded generation, vault sealing, BFS validation
│   ├── variants.py             # the two perturbed transfer targets
│   └── maps/                   # the saved mazes every algorithm loads
├── agents/
│   ├── base.py                 # tabular scaffolding, energy binning, logging, agreement
│   ├── value_iteration.py      # exact backward induction + residual check
│   ├── q_learning.py           # off-policy TD control
│   └── sarsa_lambda.py         # on-policy TD with sparse eligibility traces
├── transfer/transfer_learning.py   # scratch / full / scaled / selective
├── gui/
│   ├── renderer.py             # vector drawing, no image assets needed
│   └── app.py                  # interactive viewer, live training, figure gallery
├── experiments/
│   ├── common.py               # config, maze, agent factory, result writing
│   ├── run_value_iteration.py  # one runner per algorithm
│   ├── run_q_learning.py
│   ├── run_sarsa_lambda.py
│   ├── run_comparison.py       # cross-algorithm table + policy agreement
│   ├── run_transfer.py
│   ├── run_experiments.py      # runs them all, then the analysis
│   ├── analysis.py             # figures and comparison tables
│   └── configs/default.json    # every hyperparameter lives here
├── results/
│   ├── raw_data/<algorithm>/   # per-episode CSVs, JSON logs, summary tables
│   ├── models/<algorithm>/     # saved value functions and Q-tables (.npz)
│   ├── figures/<algorithm>/
│   └── videos/<algorithm>/
├── tests/                      # unit tests for the env, generator, agents,
│                               # logging, variants and transfer scenarios
├── paths.py                    # repo-relative path helpers
├── main.py                     # single entry point
├── requirements.txt
└── README.md
```