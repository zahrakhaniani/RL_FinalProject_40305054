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
├── report.pdf
├── requirements.txt
└── README.md
```



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

## Interactive viewer / GUI

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
winget install ffmpeg
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

## Results

Averaged over 5 seeds, 300 greedy evaluation episodes each. Full tables are in
`results/raw_data/comparison_summary.csv`.


| algorithm       | rewards | success   | return | steps | energy left |
| --------------- | ------- | --------- | ------ | ----- | ----------- |
| Value Iteration | sparse  | **1.000** | 104.2  | 116.6 | 108.4       |
| Value Iteration | shaped  | **1.000** | 108.6  | 116.7 | 108.3       |
| Q-Learning      | sparse  | 0.011     | -0.2   | 224.8 | 0.2         |
| Q-Learning      | shaped  | **1.000** | 97.7   | 174.1 | 50.9        |
| SARSA(lambda)   | sparse  | 0.983     | 98.3   | 200.1 | 24.9        |
| SARSA(lambda)   | shaped  | **1.000** | 98.3   | 177.4 | 47.6        |


Three things stand out, and `report.pdf` discusses them in detail:

1. Value Iteration is optimal and effectively instant (0.02 s) because it owns
  the model and because energy makes the MDP acyclic — one backward pass over
   the energy dimension is exact. The Bellman residual is ~1e-14.
2. With shaped rewards both learners reach a 100% success rate but take ~50%
  more steps than optimal; they find a reliable route, not the best one.
3. With sparse rewards Q-Learning collapses to ~1% while SARSA(lambda) still
  reaches 98%. The eligibility traces carry the single terminal reward back
   across the whole episode, which one-step Q-Learning cannot do at this horizon.

Policy agreement with the Value Iteration reference is 53% of states / 66% of
visits for Q-Learning and 44% / 56% for SARSA(lambda), in shaped mode. The gap
between the two figures is the point: most disagreement sits in states a
converged agent never enters. See `results/figures/comparison/policy_difference.png`.

### Transfer learning

Target A changes 18.4% of the cells and keeps the landmarks; target B changes
35.2% and moves the key. Both are BFS validated. Averaged over 5 seeds, against a
from-scratch baseline on the same target and seed
(`results/raw_data/transfer/transfer_by_scenario.csv`):


| target      | scenario      | final success     | episodes to 80% | verdict      |
| ----------- | ------------- | ----------------- | --------------- | ------------ |
| `similar`   | scratch       | 0.999 ± 0.003     | 1440            | baseline     |
| `similar`   | full          | 0.800 ± 0.447     | **960**         | **negative** |
| `similar`   | scaled β=0.25 | **1.000 ± 0.000** | **1040**        | **positive** |
| `similar`   | selective     | 0.971 ± 0.062     | 1880            | **negative** |
| `different` | scratch       | 0.769 ± 0.287     | 2000            | baseline     |
| `different` | full          | 0.795 ± 0.445     | **1000**        | **positive** |
| `different` | scaled β=0.25 | **0.979 ± 0.034** | 1220            | **positive** |
| `different` | selective     | 0.755 ± 0.424     | **1140**        | **positive** |


No scenario has any zero-shot ability on either target, so the whole effect is in
learning speed and final quality. Transfer pays off on the hard target and
backfires on the easy one: full transfer onto `similar` reaches the threshold 480
episodes sooner but ends at 0.800 ± 0.447, because one seed in five locks into the
stale route and finishes at 0.000. Scaling the prior by β = 0.25 keeps the
speed-up with zero failures. `report.pdf` section 6 works through the numbers, the
mechanism and the negative-transfer case; the omitted β = 0.5 and β = 0.75 rows
are in the CSV.

### Figures

```
results/figures/
├── maze_layout.png                       # the maze + the optimal trajectory
├── maze_target_similar.png               # transfer target A
├── maze_target_different.png             # transfer target B
├── value_iteration/
│   ├── convergence.png                   # value change per energy level
│   ├── value_heatmap.png                 # V* for all 8 energy bins x key status
│   ├── policy_map.png                    # optimal policies for all 8 bins
│   └── gamma_sweep.png                   # gamma in {0.90, 0.95, 0.99}
├── q_learning/
│   ├── learning_curves.png
│   ├── value_heatmap.png                 # max_a Q(s,a), all 8 energy bins
│   ├── policy_map.png                    # greedy policies for all 8 bins
│   ├── visit_counts.png                  # where exploration actually went
│   └── epsilon_schedules.png             # linear vs exponential decay
├── sarsa_lambda/
│   ├── learning_curves.png
│   ├── value_heatmap.png
│   ├── policy_map.png
│   ├── visit_counts.png
│   └── lambda_sweep.png                  # lambda x reward mode
├── transfer/
│   ├── transfer_curves.png               # 4 scenarios x 2 targets
│   ├── environment_changes.png           # what changed + the reuse mask
│   ├── q_difference_similar.png          # max Q before / after / difference
│   └── q_difference_different.png
└── comparison/
    ├── algorithm_comparison.png
    ├── policy_difference.png             # model-free greedy action vs VI
    └── learning_vs_optimal.png
```

Each value and policy figure contains 16 panels: eight energy bins, each shown
with `has_key=0` and `has_key=1`. Q-Learning and SARSA use one table slice per
bin. Value Iteration stores every energy level exactly, so its panel uses the
midpoint energy printed in the title. Policy arrows are omitted on the terminal
goal and on states unreachable with the selected key status.

### Logs

Every training run writes a per-episode CSV to
`results/raw_data/<algorithm>/episodes_<run>.csv`. Each row carries the algorithm,
run label, reward mode, seed and a hash of the config it came from, then:

`episode`, `steps`, `reward`, `mean_return_100`, `epsilon`, `alpha`,
`energy_left`, `outcome`, `goal_success`, `energy_exhausted`,
`max_steps_reached`, `wall_collisions`, `penalty_entries`, `door_blocked`,
`door_crossings`, `key_pickups`, `slips`, `mean_abs_td_error`,
`max_abs_td_error`, `mean_active_traces`, `max_active_traces`.

The last four cover the per-episode delta and eligibility-trace statistics; the
trace columns are zero for Q-Learning, which has no traces.
`results/raw_data/q_learning/q_updates_<run>.csv` holds a thinned sample of
individual Q-updates with `state`, `action`, `reward`, `next_state`, `done`,
`q_old`, `q_new` and `td_error`, so the update rule can be checked by hand.


## Design notes

- **Energy in the tabular state.** Value Iteration uses the exact state, all
`138 x 2 x 226` of them. The two learners would need a table row per energy
level, which experience can never fill, so they bucket energy into
`energy_bins` (8 by default) groups. This is the only difference between what
the planner and the learners see, and it is recorded in every saved model.
- **Truncation is not termination.** Episodes cut off by `max_steps` are
bootstrapped normally rather than treated as absorbing, since the limit is an
experiment-level convenience, not part of the MDP.
- **Eligibility traces are sparse.** Traces decay by `gamma * lambda`, so only a
short window matters. They are kept in compact NumPy arrays and pruned below a
threshold, making a trace update a couple of vector operations instead of a
full-table sweep. A test verifies the sparse store is numerically identical to
a dense trace vector.

