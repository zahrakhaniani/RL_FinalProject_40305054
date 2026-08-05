# RL_FinalProject_40305054

Reinforcement learning on a 16x16 stochastic maze with a key, a locked door,
penalty cells and a limited energy budget. Three algorithms are implemented from
scratch on top of NumPy only — no RL libraries:

| algorithm | family | uses the transition model? |
| --- | --- | --- |
| Value Iteration | model-based, dynamic programming | yes, the exact 0.8 / 0.1 / 0.1 model |
| Q-Learning | model-free, off-policy TD | no, samples only |
| SARSA(lambda) | model-free, on-policy TD with eligibility traces | no, samples only |

A fourth study transfers a learned Q-table to a second, differently shaped maze.

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
python main.py test                        # 46 unit tests

python experiments/run_value_iteration.py  # model-based baseline (optimal reference)
python experiments/run_q_learning.py       # 2 reward modes x 5 seeds
python experiments/run_sarsa_lambda.py     # same, plus a lambda sweep per reward mode
python experiments/run_transfer.py         # scratch vs warm start vs policy reuse
python experiments/analysis.py             # all figures and comparison tables

python experiments/run_experiments.py      # everything above in one go (~4.5 min)
python experiments/run_experiments.py --quick   # tiny smoke run (~15 s)
```

Every run is seeded, so repeating any command reproduces the same numbers. The
maze is generated once and saved; all algorithms then load that same file.

## Interactive viewer

```bash
python main.py gui                                  # or: python gui/app.py
python gui/app.py --algorithm q_learning --reward-mode sparse
python gui/app.py --record                          # PNG frames to results/videos/
```

The viewer animates greedy episodes under the real stochastic dynamics, so you
can watch the agent slip. It loads trained models from `results/models/` when
they exist and trains on the spot when they do not.

| key | action |
| --- | --- |
| `space` | play / pause |
| `n` | single step |
| `r` | restart the episode |
| `1` `2` `3` | Value Iteration / Q-Learning / SARSA(lambda) |
| `m` | switch sparse <-> shaped rewards |
| `v` | optimal-value heat map |
| `p` | policy arrows |
| `t` | trail |
| `+` `-` | speed |
| `c` | start / stop recording frames |
| `esc` | quit |

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

| property | value |
| --- | --- |
| size | 16 x 16 |
| walls | 118 / 256 = **46.1%** (minimum required: 15%) |
| passable cells | 138 |
| penalty cells | **8** (minimum required: 5) |
| start | `(1, 1)` |
| key | `(11, 15)` |
| locked door | `(15, 12)` |
| goal | `(13, 15)`, inside a sealed 3x3 vault |

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

| quantity | value | where it comes from |
| --- | --- | --- |
| `d(start -> key)` | 36 | BFS with the door shut |
| `d(key -> goal)` | 54 | BFS with the door open |
| optimal path | **90** steps | 36 + 54 |
| `max_steps` | **414** | `max(200, 3 x 138 passable cells)` |
| `max_energy` | **225** | `ceil(2.5 x 90)`, capped at `max_steps` |

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

| term | value |
| --- | --- |
| step cost | -0.05 |
| key | +10 |
| goal | +100 |

**Shaped** — the same base terms plus:

| term | value |
| --- | --- |
| progress shaping | `0.5 x (distance_before - distance_after)` |
| wall / boundary collision | -1 |
| locked-door attempt | -1 |
| penalty cell | -5 |
| energy exhausted | -20 |

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

| algorithm | rewards | success | return | steps | energy left |
| --- | --- | --- | --- | --- | --- |
| Value Iteration | sparse | **1.000** | 104.2 | 116.6 | 108.4 |
| Value Iteration | shaped | **1.000** | 108.6 | 116.7 | 108.3 |
| Q-Learning | sparse | 0.011 | -0.2 | 224.8 | 0.2 |
| Q-Learning | shaped | **1.000** | 97.7 | 174.1 | 50.9 |
| SARSA(lambda) | sparse | 0.983 | 98.3 | 200.1 | 24.9 |
| SARSA(lambda) | shaped | **1.000** | 98.3 | 177.4 | 47.6 |

Three things stand out, and `report.md` discusses them in detail:

1. Value Iteration is optimal and effectively instant (0.02 s) because it owns
   the model and because energy makes the MDP acyclic — one backward pass over
   the energy dimension is exact. The Bellman residual is ~1e-14.
2. With shaped rewards both learners reach a 100% success rate but take ~50%
   more steps than optimal; they find a reliable route, not the best one.
3. With sparse rewards Q-Learning collapses to ~1% while SARSA(lambda) still
   reaches 98%. The eligibility traces carry the single terminal reward back
   across the whole episode, which one-step Q-Learning cannot do at this horizon.

### Figures

```
results/figures/
├── maze_layout.png                       # the maze + the optimal trajectory
├── value_iteration/
│   ├── convergence.png                   # value change per energy level
│   ├── value_landscape.png               # V* before and after the key
│   └── policy_arrows.png                 # the optimal policy
├── q_learning/learning_curves.png
├── sarsa_lambda/
│   ├── learning_curves.png
│   └── lambda_sweep.png                  # lambda x reward mode
├── transfer/transfer_curves.png
└── comparison/
    ├── algorithm_comparison.png
    └── learning_vs_optimal.png
```

## Project layout

```
RL_FinalProject_40305054/
├── environments/
│   ├── maze.py                 # the MDP: state, dynamics, both reward modes
│   ├── generator.py            # seeded generation, vault sealing, BFS validation
│   └── maps/                   # the saved maze every algorithm loads
├── agents/
│   ├── base.py                 # tabular scaffolding, energy binning, evaluation
│   ├── value_iteration.py      # exact backward induction + residual check
│   ├── q_learning.py           # off-policy TD control
│   └── sarsa_lambda.py         # on-policy TD with sparse eligibility traces
├── transfer/transfer_learning.py
├── gui/
│   ├── renderer.py             # vector drawing, no image assets needed
│   └── app.py                  # interactive viewer and frame recorder
├── experiments/
│   ├── common.py               # config, maze, agent factory, result writing
│   ├── run_value_iteration.py  # one runner per algorithm
│   ├── run_q_learning.py
│   ├── run_sarsa_lambda.py
│   ├── run_transfer.py
│   ├── run_experiments.py      # runs them all, then the analysis
│   ├── analysis.py             # figures and comparison tables
│   └── configs/default.json    # every hyperparameter lives here
├── results/
│   ├── raw_data/<algorithm>/   # per-run JSON logs + summary CSVs
│   ├── models/<algorithm>/     # saved value functions and Q-tables (.npz)
│   ├── figures/<algorithm>/
│   └── videos/<algorithm>/
├── tests/                      # 46 unit tests
├── paths.py                    # repo-relative path helpers
├── main.py                     # single entry point
├── report.md
├── requirements.txt
└── README.md
```

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
