# RL Maze Solver

A reinforcement learning project that solves maze environments using three different algorithms:
- **Value Iteration** (model-based)
- **Q-Learning** (model-free)
- **SARSA(lambda)** (model-free with eligibility traces)

## Project Structure

```
RL_FinalProject_/
├── environments/
│   ├── maze.py          # Maze environment (states, actions, rewards)
│   ├── generator.py     # Maze generation algorithms
│   └── maps/            # Custom maze map files
├── agents/
│   ├── value_iteration.py   # Model-based Value Iteration
│   ├── q_learning.py        # Model-free Q-Learning
│   └── sarsa_lambda.py      # Model-free SARSA(lambda)
├── transfer/
│   └── transfer_learning.py # Transfer learning between mazes
├── gui/
│   ├── app.py           # Pygame GUI application
│   └── renderer.py      # Maze visualization
├── experiments/
│   ├── run_experiments.py    # Run experiments
│   ├── analysis.py           # Result analysis & plotting
│   └── configs/
│       └── default.json      # Default experiment config
├── results/
│   ├── raw_data/        # Experiment result JSON files
│   ├── models/          # Saved agent models
│   ├── figures/         # Generated plots
│   └── videos/          # Visualization recordings
├── tests/
│   └── test_maze.py     # Unit tests
├── main.py              # Main entry point
├── requirements.txt     # Python dependencies
└── README.md
```

## Setup

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

To deactivate the virtual environment when done:
```bash
deactivate
```

## Usage

### Run the main menu
```bash
python main.py
```

### Run GUI -> you can actually play or run an agent to solve it for you + compare all agents 
```bash
python gui/app.py
```

### Run experiments
```bash
python experiments/run_experiments.py
```

### Run tests
```bash
python tests/test_maze.py
```

## Algorithms

### Value Iteration
Model-based algorithm that computes optimal values for all states by iteratively applying the Bellman equation until convergence.

### Q-Learning
Model-free algorithm that learns action values (Q-values) through interaction with the environment using temporal difference learning.

### SARSA(lambda)
Model-free on-policy algorithm that uses eligibility traces to propagate reward information more efficiently across episodes.

## Configuration

Edit `experiments/configs/default.json` to change:
- Maze size (rows, cols)
- Wall density
- Discount factor (gamma)
- Learning rate (alpha)
- Exploration rate (epsilon)
- Eligibility trace decay (lambda)
- Number of training episodes
