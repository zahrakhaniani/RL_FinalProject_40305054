"""Single entry point for the project.

    python main.py info                     # config, maze stats, energy budgets
    python main.py generate                 # (re)build the shared maze
    python main.py demo                     # greedy episode per algorithm, in the console
    python main.py train --algorithm q_learning
    python main.py experiments              # full suite, then figures and tables
    python main.py experiments --quick      # small smoke run
    python main.py analyze                  # rebuild figures from saved results
    python main.py gui                      # interactive pygame viewer
    python main.py test                     # unit tests
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
from agents.base import evaluate_policy, rollout
from experiments import common

ALGORITHMS = ("value_iteration", "q_learning", "sarsa_lambda")


def command_info(args) -> None:
    config = common.load_config(args.config)
    env = common.build_env(config, reward_mode="shaped")
    metadata = env.metadata

    print("Configuration")
    print(f"  config file        {config['_config_file']}")
    print(f"  student id         {config['student_id']}")
    print(f"  base_seed          {metadata['base_seed']}  (= int(student_id[-2]))")
    print(f"  maze size          {metadata['size']}x{metadata['size']}"
          f"  (= 15 + base_seed % 4)")
    print(f"  layout seed        {metadata['layout_seed']}")
    print()
    print("Maze")
    print(f"  walls              {metadata['n_wall_cells']}/{metadata['size'] ** 2}"
          f" = {metadata['wall_fraction']:.1%}  (minimum {metadata['min_wall_fraction']:.0%})")
    print(f"  passable cells     {metadata['n_passable_cells']}")
    print(f"  penalty cells      {metadata['n_penalty_cells']}  (minimum 5)")
    print(f"  start / key        {env.start} / {env.key}")
    print(f"  door / goal        {env.door} / {env.goal}")
    print()
    print("Episode budgets")
    print(f"  d(start -> key)    {metadata['d_start_key']}")
    print(f"  d(key -> goal)     {metadata['d_key_goal']}")
    print(f"  optimal path       {metadata['optimal_path_length']} steps")
    print(f"  max_steps          {env.max_steps}"
          f"  (= max(200, 3 x {metadata['n_passable_cells']} passable cells))")
    print(f"  max_energy         {env.max_energy}"
          f"  (= ceil({metadata['energy_slack']} x optimal path), capped at max_steps)")
    print()
    print("Dynamics")
    print(f"  intended action    {env.p_intended}")
    print(f"  each slip          {env.p_slip}")
    print()
    print("Rewards")
    for name, value in env.rewards.items():
        print(f"  {name:<20} {value:>8}")
    print()
    env.render()


def command_generate(args) -> None:
    config = common.load_config(args.config)
    paths.ensure_dirs()
    common.ensure_map(config, force=True, verbose=True)


def command_demo(args) -> None:
    config = common.load_config(args.config)
    paths.ensure_dirs()
    common.ensure_map(config, verbose=False)
    reward_mode = args.reward_mode

    for algorithm in ALGORITHMS:
        common.banner(f"{algorithm} ({reward_mode} rewards)")
        env = common.build_env(config, reward_mode=reward_mode, seed=0)

        if algorithm == "value_iteration":
            agent = common.make_planner(env, config)
            model = common.model_path(algorithm, f"{algorithm}_{reward_mode}.npz")
            if model.exists():
                agent.load(model)
            else:
                agent.train()
        else:
            seeds = config["training"]["seeds"]
            agent = common.make_learner(algorithm, env, config, seed=seeds[0])
            model = next(
                (
                    path
                    for path in (
                        common.model_path(algorithm, f"{algorithm}_{reward_mode}_seed{s}.npz")
                        for s in seeds
                    )
                    if path.exists()
                ),
                None,
            )
            if model is not None:
                agent.load(model)
            else:
                print(f"  no saved model; training {config[algorithm]['episodes']} episodes")
                agent.train(episodes=config[algorithm]["episodes"], eval_every=0)

        trace = rollout(env, agent, seed=config["training"]["eval_seed"])
        metrics = evaluate_policy(env, agent, episodes=200, seed=config["training"]["eval_seed"])
        print(f"  one greedy episode: {trace['steps']} steps, "
              f"return {trace['total_reward']:+.2f}, outcome {trace['outcome']}")
        print(f"  over 200 episodes : success {metrics['success_rate']:.3f}, "
              f"return {metrics['mean_return']:+.2f}, "
              f"steps {metrics['mean_steps']:.1f}, "
              f"energy left {metrics['mean_energy_left']:.1f}")
        env.render()


def command_train(args) -> None:
    config = common.load_config(args.config)
    paths.ensure_dirs()
    common.ensure_map(config, verbose=False)

    if args.algorithm == "value_iteration":
        from experiments import run_value_iteration

        run_value_iteration.run(config, [args.reward_mode])
        return

    record = common.train_learner(
        args.algorithm,
        config,
        reward_mode=args.reward_mode,
        seed=args.seed,
        episodes=args.episodes,
    )
    print(f"  model saved to {record['model_file']}")


def command_experiments(args) -> None:
    from experiments import run_experiments

    forwarded = []
    if args.config:
        forwarded += ["--config", args.config]
    if args.quick:
        forwarded.append("--quick")
    run_experiments.main(forwarded)


def command_analyze(args) -> None:
    from experiments import analysis

    analysis.main(["--config", args.config] if args.config else [])


def command_gui(args) -> None:
    from gui import app

    forwarded = ["--algorithm", args.algorithm, "--reward-mode", args.reward_mode]
    if args.config:
        forwarded += ["--config", args.config]
    if args.record:
        forwarded.append("--record")
    app.main(forwarded)


def command_test(args) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"],
        cwd=paths.ROOT,
    )
    sys.exit(result.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RL maze solver -- student 40305054",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default=None, help="path to a config JSON file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("info", help="print the config, maze stats and budgets")
    subparsers.add_parser("generate", help="regenerate and save the shared maze")

    demo = subparsers.add_parser("demo", help="run one greedy episode per algorithm")
    demo.add_argument("--reward-mode", default="shaped", choices=("sparse", "shaped"))

    train = subparsers.add_parser("train", help="train a single agent")
    train.add_argument("--algorithm", default="q_learning", choices=ALGORITHMS)
    train.add_argument("--reward-mode", default="shaped", choices=("sparse", "shaped"))
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--episodes", type=int, default=None)

    experiments = subparsers.add_parser("experiments", help="run the full suite")
    experiments.add_argument("--quick", action="store_true")

    subparsers.add_parser("analyze", help="rebuild figures and tables")

    gui = subparsers.add_parser("gui", help="launch the interactive viewer")
    gui.add_argument("--algorithm", default="value_iteration", choices=ALGORITHMS)
    gui.add_argument("--reward-mode", default="shaped", choices=("sparse", "shaped"))
    gui.add_argument("--record", action="store_true")

    subparsers.add_parser("test", help="run the unit tests")
    return parser


COMMANDS = {
    "info": command_info,
    "generate": command_generate,
    "demo": command_demo,
    "train": command_train,
    "experiments": command_experiments,
    "analyze": command_analyze,
    "gui": command_gui,
    "test": command_test,
}


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
