from utils.plotting_utils import plot_curves
import sys

# Usage: python plot_results.py [arg] 
# arg: 'halfcheetah', 'cartpole', 'pong', or 'all'. If no arg is provided, 'all' is used by default.

# This script is used to plot the results of the experiments and to print tex tables for final RQO and performance

if len(sys.argv) < 2:
    arg = 'all'
else:
    arg = sys.argv[1]


# if arg not in ['all', 'cartpole', 'pendulum', 'halfcheetah']:
    # raise ValueError("Invalid argument. Please use 'all', 'cartpole', 'pendulum', or 'halfcheetah'.")


if arg in ['all', 'cartpole']:

    methods = ["VPER_actor_relu", "VPER_both_relu", "VPER_actor_exp", "VPER_both_exp"]
    omegas = [
        [0.2, 0.4, 0.6, 0.8, 1.0], 
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.2, 0.4, 0.6, 0.8, 1.0],
    ]

    # One color per method (control is usually plotted separately inside plot_curves
    # or defaults to C0 if you coded it that way)
    line_colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']

    names = [
        "Valence-Prioritized Experience Replay (VPER) -- Linear Priorities, Actor Update Only",
        "Valence-Prioritized Experience Replay (VPER) -- Linear Priorities, Actor/Critic Update",
        "Valence-Prioritized Experience Replay (VPER) -- Exp. Priorities, Actor Update Only",
        "Valence-Prioritized Experience Replay (VPER) -- Exp. Priorities, Actor/Critic Update",
    ]

    plot_curves(
        methods=methods,
        names=names,
        omegas=omegas,
        line_colors=line_colors,
        path='results/cartpole_results',
        env="CartPole-v1",
        save_dir='plots/cartpole/',
        qe_lims=[0.0, 0.00005],
        NUM_TRIALS=100,
        NUM_BINS=100,
    )


if arg in ['all', 'pendulum']:

    methods = ["VPER_actor_relu", "VPER_both_relu", "VPER_actor_exp", "VPER_both_exp"]
    omegas = [
        [0.2, 0.4, 0.6, 0.8, 1.0], 
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.2, 0.4, 0.6, 0.8, 1.0],
    ]

    # One color per method (control is usually plotted separately inside plot_curves
    # or defaults to C0 if you coded it that way)
    line_colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    names = [
        "Valence-Prioritized Experience Replay (VPER) -- Linear Priorities, Actor Update Only",
        "Valence-Prioritized Experience Replay (VPER) -- Linear Priorities, Actor/Critic Update",
        "Valence-Prioritized Experience Replay (VPER) -- Exp. Priorities, Actor Update Only",
        "Valence-Prioritized Experience Replay (VPER) -- Exp. Priorities, Actor/Critic Update",
    ]

    plot_curves(
        methods=methods,
        names=names,
        omegas=omegas,
        line_colors=line_colors,
        path='results/pendulum_results',
        env="Pendulum-v1",
        save_dir='plots/pendulum/',
        qe_lims=[0, 700],
        NUM_TRIALS=100,
        NUM_BINS=100,
    )




if arg in ['all', 'halfcheetah']:

    methods = ["VPER_actor_relu", "VPER_both_relu", "VPER_actor_exp", "VPER_both_exp"]
    omegas = [
        [0.2, 0.4, 0.6, 0.8, 1.0], 
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [0.2, 0.4, 0.6, 0.8, 1.0],
    ]

    # One color per method (control is usually plotted separately inside plot_curves
    # or defaults to C0 if you coded it that way)
    line_colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    names = [
        "Valence-Prioritized Experience Replay (VPER) -- Linear Priorities, Actor Update Only",
        "Valence-Prioritized Experience Replay (VPER) -- Linear Priorities, Actor/Critic Update",
        "Valence-Prioritized Experience Replay (VPER) -- Exp. Priorities, Actor Update Only",
        "Valence-Prioritized Experience Replay (VPER) -- Exp. Priorities, Actor/Critic Update",
    ]

    plot_curves(
        methods=methods,
        names=names,
        omegas=omegas,
        line_colors=line_colors,
        path='results/halfcheetah_results',
        env="HalfCheetah-v4",
        save_dir='plots/halfcheetah/',
        qe_lims=[0, 300],
        NUM_TRIALS=100,
        NUM_BINS=100,
    )