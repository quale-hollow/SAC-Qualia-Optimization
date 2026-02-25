import random
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path

import torch

from sac_implementations.SAC_Qualia_Continuous import Args as ContinuousArgs
from sac_implementations.SAC_Qualia_Discrete import Args as DiscreteArgs

# usage:
#   python sac_experiment.py [environment] [results_directory] [method] [omega] [num_trials]

# --------------------------------------------------
# Default configs
# --------------------------------------------------

# Default config for Pendulum (shorter, cheaper runs)
PendulumArgs = ContinuousArgs(
    env_id="Pendulum-v1",
    total_timesteps=200_000,
    buffer_size=int(1e5),
    learning_starts=1_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    policy_lr=3e-4,
    q_lr=1e-3,
    policy_frequency=2,
    target_network_frequency=1,
    alpha=0.2,
    autotune=True,
    qualia_method=None,
    qualia_omega=0.0,
)

# Default config for CartPole (discrete SAC)
CartPoleArgs = DiscreteArgs(
    env_id="CartPole-v1",
    total_timesteps=500_000,
    buffer_size=int(1e5),
    learning_starts=1_000,
    batch_size=64,
    gamma=0.99,
    tau=1.0,
    policy_lr=3e-4,
    q_lr=3e-4,
    update_frequency=4,
    target_network_frequency=8_000,
    alpha=0.2,
    autotune=True,
    target_entropy_scale=0.89,
    qualia_method=None,
    qualia_omega=0.0,
)

HalfCheetahArgs = ContinuousArgs(
    env_id="HalfCheetah-v4",
    total_timesteps=1_000_000,
    buffer_size=int(1e6),
    learning_starts=5_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    policy_lr=3e-4,
    q_lr=1e-3,
    policy_frequency=2,
    target_network_frequency=1,
    alpha=0.2,
    autotune=True,
    qualia_method=None,
    qualia_omega=0.0,
)


def run_experiment(environment, results_dir, method, omega, num_trials):
    # --------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------
    methods = ["control", "VPER_actor_relu", "VPER_both_relu",
                "VPER_actor_exp", "VPER_both_exp", "VWAU"]
    if method not in methods:
        raise ValueError(f"Method must be one of {methods}")

    # Supported envs and which implementation they use
    continuous_envs = ["HalfCheetah-v4", "Pendulum-v1"]
    discrete_envs = ["CartPole-v1"]

    valid_envs = continuous_envs + discrete_envs
    if environment not in valid_envs:
        raise ValueError(f"Environment must be one of {valid_envs}")

    # --------------------------------------------------
    # 2. Select SAC variant + base Args per environment
    # --------------------------------------------------
    if environment in continuous_envs:
        import sac_implementations.SAC_Qualia_Continuous as sac
        ArgsClass = ContinuousArgs

        if environment == "Pendulum-v1":
            base_args = ContinuousArgs(**vars(PendulumArgs))
        elif environment == "HalfCheetah-v4":
            base_args = ContinuousArgs(**vars(HalfCheetahArgs))

    else:  # discrete envs
        import sac_implementations.SAC_Qualia_Discrete as sac
        ArgsClass = DiscreteArgs

        if environment == "CartPole-v1":
            base_args = DiscreteArgs(**vars(CartPoleArgs))
        else:
            # Future discrete envs could go here
            base_args = DiscreteArgs()
            base_args.env_id = environment

    # Set qualia params on the base config
    if method != "control":
        base_args.qualia_method = method     
        base_args.qualia_omega = float(omega)
    else:
        base_args.qualia_method = None        # plain SAC
        base_args.qualia_omega = 0.0

    # Create base results dir
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    omega = float(omega)
    num_trials = int(num_trials)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(num_trials)]

    # Directory to save results for this condition
    condition_dir = results_dir / method / f"omega_{omega}"
    condition_dir.mkdir(parents=True, exist_ok=True)

    print("CUDA available:", torch.cuda.is_available(), flush=True)
    print("Environment: ", environment, flush=True)
    print("Method: ", method, flush=True)
    print("Omega: ", omega, flush=True)
    print(f"Results will be saved to {condition_dir}", flush=True)
    print("Start Time: ", datetime.now().strftime("%H:%M"), flush=True)

    # --------------------------------------------------
    # 3. Trials loop
    # --------------------------------------------------
    for i, seed in enumerate(seeds):
        trial_start = datetime.now()
        print(f"Beginning Trial {i+1} of {num_trials}", flush=True)

        # Fresh Args instance per trial (avoid mutating shared object)
        args = ArgsClass(**vars(base_args))
        args.seed = seed
        args.env_id = environment  # just to be explicit

        # Returns and replay-buffer policy ratios (for qualia analysis)
        returns, ratios = sac.learn(args)

        # Convert to numpy
        returns = np.array(returns, dtype=np.float32)
        ratios = np.array(ratios, dtype=np.float32)

        timestamp = datetime.now().strftime("%H%M%S")
        out_path = condition_dir / f"{i}_{timestamp}.npz"
        np.savez_compressed(out_path, returns=returns, ratios=ratios)

        print(
            f"Finished Trial {i+1} of {num_trials} in {datetime.now() - trial_start}",
            flush=True,
        )

    print("End Time: ", datetime.now().strftime("%H:%M"), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise ValueError(
            "Usage: python sac_experiment.py [environment] [results_directory] [method] [omega] [num_trials]"
        )

    _, environment, results_dir, method, omega, num_trials = sys.argv
    run_experiment(environment, results_dir, method, omega, num_trials)