import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from sac_implementations.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from sac_implementations.buffers import ReplayBuffer, PrioritizedReplayBuffer
import copy

@dataclass
class Args:
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances"""

    # Algorithm specific arguments
    env_id: str = "BeamRiderNoFrameskip-v4"
    """Environment id. Works for Atari + CartPole-v1."""
    total_timesteps: int = 5_000_000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    batch_size: int = 64
    """the batch size of sample from the replay memory"""
    learning_starts: int = 20_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""

    qualia_method: str = None   # "control", "PER", "VPER", "VPER_actor", "VBAL"
    qualia_omega: float = 0.0


def is_atari_env_id(env_id: str) -> bool:
    """Crude but effective check: Atari NoFrameskip envs usually contain 'NoFrameskip'."""
    return env_id in ["Pong-v5",  "BeamRiderNoFrameskip-v4"]


def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Shared env maker for:
      - Atari (NoFrameskip) → full Atari wrapper stack
      - CartPole / other vector obs → just RecordEpisodeStatistics
    """
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        # Atari-style preprocessing only if this is an Atari env
        if is_atari_env_id(env_id):
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        else:
            # Classic control (e.g., CartPole-v1) – no Atari preprocessing
            env = gym.wrappers.RecordEpisodeStatistics(env)

        env.action_space.seed(seed)
        return env

    return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class MLPSoftQ(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc_q = layer_init(nn.Linear(hidden_dim, n_actions))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_q(x)

class MLPActor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(obs_dim, hidden_dim))
        self.fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.fc_logits = layer_init(nn.Linear(hidden_dim, n_actions))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs
    
    def get_log_probs(self, x, actions):
        logits = self(x)                              # [B, n_actions]
        log_probs_all = F.log_softmax(logits, dim=1)  # [B, n_actions]

        if actions.dim() == 2 and actions.size(1) == 1:
            actions_idx = actions.long()
        else:
            actions_idx = actions.long().unsqueeze(1) # [B, 1]

        log_probs = log_probs_all.gather(1, actions_idx)  # [B, 1]
        return log_probs



class CNNSoftQ(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals

class CNNActor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs
    

    def get_log_probs(self, x, actions):
        logits = self(x)                           # [B, n_actions]
        log_probs_all = F.log_softmax(logits, dim=1)  # [B, n_actions]

        # Normalize action shape to [B, 1]
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        elif actions.dim() == 2 and actions.shape[1] != 1:
            # In case something odd happens, make it [B,1]
            actions = actions.view(-1, 1)

        # Gather log-prob of the chosen actions
        log_probs = log_probs_all.gather(1, actions.long())
        return log_probs

    

def build_discrete_nets(envs, use_cnn: bool):
    obs_shape = envs.single_observation_space.shape
    n_actions = envs.single_action_space.n

    if use_cnn:
        actor = CNNActor(envs)
        qf1 = CNNSoftQ(envs)
        qf2 = CNNSoftQ(envs)
    else:
        obs_dim = obs_shape[0]
        actor = MLPActor(obs_dim, n_actions)
        qf1 = MLPSoftQ(obs_dim, n_actions)
        qf2 = MLPSoftQ(obs_dim, n_actions)

    return actor, qf1, qf2



def learn(args: Args, run_name=None):

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), \
        "Only discrete action space is supported"

    # Build Networks Depending on Env (support for Atari, classic control)
    use_cnn = is_atari_env_id(args.env_id)
    actor, qf1, qf2 = build_discrete_nets(envs, use_cnn)
    actor = actor.to(device)
    qf1 = qf1.to(device)
    qf2 = qf2.to(device)

    qf1_target = copy.deepcopy(qf1).to(device)
    qf2_target = copy.deepcopy(qf2).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)

    # Automatic entropy tuning
    if args.autotune:
        n_actions = envs.single_action_space.n
        target_entropy = args.target_entropy_scale * torch.log(
            torch.tensor(float(n_actions), device=device)
        )
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
    else:
        alpha = args.alpha

    # Set qualia method flags (avoid future repeated string comparisons)
    VPER = args.qualia_method == "VPER"             # valence-prioritized experience replay (critic update)
    VBAL = args.qualia_method == "VBAL"             # valence biased actor/attentional learning


    if VPER:
        rb = PrioritizedReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=envs.num_envs,
            handle_timeout_termination=False,
            alpha=0.6,                          # typical value
            beta=0.4,                          # annealed from 0.4 to 1.0 (a typical approach)
            eps=1e-6
        )
    
    else:
        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=envs.num_envs,
            handle_timeout_termination=False,
        )


    # QUALIA LOGGING SETUP
    ret_by_timestep = []
    replay_buffer_ratios = []

    # Start training loop
    obs, _ = envs.reset(seed=args.seed)
    start_time = time.time()

    for global_step in range(args.total_timesteps):

        # ----- Collect action -----
        if global_step < args.learning_starts:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions, _, _ = actor.get_action(torch.as_tensor(obs, device=device, dtype=torch.float32))
            actions = actions.detach().cpu().numpy()

        # Step env
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Episodic return logging (same style as continuous)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None and "episode" in info:
                    ret_by_timestep.append(
                        [global_step, info["episode"]["r"].item()]
                    )

        # Save data to replay buffer; handle final_observation
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs


        # Training
        if global_step > args.learning_starts:
            if global_step % args.update_frequency == 0:
            
                # 1. SAMPLING FROM REPLAY BUFFER FOR CRITIC UPDATE          
                if VPER:
                    # VPER: critic uses prioritized sampling + IS weights
                    rb.beta = min(
                        1.0, rb.beta_start + global_step * (1.0 - rb.beta_start) / args.total_timesteps
                    )
                    data, idxs, is_weights = rb.sample_prioritized(args.batch_size)
            
                else:
                    # Plain ReplayBuffer (control, VBAL, etc.)
                    data = rb.sample(args.batch_size)
                    idxs, is_weights = None, None


                # ============================================================
                # CRITIC UPDATE
                # ============================================================
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)


                # --- CRITIC LOSS (with IS weights only for PER/VPER critic) ---
                if VPER:
                    qf1_loss_elementwise = F.mse_loss(qf1_a_values, next_q_value, reduction='none')
                    qf2_loss_elementwise = F.mse_loss(qf2_a_values, next_q_value, reduction='none')
                    qf1_loss = (qf1_loss_elementwise * is_weights.view(-1)).mean()
                    qf2_loss = (qf2_loss_elementwise * is_weights.view(-1)).mean()
                # critic loss for non-prioritized experience replay for critic (control, VBAL, VPER_actor)
                else:
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)


                qf_loss = qf1_loss + qf2_loss
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()


                # === GET TD ERRORS, UPDATE REPLAY BUFFER PRIORITIES ===
                with torch.no_grad():
                    td_error1 = next_q_value - qf1_a_values
                    td_error2 = next_q_value - qf2_a_values
                    td_error = 0.5 * (td_error1 + td_error2) 
            
                    if VPER:
                        # normal non-valence "surprisal" priority
                        surprisal = td_error.abs()
                        # valence based priority
                        valence = torch.relu(td_error)
                        # mixed priority --- omega is how much weight is moved to valence -- 0.0 is normal PER, 1.0 is fully valence
                        priorities = ((1.0 - args.qualia_omega) * surprisal) + (args.qualia_omega * valence)

                        rb.update_priorities(idxs, priorities)

                    if VBAL:
                        # redistribute "attention" to each sampled state based on TD errors
                        # normalize to fix reward function dependence
                        batch_mean = td_error.mean()
                        batch_std = td_error.std() + 1e-6
                        z_scored_td = (td_error - batch_mean) / batch_std
                        scaled_td = args.qualia_omega * z_scored_td
                        scaled_td = scaled_td - scaled_td.max()
                        actor_update_attention = torch.softmax(scaled_td, dim=0)
                        # Restore mean to 1.0 -- so we don't shrink learning rate and preserve "total update attention"
                        actor_update_attention = actor_update_attention * args.batch_size
                        actor_update_attention = actor_update_attention.detach()
                    else:
                        actor_update_attention = None


                # ============================================================
                # ACTOR UPDATE
                # ============================================================
        
                # Pre-update log_probs for logging
                with torch.no_grad():
                    pre_update_log_probs = actor.get_log_probs(data.observations, data.actions)

                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)


                # average over actions for each state first
                actor_loss_per_state = (action_probs * ((alpha * log_pi) - min_qf_values)).mean(dim=1)

                # Get (weighted) actor loss
                if VBAL:
                    # VBAL: redistribute fixed total attention over samples
                    # multiply average loss for each state by resaled TD error from the state/action sample
                    actor_loss = (actor_loss_per_state * actor_update_attention).mean()

                else:
                    # standard SAC
                    actor_loss = actor_loss_per_state.mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # ----- Alpha / temperature tuning -----
                if args.autotune:
                    # reuse action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha.exp() * (log_pi + target_entropy).detach())).mean()
                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

                
                # LOG THE RATIO BETWEEN OLD AND NEW POLICY OVER THE REPLAY BUFFER ACTIONS
                with torch.no_grad():
                    post_update_log_probs = actor.get_log_probs(data.observations, data.actions)

                log_ratio = post_update_log_probs - pre_update_log_probs
                replay_buffer_ratio = torch.exp(log_ratio).mean().item()
                replay_buffer_ratios.append([global_step, replay_buffer_ratio])



            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)


    envs.close()
    return ret_by_timestep, replay_buffer_ratios