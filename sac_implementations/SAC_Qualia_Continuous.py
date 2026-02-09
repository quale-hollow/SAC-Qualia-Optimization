# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
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
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    qualia_method: str = None # "control", "PER", "VPER", "VPER_actor"
    qualia_omega: float = 0.0


def cuda():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def get_log_probs(self, x, actions):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        eps = 1e-6

        # inverse scaling: go back to [-1, 1]
        y_t = (actions - self.action_bias) / self.action_scale
        y_t = torch.clamp(y_t, -1 + eps, 1 - eps)

        # stable atanh
        x_t = 0.5 * (torch.log1p(y_t) - torch.log1p(-y_t))  # atanh(y_t)

        # Gaussian log-prob in pre-tanh space
        log_prob = normal.log_prob(x_t)

        # tanh-squash + scaling Jacobian term
        jac = torch.clamp(1 - y_t.pow(2), min=eps)
        log_prob -= torch.log(self.action_scale * jac + eps)

        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob




def learn(args: Args, run_name=None):

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    
    

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

    # ============================================================
    # START TRAINING LOOP
    # ============================================================
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ret_by_timestep.append([global_step, info["episode"]["r"].item()])

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:

            # 1. SAMPLING FROM REPLAY BUFFER FOR CRITIC UPDATE          
            if VPER:
                # PER/VPER: critic uses prioritized sampling + IS weights
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
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
               
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)


            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)

            
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
            if global_step % args.policy_frequency == 0:
              
                # Pre-update log_probs for logging
                with torch.no_grad():
                    pre_update_log_probs = actor.get_log_probs(data.observations, data.actions)

                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    actor_loss_elementwise = ((alpha * log_pi) - min_qf_pi)

                    # Get (weighted) actor loss
                    if VBAL:
                        # VBAL: redistribute fixed total attention over samples
                        actor_loss = (actor_loss_elementwise * actor_update_attention).mean()

                    else:
                        # standard SAC
                        actor_loss = actor_loss_elementwise.mean()


                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                

                # LOG THE RATIO BETWEEN OLD AND NEW POLICY OVER THE REPLAY BUFFER ACTIONS
                with torch.no_grad():
                    post_update_log_probs = actor.get_log_probs(data.observations, data.actions)

                log_ratio = post_update_log_probs - pre_update_log_probs
                log_ratio = torch.clamp(log_ratio, -10.0, 10.0) # clamp to avoid massive ratios
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