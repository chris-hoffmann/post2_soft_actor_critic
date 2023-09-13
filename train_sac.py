"""
This is a script for training Soft Actor-Critic agents 
in Gym environments with continuous action space.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque
from copy import deepcopy
from enum import Enum
from typing import NamedTuple

import dill
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Data(NamedTuple):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name",
        type=str,
        default="HalfCheetah-v4",
        help="the name of the gym environment",
    )
    parser.add_argument(
        "--time_steps",
        type=int,
        default=100_000,
        help="the total number of time steps of the experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="the seed of the experiment")
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor")
    parser.add_argument("--alpha", type=float, default=0.1, help="the initial temperature value")
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=2_000,
        help="the number of pre-learning steps",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=int(100_000),
        help="the capacity of the replay ring buffer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="the batch size of the samples drawn from the replay buffer",
    )
    parser.add_argument(
        "--max_ep_steps",
        type=int,
        default=1000,
        help="the maximum number of time steps per episode",
    )
    parser.add_argument(
        "--q_lr",
        type=float,
        default=3e-4,
        help="the learning rate of the twin Q-networks (critic)",
    )
    parser.add_argument("--polyak", type=float, default=0.995, help="the target mixing coefficient")
    parser.add_argument(
        "--policy_lr",
        type=float,
        default=3e-4,
        help="the learning rate of the policy network (actor)",
    )
    parser.add_argument(
        "--policy_frequency",
        type=int,
        default=2,
        help="the frequency of policy updates",
    )
    parser.add_argument(
        "--log_std_bounds",
        type=lambda x: tuple([float(e) for e in x.split()]),
        default="-20 2",
        nargs="+",
        help="the bounds of the logarithm of the standard deviation \
            of the policy distribution",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=1000,
        help="the frequency of deterministic policy evaluations",
    )
    parser.add_argument(
        "--video_freq",
        type=int,
        default=10_000,
        help="the frequency of capturing videos if at all (-1 means no video recording)",
    )
    args = parser.parse_args()
    return args


class TwinQNets(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(TwinQNets, self).__init__()
        self.q1 = nn.Sequential(*self._make_layers(state_dim, action_dim, hidden_dim))
        self.q2 = nn.Sequential(*self._make_layers(state_dim, action_dim, hidden_dim))

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([states, actions], 1)
        return self.q1(x), self.q2(x)

    def _make_layers(self, state_dim: int, action_dim: int, hidden_dim: int) -> list[nn.Module]:
        return [
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ]


class TargetTwinQNets(nn.Module):
    def __init__(self, local_model: nn.Module, polyak: float):
        super(TargetTwinQNets, self).__init__()
        self.target_qnets = deepcopy(local_model)
        self._freeze_layers()
        self.polyak = polyak

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.target_qnets(states, actions)

    def _freeze_layers(self) -> None:
        for p in self.target_qnets.parameters():
            p.requires_grad = False

    def soft_update(self, twin_qnets: nn.Module) -> None:
        with torch.no_grad():
            for target_param, model_param in zip(self.target_qnets.parameters(), twin_qnets.parameters()):
                target_param.data.mul_(self.polyak)
                target_param.data.add_((1 - self.polyak) * model_param.data)


class Actor(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_range: tuple[np.ndarray, np.ndarray],
        log_std_bounds=tuple[float, float],
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            *[
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
        )
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_logstd = nn.Linear(hidden_dim, action_dim)
        self.min_log_std, self.max_log_std = log_std_bounds
        self.register_buffer(
            "action_scale",
            torch.tensor((action_range[1] - action_range[0]) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((action_range[1] + action_range[0]) / 2.0, dtype=torch.float32),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.trunk(state)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.min_log_std + 0.5 * (self.max_log_std - self.min_log_std) * (log_std + 1)
        return mean, log_std

    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        action = action * self.action_scale + self.action_bias
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class ReplayBuffer:
    """Fixed-size ring buffer to store experience tuples (s, a, r, s′, d)."""

    def __init__(self, buffer_size: int, batch_size: int, seed: int, device: torch.device):
        self.memory: deque[Data] = deque(maxlen=int(buffer_size))
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device

    def load_from_checkpoint(self, checkpoint: dict) -> None:
        """Load memory from checkpoint file"""
        buffer_state: bytes = checkpoint["buffer"]
        self.memory = dill.loads(buffer_state)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> None:
        """Add a new experience to the memory."""
        experience = Data(
            torch.as_tensor(state, dtype=torch.float32).to(self.device),
            torch.as_tensor(action, dtype=torch.float32).to(self.device),
            torch.as_tensor(reward, dtype=torch.float32).to(self.device),
            torch.as_tensor(next_state, dtype=torch.float32).to(self.device),
            torch.as_tensor(done, dtype=torch.float32).to(self.device),
        )
        self.memory.append(experience)

    def sample(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly sample a batch of experience tuples from memory."""
        indices = np.random.randint(0, high=len(self.memory), size=(self.batch_size,))
        s, a, r, s_prime, d = zip(*[self.memory[i] for i in indices])
        return (torch.cat(s), torch.cat(a), torch.stack(r), torch.cat(s_prime), torch.stack(d))

    def __len__(self):
        """Return the number of stored experience tuples."""
        return len(self.memory)


class RenderMode(Enum):
    human = "human"
    rgb_array = "rgb_array"
    depth_array = "depth_array"
    single_rgb_array = "single_rgb_array"
    single_depth_array = "single_depth_array"


class EnvMaker:
    def __init__(
        self,
        env_name: str,
        seed: int,
        exp_folder: str,
        max_episode_steps: int,
        render_mode: RenderMode = RenderMode.rgb_array,
        width: int = 256,
        height: int = 256,
        extra_videos: int = -1,
    ):
        self.env_name = env_name
        self.seed = seed
        self.exp_folder = exp_folder
        self.max_episode_steps = max_episode_steps
        self.render_mode: str = render_mode.value
        self.width = width
        self.height = height
        self.extra_videos = extra_videos

    def __call__(self, env_idx: int) -> gym.Env:
        env = gym.make(
            self.env_name,
            max_episode_steps=self.max_episode_steps,
            render_mode=self.render_mode,
            width=self.width,
            height=self.height,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if self.extra_videos > 0 and env_idx == 0:
            env = gym.wrappers.RecordVideo(env, self.exp_folder)
        env.seed(self.seed)
        env.action_space.seed(self.seed)
        env.observation_space.seed(self.seed)
        return env


def eval(env: gym.Env, policy: nn.Module, device: torch.device, n_traj: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the current policy deterministically
    by selecting the mean action from the state-conditioned policy"""
    ret_per_traj = list()
    len_per_traj = list()
    d = False
    ret = 0.0
    T = 0
    s: np.ndarray | torch.Tensor = env.reset()
    for _ in range(n_traj):
        while not d:
            with torch.no_grad():
                s = torch.tensor(s, dtype=torch.float32).to(device)
                _, _, a = policy.get_action(s)
            s, r, d, _ = env.step(a.detach().cpu().numpy())
            ret += r
            T += 1
        ret_per_traj.append(ret)
        len_per_traj.append(T)
        d = False
        ret = 0
        T = 0

    return np.asarray(ret_per_traj), np.asarray(len_per_traj)


def collect_video_frames(env: gym.Env, policy: nn.Module, device: torch.device, n_traj: int = 3) -> list[torch.Tensor]:
    s = env.reset()
    frames = list()
    videos = list()
    d = False
    frames.extend(env.envs[0].render())
    for _ in range(n_traj):
        while not d:
            with torch.no_grad():
                _, _, a = policy.get_action(torch.Tensor(s).to(device))
            s, _, d, _ = env.step(a.detach().cpu().numpy())
            frames.extend(env.envs[0].render())
        videos.append(torch.tensor(np.stack(frames, axis=0)).permute(0, 3, 1, 2).unsqueeze(0))
        s = env.reset()
        frames = list()
        frames.extend(env.envs[0].render())
        d = False
    return videos


def main(args: argparse.Namespace):
    args = parse_args()
    exp_folder = f"Experiments/{args.env_name}__seed_{args.seed}"
    os.makedirs(exp_folder)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    writer = SummaryWriter(exp_folder)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp_folder = ""
    env_maker = EnvMaker(
        env_name=args.env_name, seed=args.seed, exp_folder=exp_folder, max_episode_steps=args.max_ep_steps
    )
    envs = gym.vector.SyncVectorEnv([lambda: env_maker(id) for id in [0]])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "Action space is NOT continuous"
    eval_env = deepcopy(envs)

    state_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    action_range = (envs.action_space.low, envs.action_space.high)

    actor = Actor(state_dim, action_dim, action_range, args.log_std_bounds).to(device)
    qnets = TwinQNets(state_dim, action_dim, hidden_dim=256).to(device)
    target_qnets = TargetTwinQNets(qnets, polyak=args.polyak).to(device)

    q_optimizer = optim.Adam(qnets.parameters(), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )

    state: np.ndarray = envs.reset()
    ep_ret = 0
    ep_steps = 0
    action: np.ndarray | torch.Tensor

    for time_step in range(args.time_steps):
        if time_step < args.learning_starts:
            action = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            action, _, _ = actor.get_action(torch.as_tensor(state, dtype=torch.float32).to(device))
            action = action.detach().cpu().numpy()

        next_state, reward, done, _ = envs.step(action)

        ep_ret += reward
        ep_steps += 1

        if done:
            writer.add_scalar("Train/episodic_return", ep_ret, time_step)
            writer.add_scalar("Train/episodic_length", ep_steps, time_step)
            if ep_steps == args.max_ep_steps:
                done = ~done
            ep_ret = 0
            ep_steps = 0

        rb.add(state, action, reward, next_state.copy(), done)
        state = next_state

        if time_step > args.learning_starts:
            batch = Data(*rb.sample())
            with torch.no_grad():
                next_actions, log_p_next_actions, _ = actor.get_action(batch.next_states)
                q1_next, q2_next = target_qnets(batch.next_states, next_actions)
                min_q_next = torch.min(q1_next, q2_next) - alpha * log_p_next_actions
                next_q_value = batch.rewards + (1 - batch.dones) * args.gamma * min_q_next

            q1_values, q2_values = qnets(batch.states, batch.actions)
            q1_loss = F.mse_loss(q1_values, next_q_value)
            q2_loss = F.mse_loss(q2_values, next_q_value)
            q_loss = q1_loss + q2_loss

            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            if time_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    actions, log_p_actions, _ = actor.get_action(batch.states)
                    q1, q2 = qnets(batch.states, actions)
                    min_q = torch.min(q1, q2)
                    policy_loss = ((alpha * log_p_actions) - min_q).mean()

                    actor_optimizer.zero_grad()
                    policy_loss.backward()
                    actor_optimizer.step()

                    with torch.no_grad():
                        _, log_p_actions, _ = actor.get_action(batch.states)
                    alpha_loss = (-log_alpha * (log_p_actions + target_entropy)).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()
            target_qnets.soft_update(qnets)

            if time_step % args.eval_freq == 0:
                ep_return, ep_length = eval(eval_env, actor, device, n_traj=10)
                writer.add_scalar("Eval/episodic_return", ep_return.mean().item(), time_step)
                writer.add_scalar("Eval/episodic_length", ep_length.mean().item(), time_step)
                print(
                    ", ".join(
                        [
                            f"time_step: {time_step:6d}",
                            f"episodic_return: {ep_return.mean().item():>+8.2f}",
                            f"episodic_length: {ep_length.mean().item():4.0f}",
                        ]
                    )
                )

            if time_step % args.video_freq == 0:
                videos = collect_video_frames(eval_env, actor, device, n_traj=3)
                for i, vid in enumerate(videos):
                    writer.add_video(f"Video_{i}", vid, time_step)

            if time_step % 100 == 0:
                writer.add_scalar("losses/policy_loss", policy_loss.item(), time_step)
                writer.add_scalar("losses/alpha_loss", alpha_loss.item(), time_step)
                writer.add_scalar("losses/q_loss", q_loss.item() / 2.0, time_step)
                writer.add_scalar("losses/q1_loss", q1_loss.item(), time_step)
                writer.add_scalar("losses/q2_loss", q2_loss.item(), time_step)
                writer.add_scalar("values/q1_values", q1_values.mean().item(), time_step)
                writer.add_scalar("values/q2_values", q2_values.mean().item(), time_step)
                writer.add_scalar("values/alpha", alpha, time_step)

    torch.save(
        {"time_step": args.time_steps, "state_dict": actor.state_dict()},
        f"{exp_folder}/policy_ckpt.pth",
    )
    envs.close()
    writer.close()


if __name__ == "__main__":
    main(parse_args())
