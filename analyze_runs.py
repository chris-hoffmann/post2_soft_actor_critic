"""
This file contains code to plot learning curves and 
to create gif files for visualizing the performance of a trained agents.
The results are shown in the README file. 
"""


from __future__ import annotations

import os
from glob import glob
from typing import Optional

import argparse
import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from train_sac import Actor, EnvMaker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, required=True, help="the path for saving plots and gif files")
    args = parser.parse_args()
    return args


def _get_lcurves_from_tfevents(
    exp_tag: str, tag: str = "Eval/episodic_return"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed_dirs = glob(exp_tag)
    steps = list()
    values = list()
    for i, dir in enumerate(seed_dirs):
        event_acc = EventAccumulator(dir)
        event_acc.Reload()
        data = event_acc.Scalars(tag)
        time_series = list(zip(*[(e.step, e.value) for e in data]))
        if i == 0:
            steps.extend([*time_series[0]])
        values.append(time_series[1][: len(steps)])
    values = np.asarray(values).T
    return np.asarray(steps), values.mean(axis=-1), values.std(axis=-1)


def _save_lcurve(
    steps: list | np.ndarray, mean_values: np.ndarray, out_path: str, std_values: Optional[np.ndarray] = None
) -> None:
    plt.figure(figsize=(4, 3))
    plt.style.use("seaborn")
    plt.plot(steps, mean_values, ".-", c="#CC4F1B")
    if std_values is not None:
        plt.fill_between(
            steps,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.3,
            edgecolor="#CC4F1B",
            facecolor="#FF9848",
            linewidth=0.1,
        )
    plt.xlabel("Time steps")
    plt.ylabel("Average return")
    plt.savefig(f"{out_path}.png", bbox_inches="tight", dpi=300)
    plt.close()


def _collect_images(env: gym.Env, policy: torch.nn.Module) -> list[Image.Image]:
    s = env.reset()
    frames = list()
    d = False
    frames.extend(env.envs[0].render())
    while not d:
        with torch.no_grad():
            _, _, a = policy.get_action(torch.Tensor(s))
        s, _, d, _ = env.step(a.numpy())
        frames.extend(env.envs[0].render())
    return [Image.fromarray(f) for f in frames]


def _save_gif(images: list[Image.Image], out_name: str) -> None:
    images[0].save(f"{out_name}.gif", format="GIF", append_images=images[1:], save_all=True, duration=10, loop=0)


def write_learning_curve(env_name: str, out_dir: str) -> None:
    x, y, y_err = _get_lcurves_from_tfevents(exp_tag=f"Experiments/{env_name}*")
    _save_lcurve(x, y, out_path=f"{out_dir}/{env_name}", std_values=y_err)


def write_gif(env_name: str, out_dir: str) -> None:
    max_episode_steps = 150 if env_name == "HalfCheetah-v4" else 1000
    env_maker = EnvMaker(env_name=env_name, seed=1, exp_folder="", max_episode_steps=max_episode_steps)
    envs = gym.vector.SyncVectorEnv([lambda: env_maker(id) for id in [0]])
    actor = Actor(
        state_dim=envs.single_observation_space.shape[0],
        action_dim=envs.single_action_space.shape[0],
        action_range=(envs.action_space.low, envs.action_space.high),
        log_std_bounds=(-20.0, 2.0),
    )
    state_dict = torch.load(f"Experiments/{env_name}__seed_3/policy_ckpt.pth")["state_dict"]
    actor.load_state_dict(state_dict)
    images = _collect_images(envs, actor)
    _save_gif(images, out_name=f"{out_dir}/{env_name}__seed_3")


def main(args: argparse.Namespace):
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for name in ["HalfCheetah-v4", "Hopper-v4", "InvertedPendulum-v4"]:
        write_gif(name, out_dir)
        write_learning_curve(name, out_dir)


if __name__ == "__main__":
    main(parse_args())
