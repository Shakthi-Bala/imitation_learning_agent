"""Closed-loop rollouts of action-chunking policies on the UR5e Push-T environment."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ur_pusht.env import URPushTEnv
from ur_pusht.policy import Policy


@dataclass
class Episode:
    seed: int
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    success: bool

    @property
    def max_reward(self) -> float:
        return float(self.rewards.max(initial=0.0))

    def __len__(self) -> int:
        return len(self.actions)


def run_episode(
    env: URPushTEnv, policy: Policy, seed: int, sample_seed: int | None = None, on_step=None
) -> Episode:
    torch.manual_seed(seed if sample_seed is None else sample_seed)
    observation = env.reset(seed)
    states, actions, rewards = [], [], []
    chunk, index, done, success = np.empty((0, 2)), 0, False, False
    while not done:
        if index == len(chunk):
            chunk, index = np.clip(policy.act(observation), 0.0, 512.0), 0
        action = chunk[index]
        index += 1
        states.append(observation)
        actions.append(action)
        observation, reward, success, truncated, _ = env.step(action)
        rewards.append(reward)
        done = success or truncated
        if on_step is not None:
            on_step(env)
    return Episode(seed, np.array(states), np.array(actions), np.array(rewards), bool(success))


_worker: tuple[URPushTEnv, Policy] | None = None


def _init_worker(policy_path: Path) -> None:
    global _worker
    torch.set_num_threads(1)
    _worker = (URPushTEnv(), Policy.load(policy_path))


def _run(job: tuple[int, int]) -> Episode:
    env, policy = _worker
    seed, sample_seed = job
    return run_episode(env, policy, seed, sample_seed)


def run_episodes(
    policy_path: Path, seeds: list[int], workers: int, sample_seeds: list[int] | None = None
) -> list[Episode]:
    jobs = list(zip(seeds, sample_seeds or seeds))
    context = mp.get_context("spawn")
    with context.Pool(workers, initializer=_init_worker, initargs=(policy_path,)) as pool:
        return pool.map(_run, jobs, chunksize=1)


def summarize(episodes: list[Episode]) -> dict[str, float]:
    rewards = np.array([e.max_reward for e in episodes])
    return {
        "mean_reward": float(rewards.mean()),
        "success_rate": float(np.mean([e.success for e in episodes])),
        "near_success_rate": float(np.mean(rewards >= 0.8)),
        "failure_rate": float(np.mean(rewards < 0.3)),
        "mean_length": float(np.mean([len(e) for e in episodes])),
    }
