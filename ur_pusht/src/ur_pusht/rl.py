"""Reinforcement-learning fine-tuning of a flow matching policy on the UR5e Push-T task.

Each policy query is one decision: the state at the start of a chunk and the 8-action chunk.
Its reward is the gain in Push-T reward over the chunk plus a bonus on success, so an episode's
return is its final coverage reward (minus the start) plus the bonus.

Two algorithms share the rollouts and the critic:

* ``awr``: advantage-weighted regression. The flow matching loss of each executed chunk is
  weighted by exp(advantage / beta).
* ``ppo``: DPPO-style policy gradient. Sampling adds Gaussian noise of scale ``sigma`` to every
  Euler step, which makes each denoising step a Gaussian policy whose log-probability PPO can
  clip and differentiate.

Evaluation always uses the deterministic Euler sampler of the original policy.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from torch import nn

from ur_pusht import geometry as geo
from ur_pusht.env import URPushTEnv
from ur_pusht.policy import Policy
from ur_pusht.rollout import run_episodes, summarize

RL_SEED_OFFSET = 100_000


@dataclass
class RLConfig:
    algo: Literal["ppo", "awr"] = "ppo"
    init_policy: Path = Path("checkpoints/pusht2d_flow.pt")
    out: Path = Path("runs/rl")
    iterations: int = 30
    episodes_per_iter: int = 256
    workers: int = 32
    sigma: float = 0.05
    gamma: float = 0.99
    gae_lambda: float = 0.95
    success_bonus: float = 1.0
    critic_warmup: int = 2
    actor_lr: float = 1e-5
    critic_lr: float = 1e-3
    epochs: int = 5
    minibatch: int = 4096
    clip: float = 0.01
    awr_beta: float = 0.5
    awr_max_weight: float = 20.0
    eval_every: int = 5
    eval_episodes: int = 100
    critic_hidden: tuple[int, ...] = field(default=(256, 256))
    seed: int = 0


@dataclass
class EpisodeChunks:
    states: np.ndarray
    chains: np.ndarray
    rewards: np.ndarray
    final_state: np.ndarray
    terminal: bool
    max_reward: float


def sample_chain(model: nn.Module, state: torch.Tensor, steps: int, sigma: float) -> torch.Tensor:
    """Euler-integrate the flow from noise with optional per-step Gaussian noise.

    Returns the whole chain, shape (batch, steps + 1, chunk_size, action_dim).
    """
    batch = state.shape[0]
    chunk = torch.randn(batch, model.chunk_size, model.action_dim)
    chain, dt = [chunk], 1.0 / steps
    for k in range(steps):
        tau = torch.full((batch, 1), k * dt)
        chunk = chunk + dt * model.velocity(state, chunk, tau)
        if sigma > 0:
            chunk = chunk + sigma * torch.randn_like(chunk)
        chain.append(chunk)
    return torch.stack(chain, dim=1)


def chain_log_prob(
    model: nn.Module, states: torch.Tensor, chains: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Log-probability of every denoising step, shape (batch * steps,), up to a constant."""
    batch, steps = chains.shape[0], chains.shape[1] - 1
    dt = 1.0 / steps
    current = chains[:, :-1].reshape(batch * steps, *chains.shape[2:])
    following = chains[:, 1:].reshape(batch * steps, *chains.shape[2:])
    tau = (torch.arange(steps, dtype=torch.float32) * dt).repeat(batch)[:, None]
    mean = current + dt * model.velocity(states.repeat_interleave(steps, 0), current, tau)
    return -((following - mean) ** 2).sum(dim=(1, 2)) / (2 * sigma**2)


def flow_loss_per_sample(
    model: nn.Module, states: torch.Tensor, chunks: torch.Tensor
) -> torch.Tensor:
    noise = torch.randn_like(chunks)
    tau = torch.rand(chunks.shape[0], 1)
    tau_b = tau[:, :, None]
    mixed = tau_b * chunks + (1 - tau_b) * noise
    prediction = model.velocity(states, mixed, tau)
    return ((prediction - (chunks - noise)) ** 2).sum(dim=(1, 2))


_worker: dict = {}


def _init_worker(policy_path: Path) -> None:
    torch.set_num_threads(1)
    _worker.update(env=URPushTEnv(), policy=Policy.load(policy_path), version=-1)


def _collect(job: tuple[int, Path, int, float, float]) -> EpisodeChunks:
    seed, weights, version, sigma, success_bonus = job
    env, policy = _worker["env"], _worker["policy"]
    if version != _worker["version"]:
        policy.model.load_state_dict(torch.load(weights, weights_only=True))
        _worker["version"] = version
    torch.manual_seed(seed)

    observation = env.reset(seed)
    states, chains, rewards = [], [], []
    done, success, max_reward = False, False, 0.0
    while not done:
        state = torch.from_numpy(policy.normalizer.normalize_state(observation)).float()[None]
        with torch.no_grad():
            chain = sample_chain(policy.model, state, policy.flow_steps, sigma)[0]
        chunk = np.clip(policy.normalizer.denormalize_action(chain[-1].numpy()), 0.0, 512.0)
        start_reward = geo.reward(observation[2:])
        for action in chunk:
            observation, reward, success, truncated, _ = env.step(action)
            max_reward = max(max_reward, reward)
            done = success or truncated
            if done:
                break
        states.append(state[0].numpy())
        chains.append(chain.numpy())
        rewards.append(reward - start_reward + success_bonus * success)
    final_state = policy.normalizer.normalize_state(observation).astype(np.float32)
    return EpisodeChunks(
        np.array(states),
        np.array(chains),
        np.array(rewards, dtype=np.float32),
        final_state,
        bool(success),
        max_reward,
    )


def _critic(state_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers, width = [], state_dim
    for size in hidden:
        layers += [nn.Linear(width, size), nn.Tanh()]
        width = size
    return nn.Sequential(*layers, nn.Linear(width, 1))


def _advantages(
    episodes: list[EpisodeChunks], critic: nn.Module, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    advantages, returns = [], []
    with torch.no_grad():
        for episode in episodes:
            values = critic(torch.from_numpy(episode.states)).squeeze(-1).numpy()
            bootstrap = (
                0.0
                if episode.terminal
                else critic(torch.from_numpy(episode.final_state)[None]).item()
            )
            next_values = np.append(values[1:], bootstrap)
            deltas = episode.rewards + gamma * next_values - values
            gae, episode_advantages = 0.0, np.zeros_like(deltas)
            for t in reversed(range(len(deltas))):
                gae = deltas[t] + gamma * lam * gae
                episode_advantages[t] = gae
            advantages.append(episode_advantages)
            returns.append(episode_advantages + values)
    return np.concatenate(advantages), np.concatenate(returns)


def _batches(size: int, minibatch: int):
    order = torch.randperm(size)
    for start in range(0, size, minibatch):
        yield order[start : start + minibatch]


def update(
    config: RLConfig,
    policy: Policy,
    critic: nn.Module,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    episodes: list[EpisodeChunks],
    train_actor: bool,
) -> dict[str, float]:
    actor_optimizer, critic_optimizer = optimizers
    advantages, returns = _advantages(episodes, critic, config.gamma, config.gae_lambda)
    states = torch.from_numpy(np.concatenate([e.states for e in episodes]))
    chains = torch.from_numpy(np.concatenate([e.chains for e in episodes]))
    advantages = torch.from_numpy((advantages - advantages.mean()) / (advantages.std() + 1e-8))
    returns = torch.from_numpy(returns)
    steps = chains.shape[1] - 1
    stats = {"critic_loss": 0.0, "actor_loss": 0.0, "clip_fraction": 0.0}

    if config.algo == "ppo":
        with torch.no_grad():
            old_log_prob = chain_log_prob(policy.model, states, chains, config.sigma)

    batches = 0
    for _ in range(config.epochs):
        for index in _batches(len(states), config.minibatch):
            value_loss = ((critic(states[index]).squeeze(-1) - returns[index]) ** 2).mean()
            critic_optimizer.zero_grad()
            value_loss.backward()
            critic_optimizer.step()
            stats["critic_loss"] += value_loss.item()
            batches += 1
            if not train_actor:
                continue

            if config.algo == "ppo":
                step_index = (index[:, None] * steps + torch.arange(steps)).reshape(-1)
                log_prob = chain_log_prob(policy.model, states[index], chains[index], config.sigma)
                ratio = torch.exp(log_prob - old_log_prob[step_index])
                advantage = advantages[index].repeat_interleave(steps)
                clipped = torch.clamp(ratio, 1 - config.clip, 1 + config.clip)
                actor_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
                stats["clip_fraction"] += ((ratio - 1).abs() > config.clip).float().mean().item()
            else:
                weights = torch.exp(advantages[index] / config.awr_beta).clamp(
                    max=config.awr_max_weight
                )
                losses = flow_loss_per_sample(policy.model, states[index], chains[index, -1])
                actor_loss = (weights * losses).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(policy.model.parameters(), 1.0)
            actor_optimizer.step()
            stats["actor_loss"] += actor_loss.item()
    return {k: v / max(batches, 1) for k, v in stats.items()}


def evaluate(config: RLConfig, policy: Policy, iteration: int) -> dict[str, float]:
    checkpoint = config.out / f"policy_iter{iteration}.pt"
    policy.save(checkpoint)
    episodes = run_episodes(checkpoint, list(range(config.eval_episodes)), config.workers)
    return {f"eval/{k}": v for k, v in summarize(episodes).items()}


def finetune(config: RLConfig) -> list[dict]:
    torch.manual_seed(config.seed)
    config.out.mkdir(parents=True, exist_ok=True)
    policy = Policy.load(config.init_policy)
    critic = _critic(policy.model.state_dim, config.critic_hidden)
    optimizers = (
        torch.optim.Adam(policy.model.parameters(), lr=config.actor_lr),
        torch.optim.Adam(critic.parameters(), lr=config.critic_lr),
    )
    weights = config.out / "actor_latest.pt"
    metrics = [{"iteration": 0} | evaluate(config, policy, 0)]
    print(json.dumps(metrics[-1]))

    context = mp.get_context("spawn")
    with context.Pool(
        config.workers, initializer=_init_worker, initargs=(config.init_policy,)
    ) as pool:
        for iteration in range(1, config.iterations + 1):
            torch.save(policy.model.state_dict(), weights)
            first = RL_SEED_OFFSET + (iteration - 1) * config.episodes_per_iter
            jobs = [
                (seed, weights, iteration, config.sigma, config.success_bonus)
                for seed in range(first, first + config.episodes_per_iter)
            ]
            episodes = pool.map(_collect, jobs, chunksize=1)

            policy.model.train()
            stats = update(
                config,
                policy,
                critic,
                optimizers,
                episodes,
                train_actor=iteration > config.critic_warmup,
            )
            policy.model.eval()
            row = {
                "iteration": iteration,
                "train/mean_reward": float(np.mean([e.max_reward for e in episodes])),
                "train/success_rate": float(np.mean([e.terminal for e in episodes])),
                "train/return": float(np.mean([e.rewards.sum() for e in episodes])),
            } | {f"train/{k}": v for k, v in stats.items()}
            if iteration % config.eval_every == 0 or iteration == config.iterations:
                row |= evaluate(config, policy, iteration)
            metrics.append(row)
            print(json.dumps(row))
            with (config.out / "metrics.jsonl").open("w") as f:
                f.writelines(json.dumps(r) + "\n" for r in metrics)

    policy.save(config.out / "policy.pt")
    return metrics


if __name__ == "__main__":
    finetune(tyro.cli(RLConfig))
