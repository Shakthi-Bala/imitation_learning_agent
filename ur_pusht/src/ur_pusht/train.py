"""Train an action-chunking policy on robot episodes and evaluate it on the UR5e."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from hw1_imitation.data import Normalizer, PushtChunkDataset
from hw1_imitation.model import PolicyType, build_policy
from torch.utils.data import DataLoader

from ur_pusht.dataset import load_datasets
from ur_pusht.policy import Policy
from ur_pusht.rollout import run_episodes, summarize


@dataclass
class TrainConfig:
    datasets: tuple[Path, ...]
    out: Path
    policy_type: PolicyType = "flow"
    chunk_size: int = 8
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    batch_size: int = 128
    lr: float = 3e-4
    steps: int = 60_000
    log_every: int = 500
    eval_every: int = 20_000
    eval_episodes: int = 100
    workers: int = 32
    threads: int = 4
    seed: int = 0


def _forever(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from loader


def train(config: TrainConfig) -> list[dict]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(config.threads)
    config.out.mkdir(parents=True, exist_ok=True)

    states, actions, episode_ends = load_datasets(list(config.datasets))
    normalizer = Normalizer.from_data(states, actions)
    dataset = PushtChunkDataset(states, actions, episode_ends, config.chunk_size, normalizer)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    print(
        f"{len(episode_ends)} episodes, {len(dataset)} chunks from {len(config.datasets)} datasets"
    )

    model = build_policy(
        config.policy_type,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        hidden_dims=config.hidden_dims,
    )
    policy = Policy(model, normalizer, config.policy_type, config.hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    compute_loss = torch.compile(model.compute_loss)

    metrics, running_loss = [], 0.0
    for step, (state, chunk) in enumerate(_forever(loader), start=1):
        loss = compute_loss(state, chunk)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if step % config.log_every == 0:
            metrics.append({"step": step, "train/loss": running_loss / config.log_every})
            running_loss = 0.0
        if step % config.eval_every == 0 or step == config.steps:
            checkpoint = config.out / f"policy_step{step}.pt"
            policy.save(checkpoint)
            episodes = run_episodes(checkpoint, list(range(config.eval_episodes)), config.workers)
            evaluation = {f"eval/{k}": v for k, v in summarize(episodes).items()}
            metrics.append({"step": step} | evaluation)
            print(json.dumps(metrics[-1]))
        if step == config.steps:
            break

    policy.save(config.out / "policy.pt")
    with (config.out / "metrics.jsonl").open("w") as f:
        f.writelines(json.dumps(row) + "\n" for row in metrics)
    return metrics


if __name__ == "__main__":
    train(tyro.cli(TrainConfig))
