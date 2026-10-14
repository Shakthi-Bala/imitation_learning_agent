"""Self-improvement: alternate between collecting successful episodes and retraining on them.

Round 0 is the 2D Push-T policy running zero-shot on the arm. Each later round collects
episodes with the previous policy on fresh seeds, keeps the successes, and trains a new policy
from scratch on every success gathered so far, plus any fixed base datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro
from hw1_imitation.model import PolicyType

from ur_pusht.collect import TRAIN_SEED_OFFSET, CollectConfig, collect
from ur_pusht.rollout import run_episodes, summarize
from ur_pusht.train import TrainConfig, train


@dataclass
class ImproveConfig:
    initial_policy: Path = Path("checkpoints/pusht2d_flow.pt")
    out: Path = Path("runs/self_improvement")
    base_datasets: tuple[Path, ...] = ()
    rounds: int = 3
    episodes_per_round: int = 2000
    policy_type: PolicyType = "flow"
    train_steps: int = 60_000
    eval_every: int = 20_000
    eval_episodes: int = 100
    workers: int = 32


def improve(config: ImproveConfig) -> list[dict]:
    config.out.mkdir(parents=True, exist_ok=True)
    zero_shot = run_episodes(
        config.initial_policy, list(range(config.eval_episodes)), config.workers
    )
    history = [{"round": 0, "policy": str(config.initial_policy), "eval": summarize(zero_shot)}]
    policy, datasets = config.initial_policy, list(config.base_datasets)

    for round_index in range(1, config.rounds + 1):
        round_dir = config.out / f"round{round_index}"
        first_seed = TRAIN_SEED_OFFSET + (round_index - 1) * config.episodes_per_round
        collected = collect(
            CollectConfig(
                policy,
                round_dir / "data.zarr",
                config.episodes_per_round,
                first_seed,
                config.workers,
            )
        )
        datasets.append(round_dir / "data.zarr")
        metrics = train(
            TrainConfig(
                datasets=tuple(datasets),
                out=round_dir,
                policy_type=config.policy_type,
                steps=config.train_steps,
                eval_every=config.eval_every,
                eval_episodes=config.eval_episodes,
                workers=config.workers,
            )
        )
        policy = round_dir / "policy.pt"
        final_eval = {k.removeprefix("eval/"): v for k, v in metrics[-1].items() if k != "step"}
        history.append(
            {"round": round_index, "policy": str(policy), "collect": collected, "eval": final_eval}
        )
        (config.out / "history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(history[-1], indent=2))
    return history


if __name__ == "__main__":
    improve(tyro.cli(ImproveConfig))
