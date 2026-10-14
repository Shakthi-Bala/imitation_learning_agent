"""Collect robot demonstrations by running a policy and keeping only its successful episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from ur_pusht.dataset import save_episodes
from ur_pusht.rollout import run_episodes, summarize

TRAIN_SEED_OFFSET = 10_000


@dataclass
class CollectConfig:
    policy: Path
    out: Path
    episodes: int = 2000
    first_seed: int = TRAIN_SEED_OFFSET
    workers: int = 32


def collect(config: CollectConfig) -> dict[str, float]:
    seeds = list(range(config.first_seed, config.first_seed + config.episodes))
    episodes = run_episodes(config.policy, seeds, config.workers)
    kept = [e for e in episodes if e.success]
    save_episodes(kept, config.out)
    stats = summarize(episodes) | {
        "episodes": len(episodes),
        "kept_episodes": len(kept),
        "kept_steps": sum(len(e) for e in kept),
    }
    (config.out / "collect.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    collect(tyro.cli(CollectConfig))
