"""Evaluate a policy on the UR5e Push-T environment over the standard 100 evaluation seeds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import tyro

from ur_pusht.rollout import run_episodes, summarize


@dataclass
class EvalConfig:
    policy: Path
    episodes: int = 100
    workers: int = 16
    out: Path | None = None


def main(config: EvalConfig) -> dict[str, float]:
    episodes = run_episodes(config.policy, list(range(config.episodes)), config.workers)
    summary = summarize(episodes) | {"episodes": config.episodes, "policy": str(config.policy)}
    print(json.dumps(summary, indent=2))
    if config.out is not None:
        config.out.parent.mkdir(parents=True, exist_ok=True)
        config.out.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
