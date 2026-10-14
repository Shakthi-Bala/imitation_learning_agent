"""Episode storage in the Push-T zarr layout (data/state, data/action, meta/episode_ends)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from hw1_imitation.data import load_pusht_zarr

from ur_pusht.rollout import Episode


def save_episodes(episodes: list[Episode], path: Path) -> None:
    root = zarr.open_group(str(path), mode="w")
    data, meta = root.create_group("data"), root.create_group("meta")
    data.create_array("state", data=np.concatenate([e.states for e in episodes]).astype(np.float32))
    data.create_array(
        "action", data=np.concatenate([e.actions for e in episodes]).astype(np.float32)
    )
    meta.create_array("episode_ends", data=np.cumsum([len(e) for e in episodes]).astype(np.int64))
    meta.create_array("seed", data=np.array([e.seed for e in episodes], dtype=np.int64))
    meta.create_array(
        "max_reward", data=np.array([e.max_reward for e in episodes], dtype=np.float32)
    )


def load_datasets(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, actions, episode_ends, offset = [], [], [], 0
    for path in paths:
        s, a, ends = load_pusht_zarr(path)
        states.append(s)
        actions.append(a)
        episode_ends.append(ends + offset)
        offset += len(s)
    return np.concatenate(states), np.concatenate(actions), np.concatenate(episode_ends)
