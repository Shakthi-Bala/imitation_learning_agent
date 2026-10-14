"""Action-chunking policies bundled with the normalizer they were trained with."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from hw1_imitation.data import Normalizer, download_pusht, load_pusht_zarr
from hw1_imitation.model import BasePolicy, FlowMatchingPolicy, PolicyType, build_policy

POLICY_TYPES: dict[type, PolicyType] = {FlowMatchingPolicy: "flow"}


@dataclass
class Policy:
    model: BasePolicy
    normalizer: Normalizer
    policy_type: PolicyType
    hidden_dims: tuple[int, ...]
    flow_steps: int = 10

    @property
    def chunk_size(self) -> int:
        return self.model.chunk_size

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        state = torch.from_numpy(self.normalizer.normalize_state(observation)).float()[None]
        chunk = self.model.sample_actions(state, num_steps=self.flow_steps)[0].numpy()
        return self.normalizer.denormalize_action(chunk)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_type": self.policy_type,
                "state_dim": self.model.state_dim,
                "action_dim": self.model.action_dim,
                "chunk_size": self.model.chunk_size,
                "hidden_dims": self.hidden_dims,
                "state_dict": self.model.state_dict(),
                "normalizer": {k: np.asarray(v) for k, v in vars(self.normalizer).items()},
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, flow_steps: int = 10) -> Policy:
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        model = build_policy(
            bundle["policy_type"],
            state_dim=bundle["state_dim"],
            action_dim=bundle["action_dim"],
            chunk_size=bundle["chunk_size"],
            hidden_dims=tuple(bundle["hidden_dims"]),
        )
        model.load_state_dict(bundle["state_dict"])
        return cls(
            model.eval(),
            Normalizer(**bundle["normalizer"]),
            bundle["policy_type"],
            tuple(bundle["hidden_dims"]),
            flow_steps,
        )


@dataclass
class ImportConfig:
    """Convert a 2D Push-T checkpoint (a pickled hw1_imitation model) into a policy bundle."""

    checkpoint: Path
    out: Path
    data_dir: Path = Path("data")


def import_pusht2d(config: ImportConfig) -> None:
    model = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    states, actions, _ = load_pusht_zarr(download_pusht(config.data_dir))
    hidden_dims = tuple(
        layer.out_features for layer in model.net if isinstance(layer, torch.nn.Linear)
    )[:-1]
    policy_type = POLICY_TYPES.get(type(model), "mse")
    Policy(model, Normalizer.from_data(states, actions), policy_type, hidden_dims).save(config.out)
    print(f"Saved {policy_type} policy {hidden_dims} to {config.out}")


if __name__ == "__main__":
    import_pusht2d(tyro.cli(ImportConfig))
