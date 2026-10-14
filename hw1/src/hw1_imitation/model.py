"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        # MLP: state -> [Linear -> ReLU] x len(hidden_dims) -> Linear -> flat chunk.
        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # (batch, state_dim) -> (batch, chunk_size, action_dim)
        return self.net(state).view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        # Eq. (1): squared L2 norm over the whole chunk, averaged over the batch.
        pred_chunk = self(state)
        return (pred_chunk - action_chunk).pow(2).sum(dim=(1, 2)).mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # Deterministic policy: a single forward pass, num_steps is unused.
        return self(state)


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        #MLP
        in_dim = state_dim + chunk_size*action_diim +1
        for hidden_dim in  hidden_dims:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, chunk_size*action_dim))
        self.net = nn.Sequential(*layers)

    def velocity(
        self,
        state: torch.Tensor,
        A_tau: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:

        x = torch.cat([state, A_tau.flatten(1)], dim = -1)
        v = self.net(x)
        return v.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        
        B = action_chunk.shape[0]
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(B, 1, device = action_chunk.device)
        tau_b = tau.view(B, 1, 1)
        A_tau = tau_b * action_chunk + (1-tau_b) * noise
        target = action_chunk - noise
        pred = self.velocity(state, A_tau, tau)
        return (pred - target).pow(2).sum(dim=(1,2)).mean()

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        raise NotImplementedError
        B = state.shape[0]
        A = torch.randn(B, self.chunk_size, self.action_dim, device = state.device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau = torch.full((B,1), i*dt, device = state.device)
            A = A +dt * self.velocity(state, A, tau)
        return A


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
