"""Shared-weight job scoring networks used by both PPO and DQN.

Every waiting job is embedded by the same MLP from its slot features and the global
features. A Deep Sets context (masked mean and max over the embeddings) lets each
job's score depend on the rest of the queue, and gives the value function a
whole-queue input. The networks are permutation-equivariant over job slots, and
padded slots do not influence any output.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lvhm.env import LVHMSchedulingEnv


@dataclass(frozen=True)
class ObsLayout:
    num_slots: int
    slot_dim: int
    global_dim: int
    num_modes: int

    @classmethod
    def from_env(cls, env: LVHMSchedulingEnv) -> ObsLayout:
        env = env.unwrapped
        return cls(env.cfg.num_slots, env.slot_dim, env.global_dim, env.num_modes)

    @property
    def num_actions(self) -> int:
        return self.num_slots * self.num_modes


def mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class QueueEncoder(nn.Module):
    def __init__(self, layout: ObsLayout, hidden: int = 64):
        super().__init__()
        self.layout = layout
        self.job = nn.Sequential(mlp([layout.slot_dim + layout.global_dim, hidden, hidden]), nn.ReLU())
        self.context_dim = 2 * hidden

    def forward(self, obs: torch.Tensor):
        """Returns job embeddings (B, K, H), queue context (B, 2H), presence (B, K), globals (B, G)."""
        k, s = self.layout.num_slots, self.layout.slot_dim
        slots = obs[:, : k * s].reshape(-1, k, s)
        glob = obs[:, k * s :]
        present = slots[..., 0] > 0.5

        h = self.job(torch.cat([slots, glob.unsqueeze(1).expand(-1, k, -1)], dim=-1))
        mask = present.unsqueeze(-1)
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = h.masked_fill(~mask, float("-inf")).max(1).values
        mx = torch.where(present.any(1, keepdim=True), mx, torch.zeros_like(mx))
        return h, torch.cat([mean, mx], dim=-1), present, glob


class JobScorer(nn.Module):
    """One score per (slot, mode), laid out as action = mode * K + slot."""

    def __init__(self, layout: ObsLayout, embed_dim: int, context_dim: int, hidden: int = 64):
        super().__init__()
        self.layout = layout
        self.net = mlp([embed_dim + context_dim, hidden, layout.num_modes])

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        k = self.layout.num_slots
        scores = self.net(torch.cat([h, context.unsqueeze(1).expand(-1, k, -1)], dim=-1))  # (B, K, M)
        return scores.transpose(1, 2).reshape(h.shape[0], -1)

    @property
    def output_layer(self) -> nn.Linear:
        return self.net[-1]


class ValueHead(nn.Module):
    def __init__(self, context_dim: int, global_dim: int, hidden: int = 64, out: int = 1):
        super().__init__()
        self.net = mlp([context_dim + global_dim, hidden, out])

    def forward(self, context: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([context, glob], dim=-1))
