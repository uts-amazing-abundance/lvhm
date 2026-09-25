"""Masked double DQN with a dueling head, in plain PyTorch.

The advantage stream is the shared-weight job scorer and the value stream reads
the pooled queue context, so Q(s, a) = V(s) + A(s, a) - mean over valid a of A.
Invalid actions are masked out of action selection and of the double-DQN target.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from lvhm.agents.common import VAL_SEED, Validator
from lvhm.agents.networks import JobScorer, ObsLayout, QueueEncoder, ValueHead
from lvhm.baselines import Policy, edf
from lvhm.config import FAMILIAR, EnvConfig
from lvhm.env import LVHMSchedulingEnv
from lvhm.instance import make_instances

MASKED = -1e9


class DuelingJobQNet(nn.Module):
    def __init__(self, layout: ObsLayout, hidden: int = 64):
        super().__init__()
        self.encoder = QueueEncoder(layout, hidden)
        self.advantage = JobScorer(layout, hidden, self.encoder.context_dim, hidden)
        self.value = ValueHead(self.encoder.context_dim, layout.global_dim, hidden)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, context, _, glob = self.encoder(obs)
        adv = self.advantage(h, context)
        adv_mean = (adv * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
        q = self.value(context, glob) + adv - adv_mean
        return q.masked_fill(~mask, MASKED)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, num_actions: int):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.mask = np.zeros((capacity, num_actions), dtype=bool)
        self.next_mask = np.zeros((capacity, num_actions), dtype=bool)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, mask, action, reward, next_obs, next_mask, done) -> None:
        i = self.pos
        self.obs[i], self.mask[i], self.action[i], self.reward[i] = obs, mask, action, reward
        self.next_obs[i], self.next_mask[i], self.done[i] = next_obs, next_mask, done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = rng.integers(self.size, size=batch_size)
        return {
            name: torch.as_tensor(getattr(self, name)[idx], device=device)
            for name in ("obs", "mask", "action", "reward", "next_obs", "next_mask", "done")
        }


@dataclass
class DQNConfig:
    timesteps: int = 300_000
    n_envs: int = 8
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 128
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    train_freq: int = 4  # env steps per gradient update
    tau: float = 0.005  # Polyak rate of the target network
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3  # of timesteps over which epsilon decays linearly
    warmup_edf_steps: int = 0  # optionally act with EDF first to seed the buffer
    hidden: int = 64
    grad_clip: float = 10.0
    device: str = "auto"  # "auto" uses CUDA when available


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def as_policy(net: DuelingJobQNet) -> Policy:
    """Greedy masked policy with the evaluate() signature."""
    device = next(net.parameters()).device

    def policy(obs: np.ndarray, env: LVHMSchedulingEnv) -> int:
        with torch.no_grad():
            q = net(torch.as_tensor(obs[None], device=device), torch.as_tensor(env.action_masks()[None], device=device))
        return int(q.argmax(1))

    return policy


def train_dqn(
    env_config: EnvConfig,
    cfg: DQNConfig,
    seed: int,
    validate: Callable[[DuelingJobQNet, int], None],
    val_every: int,
) -> DuelingJobQNet:
    """Trains and returns the online network; `validate(online_net, timesteps)` is
    called every `val_every` steps and at the end."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = resolve_device(cfg.device)
    envs = [LVHMSchedulingEnv(env_config) for _ in range(cfg.n_envs)]
    layout = ObsLayout.from_env(envs[0])
    online = DuelingJobQNet(layout, cfg.hidden).to(device)
    target = DuelingJobQNet(layout, cfg.hidden).to(device)
    target.load_state_dict(online.state_dict())
    target.requires_grad_(False)
    optimizer = torch.optim.Adam(online.parameters(), lr=cfg.lr)
    buffer = ReplayBuffer(cfg.buffer_size, envs[0].observation_space.shape[0], layout.num_actions)

    obs = np.stack([env.reset(seed=seed * 1000 + i)[0] for i, env in enumerate(envs)])
    masks = np.stack([env.action_masks() for env in envs])
    step, next_val, updates_due = 0, 0, 0.0

    while step < cfg.timesteps:
        if step >= next_val:
            validate(online, step)
            next_val += val_every

        eps = max(cfg.eps_end, cfg.eps_start - (cfg.eps_start - cfg.eps_end) * step / (cfg.eps_fraction * cfg.timesteps))
        with torch.no_grad():
            greedy = online(torch.as_tensor(obs, device=device), torch.as_tensor(masks, device=device)).argmax(1).cpu().numpy()
        for i, env in enumerate(envs):
            if step < cfg.warmup_edf_steps:
                action = edf(obs[i], env)
            elif rng.random() < eps:
                action = int(rng.choice(np.flatnonzero(masks[i])))
            else:
                action = int(greedy[i])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_mask = env.action_masks()
            buffer.add(obs[i], masks[i], action, reward, next_obs, next_mask, float(terminated))
            if terminated or truncated:
                next_obs, _ = env.reset()
                next_mask = env.action_masks()
            obs[i], masks[i] = next_obs, next_mask
        step += cfg.n_envs

        if step < cfg.learning_starts:
            continue
        updates_due += cfg.n_envs / cfg.train_freq
        while updates_due >= 1:
            updates_due -= 1
            _update(online, target, optimizer, buffer.sample(rng, cfg.batch_size, device), cfg)

    validate(online, step)
    return online


def _update(online, target, optimizer, batch, cfg: DQNConfig) -> None:
    q = online(batch["obs"], batch["mask"]).gather(1, batch["action"][:, None]).squeeze(1)
    with torch.no_grad():
        best_next = online(batch["next_obs"], batch["next_mask"]).argmax(1, keepdim=True)
        next_q = target(batch["next_obs"], batch["next_mask"]).gather(1, best_next).squeeze(1)
        done = batch["done"].bool()
        next_q = torch.where(done, torch.zeros_like(next_q), next_q)  # terminal states have no valid action
        y = batch["reward"] + cfg.gamma * next_q
    loss = F.smooth_l1_loss(q, y)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
    optimizer.step()
    with torch.no_grad():
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.lerp_(p, cfg.tau)


def run_dqn(
    env_config: EnvConfig,
    out_dir: Path,
    seed: int = 0,
    dqn_config: DQNConfig | None = None,
    val_every: int = 25_000,
    val_shifts: int = 20,
) -> dict:
    """Train masked double dueling DQN, keeping the checkpoint that scores best on the
    validation shifts as out_dir/best_model.pt."""
    cfg = dqn_config or DQNConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    validator = Validator(make_instances(env_config, FAMILIAR, val_shifts, VAL_SEED), env_config, out_dir)

    def validate(net: DuelingJobQNet, timesteps: int) -> None:
        validator(as_policy(net), timesteps, lambda: torch.save(net.state_dict(), out_dir / "best_model.pt"))

    start = time.perf_counter()
    train_dqn(env_config, cfg, seed, validate, val_every)
    return {
        "seed": seed,
        "timesteps": cfg.timesteps,
        "device": str(resolve_device(cfg.device)),
        "train_seconds": round(time.perf_counter() - start, 1),
        "best_validation_score": validator.best,
        "config": asdict(cfg),
    }


def load_dqn(path: Path, env_config: EnvConfig | None = None, hidden: int = 64, device: str = "cpu") -> Policy:
    """Greedy policy from a saved checkpoint (trained on any device)."""
    net = DuelingJobQNet(ObsLayout.from_env(LVHMSchedulingEnv(env_config)), hidden)
    net.load_state_dict(torch.load(path, map_location=device))
    return as_policy(net.to(device).eval())
