"""Masked PPO (sb3-contrib) with the shared-weight job scorer as its policy."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

from lvhm.agents.common import VAL_SEED, Validator
from lvhm.agents.networks import JobScorer, ObsLayout, QueueEncoder, mlp
from lvhm.baselines import Policy
from lvhm.config import FAMILIAR, EnvConfig
from lvhm.env import LVHMSchedulingEnv
from lvhm.instance import make_instances

# sb3 MaskablePPO arguments used unless overridden in train_ppo(**ppo_kwargs)
PPO_DEFAULTS = {
    "learning_rate": 3e-4,
    "n_steps": 256,  # rollout length per env; 8 envs -> 2048 transitions per update
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,  # entropy bonus plays the role of DQN's epsilon
}


class JobScoringExtractor(nn.Module):
    """Plays SB3's mlp_extractor role: latent_pi is already the per-action logits,
    latent_vf is a whole-queue value embedding."""

    def __init__(self, layout: ObsLayout, hidden: int):
        super().__init__()
        self.encoder = QueueEncoder(layout, hidden)
        self.scorer = JobScorer(layout, hidden, self.encoder.context_dim, hidden)
        self.value = nn.Sequential(mlp([self.encoder.context_dim + layout.global_dim, hidden, hidden]), nn.ReLU())
        self.latent_dim_pi = layout.num_actions
        self.latent_dim_vf = hidden

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, context, _, glob = self.encoder(features)
        return self.scorer(h, context), self.value(torch.cat([context, glob], dim=-1))

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        h, context, _, _ = self.encoder(features)
        return self.scorer(h, context)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        _, context, _, glob = self.encoder(features)
        return self.value(torch.cat([context, glob], dim=-1))


class JobScoringPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, layout: ObsLayout, hidden: int = 64, **kwargs):
        self.layout = layout
        self.hidden = hidden
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = JobScoringExtractor(self.layout, self.hidden)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        # Logits come straight from the job scorer; drop SB3's dense action layer,
        # give the logit layer the small initial scale SB3 uses for action_net.
        self.action_net = nn.Identity()
        nn.init.orthogonal_(self.mlp_extractor.scorer.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.mlp_extractor.scorer.output_layer.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self) -> dict:
        return super()._get_constructor_parameters() | {"layout": self.layout, "hidden": self.hidden}


def as_policy(model: MaskablePPO):
    """Deterministic masked policy with the evaluate() signature."""

    def policy(obs: np.ndarray, env) -> int:
        action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        return int(action)

    return policy


class ValidationCallback(BaseCallback):
    """Runs the Validator every `every` steps and at the end of training."""

    def __init__(self, validator: Validator, every: int):
        super().__init__()
        self.validator, self.every = validator, every
        self.next_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            self._validate()
        return True

    def _on_training_end(self) -> None:
        self._validate()

    def _validate(self) -> None:
        self.validator(as_policy(self.model), self.num_timesteps, lambda: self.model.save(self.validator.out_dir / "best_model"))


def train_ppo(
    env_config: EnvConfig,
    out_dir: Path,
    seed: int = 0,
    timesteps: int = 300_000,
    n_envs: int = 8,
    hidden: int = 64,
    device: str = "auto",
    val_every: int = 25_000,
    val_shifts: int = 20,
    **ppo_kwargs,
) -> dict:
    """Train masked PPO, keeping the checkpoint that scores best on the validation
    shifts as out_dir/best_model.zip. `device="auto"` uses CUDA when available."""
    out_dir.mkdir(parents=True, exist_ok=True)
    venv = make_vec_env(LVHMSchedulingEnv, n_envs, seed=seed, env_kwargs={"config": env_config}, vec_env_cls=DummyVecEnv)
    model = MaskablePPO(
        JobScoringPolicy,
        venv,
        policy_kwargs={"layout": ObsLayout.from_env(LVHMSchedulingEnv(env_config)), "hidden": hidden},
        seed=seed,
        device=device,
        **(PPO_DEFAULTS | ppo_kwargs),
    )
    validator = Validator(make_instances(env_config, FAMILIAR, val_shifts, VAL_SEED), env_config, out_dir)

    start = time.perf_counter()
    model.learn(timesteps, callback=ValidationCallback(validator, val_every))
    return {
        "seed": seed,
        "timesteps": timesteps,
        "device": str(model.device),
        "train_seconds": round(time.perf_counter() - start, 1),
        "best_validation_score": validator.best,
    }


def load_ppo(path: Path, device: str = "cpu") -> Policy:
    """Deterministic policy from a saved checkpoint; single-shift evaluation is
    faster on the CPU than on a GPU."""
    return as_policy(MaskablePPO.load(path, device=device))
