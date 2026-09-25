import numpy as np
import torch
from sb3_contrib import MaskablePPO

from lvhm.agents.networks import JobScorer, ObsLayout, QueueEncoder
from lvhm.agents.ppo import JobScoringPolicy, as_policy
from lvhm.config import FAMILIAR, EnvConfig
from lvhm.env import LVHMSchedulingEnv
from lvhm.evaluate import evaluate
from lvhm.instance import make_instances


def _obs_batch(n=16, seed=0):
    env = LVHMSchedulingEnv()
    obs = []
    for s in range(n):
        o, _ = env.reset(seed=seed + s, options={"arrival_rate": 0.25})
        for _ in range(s % 5):
            o, *_ = env.step(0)
        obs.append(o)
    return env, torch.tensor(np.stack(obs))


def _scores(env, obs, seed=0):
    torch.manual_seed(seed)
    layout = ObsLayout.from_env(env)
    enc = QueueEncoder(layout)
    scorer = JobScorer(layout, 64, enc.context_dim)
    h, ctx, _, _ = enc(obs)
    return layout, scorer(h, ctx)


def test_scores_follow_a_permutation_of_job_slots():
    env, obs = _obs_batch()
    layout, base = _scores(env, obs)
    k, s = layout.num_slots, layout.slot_dim
    perm = torch.randperm(k)
    slots = obs[:, : k * s].reshape(-1, k, s)[:, perm].reshape(obs.shape[0], -1)
    _, permuted = _scores(env, torch.cat([slots, obs[:, k * s :]], dim=1))
    for mode in range(layout.num_modes):
        block = slice(mode * k, (mode + 1) * k)
        assert torch.allclose(permuted[:, block], base[:, block][:, perm], atol=1e-5)


def test_padded_slots_do_not_change_real_scores():
    env, obs = _obs_batch()
    layout, base = _scores(env, obs)
    k, s = layout.num_slots, layout.slot_dim
    slots = obs[:, : k * s].reshape(-1, k, s).clone()
    empty = slots[..., 0] < 0.5
    slots[empty] = torch.rand_like(slots[empty]) * 0.5  # garbage padding, presence still 0
    slots[..., 0][empty] = 0.0
    _, noisy = _scores(env, torch.cat([slots.reshape(obs.shape[0], -1), obs[:, k * s :]], dim=1))
    valid = torch.tensor(np.stack([np.tile(~e.numpy(), layout.num_modes) for e in empty]))
    assert torch.allclose(noisy[valid], base[valid], atol=1e-5)


def test_masked_ppo_trains_saves_and_acts_validly(tmp_path):
    cfg = EnvConfig()
    env = LVHMSchedulingEnv(cfg)
    model = MaskablePPO(
        JobScoringPolicy,
        env,
        n_steps=128,
        batch_size=64,
        n_epochs=1,
        policy_kwargs={"layout": ObsLayout.from_env(env)},
        seed=0,
        device="cpu",
    )
    model.learn(256)
    model.save(tmp_path / "ppo")
    loaded = MaskablePPO.load(tmp_path / "ppo", device="cpu")
    results = evaluate(as_policy(loaded), make_instances(cfg, FAMILIAR, 2, seed=1), cfg)
    assert len(results) == 2


def test_dueling_q_masks_invalid_actions_and_trains(tmp_path):
    from lvhm.agents.dqn import DQNConfig, DuelingJobQNet, as_policy as dqn_policy, train_dqn

    env, obs = _obs_batch()
    layout = ObsLayout.from_env(env)
    net = DuelingJobQNet(layout)
    mask = obs[:, : layout.num_slots * layout.slot_dim].reshape(-1, layout.num_slots, layout.slot_dim)[..., 0] > 0.5
    mask = mask.repeat(1, layout.num_modes)
    q = net(obs, mask)
    assert torch.all(q[~mask] < -1e8)
    assert torch.all(q.argmax(1) < layout.num_actions) and mask[torch.arange(len(q)), q.argmax(1)].all()

    cfg = EnvConfig()
    seen = []
    trained = train_dqn(cfg, DQNConfig(timesteps=800, n_envs=4, learning_starts=200, batch_size=32), 0, lambda n, t: seen.append(t), 400)
    assert seen[0] == 0 and seen[-1] >= 800
    assert len(evaluate(dqn_policy(trained), make_instances(cfg, FAMILIAR, 1, seed=2), cfg)) == 1
