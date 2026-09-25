import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from lvhm.baselines import RandomPolicy, edf, fcfs, rule_baselines
from lvhm.config import EVAL_WEIGHTS, FAMILIAR, PRODUCTS, EnvConfig
from lvhm.env import LVHMSchedulingEnv
from lvhm.evaluate import evaluate, run_episode
from lvhm.instance import in_holdout, make_instances, sample_training_features


def rollout(env, policy, **reset_kwargs):
    obs, _ = env.reset(**reset_kwargs)
    steps = []
    while True:
        assert env.observation_space.contains(obs)
        mask = env.action_masks()
        action = policy(obs, env)
        assert mask[action]
        obs, reward, terminated, truncated, info = env.step(action)
        steps.append((reward, info))
        if terminated or truncated:
            return steps, info["episode_metrics"]


def test_gymnasium_env_checker():
    # The checker steps with unmasked random actions, so start with every slot filled.
    check_env(LVHMSchedulingEnv(EnvConfig(initial_jobs=8)), skip_render_check=True)


@pytest.mark.parametrize("careful", [False, True])
def test_random_rollouts_stay_valid(careful):
    env = LVHMSchedulingEnv(EnvConfig(careful_mode=careful))
    rng = np.random.default_rng(0)

    def policy(obs, env):
        return int(rng.choice(np.flatnonzero(env.action_masks())))

    for seed in range(5):
        steps, _ = rollout(env, policy, seed=seed)
        ends = [info["end"] for _, info in steps]
        assert all(b >= a for a, b in zip(ends, ends[1:]))


def test_reward_matches_metric_decomposition():
    env = LVHMSchedulingEnv()
    alpha, beta, delta = EVAL_WEIGHTS
    for seed in range(5):
        steps, m = rollout(env, edf, seed=seed, options={"weights": EVAL_WEIGHTS})
        total = sum(r for r, _ in steps)
        expected = (
            m["on_time"] * env.cfg.on_time_bonus
            - alpha * m["tardiness"]
            - beta * m["travel"]
            - delta * m["drops"]
            - env.cfg.unfinished_penalty * m["unfinished"]
        )
        assert total == pytest.approx(m["score"])
        assert total == pytest.approx(expected)


def test_incremental_tardiness_equals_per_job_lateness():
    """Lateness charged step by step must equal each job's lateness at the moment it
    leaves the system (completion or scrap) or at shift end if still waiting."""
    env = LVHMSchedulingEnv()
    for seed in range(5):
        # overloaded, so jobs remain at shift end
        steps, m = rollout(env, fcfs, seed=seed, options={"arrival_rate": 0.2})
        inst = env._inst
        leave = {}
        attempts = {}
        for _, info in steps:
            j = info["job"]
            attempts[j] = attempts.get(j, 0) + info["dropped"]
            if not info["dropped"] or attempts[j] >= env.cfg.max_attempts:
                leave[j] = info["end"]
        t_final = steps[-1][1]["end"]
        for j in env._queue:
            leave[j] = t_final
        expected = sum(max(0.0, t - inst.deadline[j]) for j, t in leave.items())
        assert m["unfinished"] > 0
        assert m["tardiness"] == pytest.approx(expected)


def test_common_random_numbers_make_runs_reproducible():
    cfg = EnvConfig()
    insts = make_instances(cfg, FAMILIAR, n=3, seed=123)
    again = make_instances(cfg, FAMILIAR, n=3, seed=123)
    assert all(np.array_equal(a.u_drop, b.u_drop) for a, b in zip(insts, again))
    assert evaluate(edf, insts, cfg) == evaluate(edf, again, cfg)


def test_familiar_instances_use_only_familiar_products():
    inst = make_instances(EnvConfig(), FAMILIAR, n=1, seed=0)[0]
    feats = {tuple(v) for v in np.stack([inst.proc, inst.weight, inst.fragility], axis=1)}
    assert feats <= {PRODUCTS[p] for p in FAMILIAR}


def test_training_never_samples_the_p5_region():
    cfg = EnvConfig()
    feats = sample_training_features(cfg, np.random.default_rng(0), 200_000)
    assert not in_holdout(cfg, feats).any()
    assert in_holdout(cfg, np.array(PRODUCTS["P5"]))
    lows, highs = cfg.features.lows, cfg.features.highs
    assert np.all((feats >= lows) & (feats <= highs))
    assert np.any((np.array(PRODUCTS["P6"]) < lows) | (np.array(PRODUCTS["P6"]) > highs))


def test_careful_mode_reduces_drops_under_common_random_numbers():
    normal_cfg = EnvConfig(careful_mode=True)
    insts = make_instances(normal_cfg, FAMILIAR, n=20, seed=7)
    env = LVHMSchedulingEnv(normal_cfg)
    k = normal_cfg.num_slots

    def careful_fcfs(obs, env):
        return k  # slot 0, careful mode

    normal = [run_episode(env, fcfs, inst) for inst in insts]
    careful = [run_episode(env, careful_fcfs, inst) for inst in insts]
    assert env.action_space.n == 2 * k
    assert all(c["drops"] <= n["drops"] for c, n in zip(careful, normal))
    assert sum(c["drops"] for c in careful) < sum(n["drops"] for n in normal)


def test_busy_day_overflows_visible_slots():
    cfg = EnvConfig()
    env = LVHMSchedulingEnv(cfg)
    env.reset(seed=0, options={"arrival_rate": 0.24})
    max_queue = 0
    terminated = False
    while not terminated:
        max_queue = max(max_queue, len(env._queue))
        assert env.action_masks()[: cfg.num_slots].sum() == min(len(env._queue), cfg.num_slots)
        _, _, terminated, _, _ = env.step(fcfs(None, env))
    assert max_queue > cfg.num_slots


def test_training_draws_arrival_rate_per_episode():
    env = LVHMSchedulingEnv()
    counts = [len(env._inst) for seed in range(200) if env.reset(seed=seed)]
    # a fixed rate of 0.12 gives Poisson std ~7.6; U(0.06, 0.18) widens it to ~18
    assert np.std(counts) > 12
    fixed = LVHMSchedulingEnv(EnvConfig(arrival_rate_range=None))
    counts = [len(fixed._inst) for seed in range(200) if fixed.reset(seed=seed)]
    assert np.std(counts) < 10


def test_empty_slot_action_is_rejected():
    env = LVHMSchedulingEnv(EnvConfig(initial_jobs=1))
    env.reset(seed=0, options={"arrival_rate": 0.0})
    with pytest.raises(ValueError):
        env.step(3)


def test_all_baselines_run():
    cfg = EnvConfig()
    insts = make_instances(cfg, FAMILIAR, n=2, seed=0)
    for name, policy in rule_baselines().items():
        assert len(evaluate(policy, insts, cfg)) == 2, name
    assert isinstance(RandomPolicy(1)(None, _reset(LVHMSchedulingEnv())), int)


def _reset(env):
    env.reset(seed=0)
    return env
