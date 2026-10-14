# Job sequence export for gui/job_sequence_dashboard.html.
#
# Paste as a cell at the end of experiments_phase2_1.ipynb, after section 17 (it needs
# CONDITIONS, POLICIES_RULES and the trained checkpoints). It replays every policy on the first
# DASH_SHIFTS test shifts of each condition and records each pick: which job, when, in which
# mode, and how it ended. Same pre-drawn shifts as the results table, so the sequences are the
# ones behind those numbers. Learned policies use their seed-0 checkpoint only.
#
# Writes results/<profile>/job_sequences.json. Rebuild the dashboard with
#     python gui/build_dashboard.py
# RollCP dominates the cost (about 0.1 s per decision), so keep DASH_SHIFTS small.

DASH_SHIFTS = 10
OUTCOME = {'on_time': 0, 'late': 1, 'drop': 2, 'scrap': 3}   # 'drop' = dropped, job stays queued


def product_of(w, f):
    for name, (pw, pf) in PRODUCTS.items():
        if abs(pw - w) < 1e-9 and abs(pf - f) < 1e-9:
            return name
    return '?'


def trace_episode(env, policy, instance, weights):
    obs, _ = env.reset(options={'instance': instance, 'weights': weights})
    steps = []
    while True:
        queued = len(env._queue)
        obs, reward, terminated, truncated, info = env.step(policy(obs, env))
        j = info['job']
        if info['dropped']:
            out = OUTCOME['scrap'] if env._attempts[j] >= env.cfg.max_attempts else OUTCOME['drop']
        else:
            out = OUTCOME['on_time'] if info['end'] <= instance.deadline[j] else OUTCOME['late']
        steps.append([int(j), round(info['start'], 3), round(info['end'], 3), round(info['travel'], 3),
                      int(info['careful']), out, queued, round(reward, 4)])
        if terminated or truncated:
            return steps, info['episode_metrics']


dash_policies = dict(POLICIES_RULES)
for algo, loader, ext in [('ppo', load_ppo, 'zip'), ('dqn', load_dqn, 'pt')]:
    path = RUN_DIR / algo / f'seed{SEEDS[0]}' / f'best_model.{ext}'
    if path.exists():
        dash_policies[algo.upper()] = loader(path)

export = {
    'profile': PROFILE,
    'shift_length': ENV.shift_length,
    'eval_weights': list(EVAL_WEIGHTS),
    'products': {k: list(v) for k, v in PRODUCTS.items()},
    'policies': list(dash_policies),
    'step_fields': ['job', 'start', 'end', 'travel', 'careful', 'outcome', 'queue', 'reward'],
    'outcomes': list(OUTCOME),
    'conditions': [],
}
t_start = time.time()
for cond in CONDITIONS:
    cfg = replace(ENV, arrival_rate=cond.arrival_rate)
    shifts = make_instances(cfg, cond.products, N_TEST_SHIFTS, TEST_SEED)[:DASH_SHIFTS]
    env = LVHMSchedulingEnv(cfg)
    c_out = {'name': cond.name, 'description': cond.description, 'weights': list(cond.weights),
             'arrival_rate': cond.arrival_rate, 'shifts': []}
    for s, inst in enumerate(shifts):
        jobs = [[round(float(inst.arrival[j]), 3), round(float(inst.deadline[j]), 3), int(inst.tray[j]),
                 product_of(inst.weight[j], inst.fragility[j]), round(float(inst.proc[j]), 3)]
                for j in range(len(inst))]
        runs = {}
        for name, policy in dash_policies.items():
            steps, metrics = trace_episode(env, policy, inst, cond.weights)
            runs[name] = {'steps': steps, 'metrics': {k: round(float(v), 4) for k, v in metrics.items()}}
        c_out['shifts'].append({'shift': s, 'jobs': jobs, 'runs': runs})
    export['conditions'].append(c_out)
    print(f'{cond.name}: {len(shifts)} shifts traced  ({time.time() - t_start:.0f}s)')

out = RESULTS_DIR / 'job_sequences.json'
out.write_text(json.dumps(export, separators=(',', ':')))
print(f'-> {out}  ({out.stat().st_size / 1e6:.1f} MB)')
