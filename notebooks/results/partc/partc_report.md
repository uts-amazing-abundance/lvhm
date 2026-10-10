# Assignment 3, Part C — Initial Experimental Results

**Project:** RL for Adaptive Job Sequencing in Low-Volume High-Mix Robotic Work Cells  
**Team:** Amazing Abundance  
**Generated from:** `experiments_phase2_1.ipynb`, profile `partc`  
> **Training budget.** Each learned policy saw 100,000 decisions (about 1299 shifts of experience). This is deliberately short for an initial report: it is enough to show the pipeline runs end to end and that learning has begun, but **the learned policies have not converged** and may not yet beat the baselines. The comparison below should be read as a starting point, not a result. The full study uses 300,000 steps across 10 seeds.

## Algorithms / methods chosen

The problem is formulated as a sequential decision problem: whenever the arm becomes free, choose which waiting job to serve next and how carefully to handle it. Future arrivals are unobservable, so this is a partially observable MDP.

| Method | Type | Role |
|---|---|---|
| FCFS, SPT, EDF, NTF, LRF, Random | Single-attribute dispatch rules | Conventional baselines — what factories run today |
| ATCSR+C | Composite priority index (Vepsalainen & Morton; Lee, Bhaskaran & Pinedo), tuned on validation shifts | Combines urgency, processing time and setup cost, which no single-attribute rule does |
| Rolling CP-SAT | Constraint-programming solver, re-solved at every decision | Provably optimal on the visible queue — isolates what learning adds beyond optimisation |
| Masked PPO | On-policy deep RL (sb3-contrib MaskablePPO) | Learned policy |
| Masked double-dueling DQN | Off-policy deep RL (PyTorch) | Learned policy |
| Clairvoyant CP-SAT | Perfect-information solver | **Upper bound, not a baseline** — it sees the whole shift |

Both learned agents share one permutation-invariant network: the same MLP embeds every waiting job, the embeddings are pooled into a queue context, and each job is scored from its own embedding plus that context (a Deep Sets architecture). Invalid actions are masked.

## Experimental settings

### Problem formulation

| Setting | Value |
|---|---|
| Episode length | 480 s (8 minutes) |
| Arrival process | Poisson, 0.14 jobs/s (≈67 jobs/shift) |
| Input trays | 2, switching costs 1.15 s |
| Observation | 79 values: 8 job slots × 9 features + 7 global |
| Action | Discrete(16): 8 slots × (normal, careful), invalid masked |
| Reward | +1.0 on time · −α/s lateness · −β/s switching · −δ/drop · −1.0 unfinished |
| Evaluation weights (α, β, δ) | (0.015, 0.15, 0.6) |
| Products | 6 total; 4 for training, P5 and P6 held out |

### Training

| Setting | PPO | DQN |
|---|---|---|
| Timesteps | 100,000 | 100,000 |
| Seeds | 3 ([0, 1, 2]) | 3 ([0, 1, 2]) |
| Learning rate | 0.0003 | 0.0001 |
| Discount γ | 0.99 | 0.99 |
| Batch size | 256 | 128 |
| Exploration | entropy coef 0.01 | ε 1.0→0.05 over 30% |
| Parallel envs | 8 | 8 |
| Replay buffer | — | 100,000 |

### Evaluation

| Setting | Value |
|---|---|
| Test shifts per condition | 30 |
| Validation shifts (checkpoint selection) | 10, separate seed |
| Conditions | 6 |
| Variance control | Common random numbers — every policy faces identical pre-drawn shifts, including the draw behind every possible drop |
| ATCSR tuning | grid search on validation shifts, best K = (8.0, 2.0, 1.0) |

## Initial results

### Normal day, familiar products (mean ± std over 30 shifts)

| policy | on_time | tardiness | travel | drops | scrapped | unfinished | careful | score |
|---|---|---|---|---|---|---|---|---|
| Random | 59.43 ± 5.31 | 143.26 ± 223.87 | 41.02 ± 7.98 | 8.73 ± 3.07 | 1.13 ± 1.04 | 1.43 ± 1.89 | 0.00 ± 0.00 | 44.46 ± 9.01 |
| FCFS | 59.53 ± 5.98 | 76.71 ± 136.92 | 39.67 ± 5.76 | 8.63 ± 3.03 | 1.13 ± 1.04 | 1.20 ± 1.61 | 0.00 ± 0.00 | 46.05 ± 8.44 |
| SPT | 61.60 ± 4.88 | 152.83 ± 248.73 | 40.17 ± 7.20 | 8.63 ± 3.02 | 1.13 ± 1.04 | 1.17 ± 1.66 | 0.00 ± 0.00 | 46.93 ± 7.00 |
| LRF | 61.27 ± 4.78 | 158.24 ± 250.41 | 40.36 ± 6.90 | 8.63 ± 3.02 | 1.13 ± 1.04 | 1.23 ± 1.68 | 0.00 ± 0.00 | 46.42 ± 6.74 |
| EDF | 63.13 ± 7.20 | 44.00 ± 120.17 | 40.67 ± 6.95 | 8.63 ± 3.03 | 1.13 ± 1.04 | 1.37 ± 1.81 | 0.00 ± 0.00 | 49.83 ± 9.68 |
| NTF | 62.33 ± 6.37 | 54.92 ± 96.88 | 29.09 ± 4.73 | 8.70 ± 3.04 | 1.13 ± 1.04 | 0.93 ± 1.51 | 0.00 ± 0.00 | 50.99 ± 6.68 |
| ATCSR+C | 61.80 ± 5.20 | 61.16 ± 103.99 | 26.60 ± 3.99 | 3.13 ± 1.89 | 0.37 ± 0.56 | 1.30 ± 1.51 | 50.00 ± 6.62 | 53.71 ± 5.72 |
| RollCP | 62.20 ± 5.77 | 61.23 ± 119.43 | 29.02 ± 3.45 | 3.10 ± 1.92 | 0.27 ± 0.52 | 1.50 ± 1.78 | 49.27 ± 8.52 | 53.57 ± 5.76 |
| PPO | 61.80 ± 5.64 | 141.26 ± 222.02 | 35.25 ± 5.18 | 8.63 ± 2.95 | 1.13 ± 1.03 | 1.17 ± 1.71 | 0.00 ± 0.00 | 48.05 ± 6.19 |
| DQN | 63.54 ± 5.86 | 64.74 ± 109.91 | 28.88 ± 3.97 | 4.69 ± 2.82 | 0.54 ± 0.78 | 1.16 ± 1.53 | 34.79 ± 10.38 | 54.27 ± 5.61 |

### Score across all test conditions

| policy | familiar | unseen_p5 | unseen_p6 | busy | damage_hi | damage_oor |
|---|---|---|---|---|---|---|
| Random | 44.46 ± 9.01 | 43.75 ± 9.30 | 36.40 ± 16.09 | -167.12 ± 57.10 | 37.45 ± 9.64 | 29.80 ± 11.54 |
| FCFS | 46.05 ± 8.44 | 45.37 ± 8.41 | 37.74 ± 15.94 | -163.34 ± 59.54 | 38.28 ± 9.99 | 30.51 ± 11.98 |
| SPT | 46.93 ± 7.00 | 45.62 ± 6.73 | 41.50 ± 10.59 | -150.24 ± 59.79 | 39.16 ± 8.56 | 31.39 ± 10.60 |
| LRF | 46.42 ± 6.74 | 45.78 ± 6.76 | 41.29 ± 10.51 | -151.11 ± 59.51 | 38.65 ± 8.42 | 30.88 ± 10.54 |
| EDF | 49.83 ± 9.68 | 48.93 ± 9.89 | 42.31 ± 17.25 | -162.30 ± 59.91 | 42.06 ± 10.95 | 34.29 ± 12.69 |
| NTF | 50.99 ± 6.68 | 50.73 ± 6.63 | 45.18 ± 9.14 | -110.21 ± 59.24 | 43.16 ± 7.67 | 35.33 ± 9.39 |
| ATCSR+C | 53.71 ± 5.72 | 52.80 ± 7.81 | 47.83 ± 10.37 | -111.36 ± 58.46 | 47.49 ± 10.06 | 44.42 ± 11.81 |
| RollCP | 53.57 ± 5.76 | 53.19 ± 6.89 | 47.15 ± 10.16 | -114.75 ± 59.68 | 46.71 ± 8.76 | 42.38 ± 11.79 |
| PPO | 48.05 ± 6.19 | 47.59 ± 6.35 | 41.12 ± 11.10 | -149.00 ± 64.09 | 39.99 ± 7.36 | 31.90 ± 9.53 |
| DQN | 54.27 ± 5.61 | 53.97 ± 6.32 | 48.89 ± 8.41 | -110.71 ± 60.08 | 49.59 ± 6.59 | 42.79 ± 9.13 |

Conditions: **familiar** — Normal day, familiar products P1-P4 · **unseen_p5** — P1-P4 plus unseen P5 (inside the training range) · **unseen_p6** — P1-P4 plus unseen P6 (outside the training range) · **busy** — Busy day, arrival rate doubled · **damage_hi** — Damage weight at the top of the training range (boundary, not extrapolation) · **damage_oor** — Damage weight outside the training range (true extrapolation)

### Reference ceiling

CP-SAT clairvoyant bound on 10 shifts, 3 solved to proven optimality. Mean achievable-with-hindsight score **67.91**, proven ceiling **70.02**.

The bound relaxes drops away, so it is an upper bound rather than the true optimum. Scores are therefore reported as *headroom captured* between the best baseline and this ceiling, not as raw values.

### Paired comparison

Because every policy faces identical shifts, each shift is a paired observation. Reported over 30 shifts on the familiar condition.

| policy | versus | mean_diff | ci_low | ci_high | win_rate |
|---|---|---|---|---|---|
| DQN | Random | 9.814 | 6.879 | 12.749 | 0.967 |
| DQN | FCFS | 8.221 | 5.609 | 10.834 | 1.0 |
| DQN | LRF | 7.848 | 5.659 | 10.037 | 0.967 |
| DQN | SPT | 7.338 | 5.197 | 9.479 | 0.967 |
| DQN | EDF | 4.447 | 1.58 | 7.313 | 0.867 |
| DQN | NTF | 3.281 | 2.48 | 4.081 | 1.0 |
| DQN | RollCP | 0.704 | -0.182 | 1.59 | 0.6 |
| DQN | ATCSR+C | 0.561 | -0.481 | 1.602 | 0.567 |
| PPO | Random | 3.588 | 1.621 | 5.555 | 0.767 |
| PPO | FCFS | 1.995 | 0.18 | 3.81 | 0.6 |
| PPO | LRF | 1.621 | 0.424 | 2.819 | 0.667 |
| PPO | SPT | 1.111 | -0.058 | 2.281 | 0.467 |

### Validation of the measurement

On 5 deterministic instances of 7 jobs — all arrivals at t=0, no timing noise, no drops — every ordering was enumerated, so the optimum is known rather than estimated.

- CP-SAT reproduced the brute-force optimum on every instance: **True**
- No policy scored above the true optimum, as must be the case.


_Generated 2026-10-04 13:13 · profile `partc` · 3 seeds · 100,000 steps · 30 test shifts._