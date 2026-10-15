# Assignment 3, Part C — Initial Experimental Results

**Project:** RL for Adaptive Job Sequencing in Low-Volume High-Mix Robotic Work Cells  
**Team:** Amazing Abundance  
**Generated from:** `experiments_phase2_1.ipynb`, profile `full`  
> **Note:** this run used the `full` profile, not `partc`.

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
| Timesteps | 1,000,000 | 1,000,000 |
| Seeds | 10 ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) | 10 ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) |
| Learning rate | 0.0003 | 0.0001 |
| Discount γ | 0.99 | 0.99 |
| Batch size | 256 | 128 |
| Exploration | entropy coef 0.01 | ε 1.0→0.05 over 30% |
| Parallel envs | 8 | 8 |
| Replay buffer | — | 100,000 |

### Evaluation

| Setting | Value |
|---|---|
| Test shifts per condition | 100 |
| Validation shifts (checkpoint selection) | 20, separate seed |
| Conditions | 6 |
| Variance control | Common random numbers — every policy faces identical pre-drawn shifts, including the draw behind every possible drop |
| ATCSR tuning | grid search on validation shifts, best K = (3.175, 1.26, 1.0) |

## Initial results

### Normal day, familiar products (mean ± std over 100 shifts)

| policy | on_time | tardiness | travel | drops | scrapped | unfinished | careful | score |
|---|---|---|---|---|---|---|---|---|
| Random | 59.77 ± 5.66 | 136.46 ± 191.02 | 41.74 ± 6.93 | 9.08 ± 3.02 | 0.97 ± 0.88 | 1.01 ± 1.51 | 0.00 ± 0.00 | 45.00 ± 8.35 |
| FCFS | 59.86 ± 6.51 | 73.13 ± 112.29 | 39.54 ± 6.24 | 9.05 ± 3.02 | 0.98 ± 0.91 | 0.89 ± 1.32 | 0.00 ± 0.00 | 46.51 ± 8.27 |
| SPT | 61.57 ± 5.21 | 138.03 ± 183.19 | 40.40 ± 6.90 | 9.03 ± 3.03 | 0.97 ± 0.92 | 0.90 ± 1.34 | 0.00 ± 0.00 | 47.12 ± 6.19 |
| LRF | 61.34 ± 5.19 | 147.56 ± 191.92 | 40.76 ± 6.69 | 9.04 ± 3.03 | 0.97 ± 0.92 | 0.92 ± 1.35 | 0.00 ± 0.00 | 46.67 ± 6.27 |
| EDF | 64.05 ± 7.88 | 33.54 ± 96.54 | 40.82 ± 6.39 | 9.05 ± 3.02 | 0.98 ± 0.91 | 0.96 ± 1.46 | 0.00 ± 0.00 | 51.03 ± 9.50 |
| NTF | 62.47 ± 5.98 | 48.81 ± 70.65 | 28.81 ± 4.68 | 9.08 ± 3.03 | 0.98 ± 0.91 | 0.73 ± 1.21 | 0.00 ± 0.00 | 51.24 ± 6.30 |
| ATCSR+C | 62.87 ± 5.67 | 41.26 ± 68.24 | 28.37 ± 3.97 | 3.87 ± 2.41 | 0.33 ± 0.57 | 1.05 ± 1.30 | 49.52 ± 6.88 | 54.62 ± 5.98 |
| RollCP | 62.85 ± 5.55 | 50.28 ± 83.52 | 28.65 ± 3.86 | 3.81 ± 2.26 | 0.28 ± 0.49 | 1.15 ± 1.45 | 49.31 ± 7.10 | 54.36 ± 5.60 |
| PPO | 63.62 ± 5.89 | 68.63 ± 115.44 | 32.98 ± 5.30 | 8.30 ± 3.08 | 0.86 ± 0.89 | 0.84 ± 1.28 | 6.92 ± 8.56 | 51.82 ± 6.14 |
| DQN | 65.84 ± 6.12 | 51.30 ± 107.77 | 31.08 ± 4.35 | 5.07 ± 2.67 | 0.45 ± 0.65 | 0.98 ± 1.36 | 38.06 ± 8.08 | 56.38 ± 5.89 |

### Score across all test conditions

| policy | familiar | unseen_p5 | unseen_p6 | busy | damage_hi | damage_oor |
|---|---|---|---|---|---|---|
| Random | 45.00 ± 8.35 | 44.82 ± 7.93 | 37.15 ± 12.49 | -175.44 ± 52.81 | 37.21 ± 9.49 | 28.68 ± 11.83 |
| FCFS | 46.51 ± 8.27 | 46.08 ± 8.15 | 38.49 ± 12.91 | -173.94 ± 51.38 | 38.37 ± 9.75 | 30.22 ± 11.68 |
| SPT | 47.12 ± 6.19 | 46.25 ± 6.32 | 41.29 ± 8.03 | -161.21 ± 52.74 | 38.99 ± 7.86 | 30.87 ± 10.01 |
| LRF | 46.67 ± 6.27 | 46.25 ± 6.49 | 41.13 ± 8.08 | -161.72 ± 52.60 | 38.53 ± 7.95 | 30.40 ± 10.10 |
| EDF | 51.03 ± 9.50 | 50.57 ± 9.47 | 43.74 ± 14.24 | -173.33 ± 52.17 | 42.89 ± 10.78 | 34.74 ± 12.52 |
| NTF | 51.24 ± 6.30 | 51.02 ± 6.11 | 45.59 ± 7.59 | -120.04 ± 52.21 | 43.07 ± 7.56 | 34.89 ± 9.46 |
| ATCSR+C | 54.62 ± 5.98 | 54.20 ± 6.35 | 48.64 ± 8.60 | -123.09 ± 50.24 | 48.18 ± 8.68 | 44.10 ± 11.21 |
| RollCP | 54.36 ± 5.60 | 54.11 ± 5.51 | 48.70 ± 6.98 | -124.58 ± 52.64 | 47.56 ± 7.14 | 42.99 ± 9.71 |
| PPO | 51.82 ± 6.14 | 51.53 ± 6.22 | 45.99 ± 7.62 | -146.10 ± 57.90 | 44.39 ± 7.64 | 36.97 ± 10.01 |
| DQN | 56.38 ± 5.89 | 56.04 ± 5.82 | 51.24 ± 6.97 | -126.62 ± 55.95 | 51.72 ± 6.64 | 46.53 ± 8.43 |

Conditions: **familiar** — Normal day, familiar products P1-P4 · **unseen_p5** — P1-P4 plus unseen P5 (inside the training range) · **unseen_p6** — P1-P4 plus unseen P6 (outside the training range) · **busy** — Busy day, arrival rate doubled · **damage_hi** — Damage weight at the top of the training range (boundary, not extrapolation) · **damage_oor** — Damage weight outside the training range (true extrapolation)

### Reference ceiling

CP-SAT clairvoyant bound on 20 shifts, 2 solved to proven optimality. Mean achievable-with-hindsight score **63.66**, proven ceiling **67.64**.

The bound relaxes drops away, so it is an upper bound rather than the true optimum. Scores are therefore reported as *headroom captured* between the best baseline and this ceiling, not as raw values.

### Paired comparison

Because every policy faces identical shifts, each shift is a paired observation. Reported over 100 shifts on the familiar condition.

| policy | versus | mean_diff | ci_low | ci_high | win_rate |
|---|---|---|---|---|---|
| DQN | Random | 11.374 | 9.983 | 12.765 | 1.0 |
| DQN | FCFS | 9.865 | 8.565 | 11.164 | 0.99 |
| DQN | LRF | 9.708 | 8.709 | 10.706 | 1.0 |
| DQN | SPT | 9.255 | 8.285 | 10.226 | 1.0 |
| DQN | EDF | 5.344 | 4.028 | 6.66 | 0.91 |
| DQN | NTF | 5.138 | 4.57 | 5.706 | 0.97 |
| DQN | RollCP | 2.014 | 1.504 | 2.525 | 0.8 |
| DQN | ATCSR+C | 1.754 | 1.242 | 2.265 | 0.74 |
| PPO | Random | 6.818 | 5.644 | 7.993 | 0.96 |
| PPO | FCFS | 5.309 | 4.202 | 6.416 | 0.96 |
| PPO | LRF | 5.152 | 4.393 | 5.912 | 0.99 |
| PPO | SPT | 4.7 | 3.964 | 5.435 | 0.97 |

### Validation of the measurement

On 5 deterministic instances of 7 jobs — all arrivals at t=0, no timing noise, no drops — every ordering was enumerated, so the optimum is known rather than estimated.

- CP-SAT reproduced the brute-force optimum on every instance: **True**
- No policy scored above the true optimum, as must be the case.


_Generated 2026-10-15 12:09 · profile `full` · 10 seeds · 1,000,000 steps · 100 test shifts._