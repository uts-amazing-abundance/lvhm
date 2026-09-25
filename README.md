# LVHM: RL job sequencing for an SO-ARM101 work cell

UTS 43008 Reinforcement Learning, Assignment 3, project [68] *Reinforcement Learning for
Adaptive Job Sequencing in Low-Volume High-Mix Robotic Work Cells* (team Amazing Abundance).

An SO-ARM101 arm serves jobs that arrive at two input trays over a shift. Whenever the arm is
idle, a policy picks the next job and how carefully to handle it, trading on-time completion
against station switches and damage. Learned policies (masked PPO, masked double dueling DQN)
are compared with six dispatch rules on identical shifts.

> **Start here: [`notebooks/experiments.ipynb`](notebooks/experiments.ipynb).**
> Open the notebook first. It walks through the whole experiment for Part C from top to bottom,
> with every cell commented. It is self-contained, so on Amazon SageMaker (T4 GPU) you only
> upload that one file. The rest of this README documents the environment and design decisions
> behind it.

## Experiments notebook

**Run the Part C experiments from [`notebooks/experiments.ipynb`](notebooks/experiments.ipynb).**
The notebook holds the whole environment, the six dispatch rules, masked PPO, masked double
dueling DQN and the evaluation; it does not import this repository's package. It trains both
agents (one run per seed), plots the learning curves, evaluates every policy on the four Part B
test conditions, and writes the report tables to `results/partc_tables.md`. The settings (seeds,
training steps, retraining) are in one cell near the top.

**Amazon SageMaker (T4 GPU)**

1. Download `notebooks/experiments.ipynb` from this repository.
2. Upload it to SageMaker (JupyterLab: Upload Files) on an instance with a T4 GPU, for example
   `ml.g4dn.xlarge`, and pick a PyTorch kernel.
3. Run all cells. The first cell installs the packages it needs with pip
   (`gymnasium`, `sb3-contrib`, `pandas`, `matplotlib`; torch comes with the PyTorch kernel).
   Training uses CUDA when it is available.

**Locally:** `uv sync --extra notebook`, then
`uv run --extra notebook --with jupyterlab jupyter lab notebooks/experiments.ipynb`, and skip the
pip cell.

The notebook mirrors the code in `src/lvhm`, which the tests and command-line scripts use. A
change to the environment or the agents has to be made in both places. The same training is
available from the command line:

```bash
uv run python scripts/train_ppo.py --timesteps 300000 --seed 0 --device auto
uv run python scripts/train_dqn.py --timesteps 300000 --seed 0 --device auto
```

PPO uses sb3-contrib's `MaskablePPO`. The DQN is plain PyTorch (`src/lvhm/agents/dqn.py`)
because SB3's `DQN` supports neither action masking nor the double-DQN target and dueling head
from the Part B design. Both agents score each waiting job with the same shared-weight network
(`src/lvhm/agents/networks.py`): a per-job MLP plus a pooled whole-queue context (Deep Sets).
Checkpoints are selected on 20 validation shifts (seed 20000), separate from the 100 test
shifts (seed 10000).

## Quick start

```bash
uv sync                                               # Python 3.12 + dependencies
uv run pytest                                         # environment and agent tests
uv run python scripts/baseline_check.py               # six rules on 100 normal-day shifts
uv run python scripts/baseline_check.py --busy        # busy day (arrival rate doubled)
uv run --group sim python scripts/calibrate_so101.py  # re-measure SO-ARM101 timing in MuJoCo
```

## Environment

`lvhm.LVHMSchedulingEnv` (also registered as `LVHM-v0`) is a discrete-event Gymnasium
environment. Time is in seconds; one episode is an 8-minute run of the cell (480 s).

- **Decision points.** A decision happens whenever the arm is idle and the queue is non-empty.
  The chosen job is simulated to completion (station switch, pick-and-place, possible drop) in
  one step, and jobs that arrived meanwhile join the queue. An empty queue advances the clock
  to the next arrival. The episode terminates when the shift clock expires.
- **Jobs** are described only by features, never by product identity: weight, fragility,
  source tray and deadline. Processing time follows from weight and fragility (see below).
- **Drops.** Each pick drops the item with probability
  `0.01 + 0.20·fragility + 0.05·weight/0.25`. A dropped job loses half its processing time and
  is re-queued; a second drop scraps it.

### Observation (79 values)

The eight oldest waiting jobs (ordered by arrival, zero-padded) plus global features.

| Group | Feature | Size | Encoding |
|---|---|---|---|
| Per slot (×8) | presence flag | 1 | 0/1, drives the action mask |
| | nominal processing time | 1 | seconds / 8, clipped to [0, 1] |
| | weight | 1 | kg / 0.25, clipped to [0, 1] |
| | fragility | 1 | [0, 1] |
| | time to deadline | 1 | seconds / 60, clipped to [−1, 1] |
| | source tray | 2 | one-hot |
| | waiting time | 1 | seconds / 120, clipped to [0, 1] |
| | dropped once | 1 | 0/1 |
| Global | shift remaining | 1 | fraction of the shift |
| | arm station | 2 | one-hot |
| | queue length | 1 | jobs / 8, up to 4 (jobs beyond slot 8 are not shown) |
| | penalty weights (α, β, δ) | 3 | normalised to the training range, clipped to [0, 2] |

### Action

`Discrete(16)`: action `a` serves slot `a % 8` in normal mode (`a < 8`) or careful mode
(`a ≥ 8`). Careful mode slows the loaded motion by 1.5× and multiplies the drop probability by
0.3. Empty slots are masked (`env.action_masks()`, the method sb3-contrib's `MaskablePPO` uses);
selecting one raises an error.

### Reward

| Term | Value |
|---|---|
| On-time completion | +1 per job finished by its deadline |
| Tardiness | −α per second of lateness, charged as it accrues for every job in the system |
| Station switch | −β per second of switching |
| Damage | −δ per drop |
| Unfinished jobs | −1 per job still waiting when the shift ends |

(α, β, δ) are sampled per training episode and given to the agent in the observation, so the
penalty structure can change at test time without retraining. Evaluation uses
(0.015, 0.15, 0.6).

### Training and evaluation splits

| Axis | Training | Evaluation |
|---|---|---|
| Products | weight U(0.02, 0.20) kg, fragility U(0, 0.9), excluding a box of ±10% of each range around P5 | P1–P4 (familiar), P5 (inside the range), P6 (outside it) |
| Arrival rate | U(0.07, 0.21) jobs/s per episode | 0.14 (normal day), 0.28 (busy day, outside the range) |
| Penalty weights | α U(0.005, 0.03), β U(0.05, 0.30), δ U(0.2, 1.5) | fixed; δ raised for the changed-penalty test |

| Product | Weight (kg) | Fragility | Nominal processing (s) | Drop probability (normal mode) |
|---|---|---|---|---|
| P1 | 0.03 | 0.10 | 3.63 | 0.036 |
| P2 | 0.08 | 0.60 | 4.67 | 0.146 |
| P3 | 0.18 | 0.20 | 4.67 | 0.086 |
| P4 | 0.05 | 0.85 | 4.86 | 0.190 |
| P5 (held out) | 0.12 | 0.45 | 4.68 | 0.124 |
| P6 (held out) | 0.23 | 0.95 | 6.07 | 0.246 |

Evaluation shifts are pre-drawn (`lvhm.make_instances`): arrivals, deadlines, processing noise
and the random draw behind every possible drop are fixed per job and attempt. Every policy
therefore faces the same jobs and the same drop outcome for the same pick (common random
numbers). The 100 test shifts use seed 10000.

## SO-ARM101 grounding

Motion timing is measured in MuJoCo with the SO101 model published by The Robot Studio
(`assets/so101`, see its README for provenance and licence), whose actuators carry STS3215
servo parameters. `scripts/calibrate_so101.py` writes `assets/so101/cell_timing.json`, and a
test keeps `EnvConfig` in sync with it.

**Cell layout.** Two stations, one per tray, at ±60° from the arm's forward axis. Each station
holds its input tray and its own output spot side by side (±20° from the station centre) at a
radius of 18 cm, inside the region the SO101 can reach with a top-down grasp. Consecutive jobs
from the same tray need a short swing; switching trays swings the arm across the cell.

**Measurement.** Top-down grasp waypoints (hover at 8 cm, grasp/place at 3 cm) are solved by
damped least-squares IK. Each move is commanded as a linear joint-space trajectory at 1.5 rad/s
on the joint that moves furthest and simulated until the servos settle.

| Segment | Time (s) |
|---|---|
| Empty: swing to the tray, descend, close the gripper | 1.388 |
| Loaded: lift, swing to the output, descend, release, rise | 0.929 + 0.987 × slowdown |
| Extra swing when switching station | 1.150 |

**Processing time.** Heavy or fragile items are moved more slowly:
`slowdown = 1 + 1.5·weight/0.25 + 1.5·fragility`, and
`processing = empty + loaded(slowdown)`, times log-normal noise (σ = 0.1) per attempt.
Careful mode multiplies the slowdown by 1.5.

**Payload check.** Holding the heaviest product (0.23 kg) at a hover pose needs 0.75 N·m at the
shoulder, against a servo force range of 3.35 N·m.

### Assumptions

- The layout, the 1.5 rad/s operating joint speed and the slowdown coefficients are design
  choices, not measurements. The switch-to-processing time ratio (about 0.25) depends on the
  station spacing, so moving the stations changes how much tray batching matters.
- Motion timing is simulated without payload; the payload only enters through the handling
  slowdown and the static torque check.
- The joint speed should be validated by timing the same waypoint moves on the real arm with
  LeRobot. The waypoints in `cell_timing.json` are the ones the browser replay and the hardware
  demo will use.
- Rule-based baselines always pick in normal mode; careful mode is available only to the
  learned policies.

## Changes from the Part B design

| Change | Reason |
|---|---|
| Tardiness is charged as it accrues instead of at completion | With a fixed unfinished-job penalty, a very late job was cheaper to never serve |
| Careful pick mode; the action space grows from 8 to 16 | Drop probability depended only on the job, so no ordering could reduce drops (every rule had the same drop count), which made LRF and the changed-penalty test uninformative |
| P5 is held out as a region, not a single vector | With continuous sampling the exact P5 vector is never drawn, so the Part B assertion excluded nothing |
| Arrival rate is sampled per training episode | The agent sees varying load during training; the busy day stays outside the training range as a generalisation test |
| Time in seconds, one run lasts 480 s | Timing follows the SO-ARM101 instead of the illustrative minutes in the proposal |
| Processing time derived from weight and fragility; products are (weight, fragility) | On the arm, a pick-and-place has a fixed path; items differ in how slowly they must be moved |
| Weights of 20–230 g | Within the SO-ARM101 payload |
| Two-station layout with per-station output spots | A single shared output spot would make tray switches free |
| "Dropped once" flag added to the observation | A retried job risks being scrapped |
| Pooled queue context (Deep Sets) added to the per-job network | Scoring each job from its own features alone cannot see the rest of the queue, and the PPO critic and dueling value stream need a whole-queue input |
| Unseen products are tested mixed into the familiar mix (P1–P4 plus P5 or P6) | A shift of a single product leaves little to sequence; the mix shows what a newly introduced product does to the whole schedule |

## Baseline results

Rule-based policies on 100 test shifts of familiar products (P1–P4), mean ± std across shifts.
Tardiness and switching are in seconds.

**Normal day** (0.14 jobs/s, 70 jobs per shift)

| Policy | On time | Tardiness | Switching | Drops | Scrapped | Unfinished | Score |
|---|---|---|---|---|---|---|---|
| FCFS | 59.86 ± 6.48 | 73.1 ± 111.7 | 39.5 ± 6.2 | 9.05 ± 3.00 | 0.98 ± 0.91 | 0.89 ± 1.32 | 46.51 ± 8.23 |
| SPT | 61.57 ± 5.18 | 138.0 ± 182.3 | 40.4 ± 6.9 | 9.03 ± 3.01 | 0.97 ± 0.91 | 0.90 ± 1.33 | 47.12 ± 6.16 |
| LRF | 61.34 ± 5.17 | 147.6 ± 191.0 | 40.8 ± 6.7 | 9.04 ± 3.02 | 0.97 ± 0.91 | 0.92 ± 1.34 | 46.67 ± 6.23 |
| EDF | 64.05 ± 7.84 | 33.5 ± 96.1 | 40.8 ± 6.4 | 9.05 ± 3.00 | 0.98 ± 0.91 | 0.96 ± 1.45 | 51.03 ± 9.45 |
| NTF | 62.47 ± 5.95 | 48.8 ± 70.3 | 28.8 ± 4.7 | 9.08 ± 3.02 | 0.98 ± 0.91 | 0.73 ± 1.21 | 51.24 ± 6.27 |
| Random | 59.95 ± 5.78 | 136.0 ± 180.0 | 41.8 ± 7.1 | 9.07 ± 2.96 | 0.97 ± 0.91 | 0.98 ± 1.50 | 45.22 ± 8.13 |

**Busy day** (0.28 jobs/s, 137 jobs per shift)

| Policy | On time | Tardiness | Switching | Drops | Scrapped | Unfinished | Score |
|---|---|---|---|---|---|---|---|
| FCFS | 13.06 ± 5.88 | 8401 ± 2559 | 53.1 ± 5.2 | 11.85 ± 3.96 | 1.52 ± 1.29 | 45.91 ± 10.23 | −173.94 ± 51.13 |
| SPT | 23.46 ± 8.97 | 8319 ± 2475 | 52.8 ± 5.2 | 11.41 ± 3.93 | 1.38 ± 1.25 | 45.12 ± 10.34 | −161.21 ± 52.47 |
| LRF | 23.66 ± 8.98 | 8365 ± 2463 | 53.1 ± 5.3 | 11.40 ± 3.91 | 1.39 ± 1.25 | 45.11 ± 10.31 | −161.72 ± 52.34 |
| EDF | 12.77 ± 6.22 | 8342 ± 2595 | 53.0 ± 4.8 | 11.95 ± 3.83 | 1.55 ± 1.29 | 45.84 ± 10.40 | −173.33 ± 51.91 |
| NTF | 21.25 ± 9.02 | 6372 ± 2419 | 8.9 ± 1.7 | 13.09 ± 4.14 | 1.65 ± 1.34 | 36.53 ± 10.35 | −120.04 ± 51.94 |
| Random | 16.45 ± 6.26 | 8740 ± 2532 | 56.9 ± 5.3 | 11.74 ± 3.93 | 1.42 ± 1.20 | 46.74 ± 10.30 | −176.97 ± 51.30 |

EDF and NTF tie on a normal day, while NTF clearly leads on a busy day: the best rule depends
on the load. On a normal day the rules span only about 6 points; the CP-SAT clairvoyant bound
will show how much room remains for a learned policy.

## Repository layout

```
src/lvhm/
  config.py      EnvConfig, products, penalty ranges, processing-time and drop models
  instance.py    pre-drawn shift instances, hold-out region, evaluation shift sets
  env.py         LVHMSchedulingEnv
  baselines.py   FCFS, SPT, LRF, EDF, NTF, Random
  evaluate.py    run a policy on fixed shifts, summarise metrics
  experiments.py the four Part B test conditions, evaluation of many policies
  so101.py       SO101 IK, MuJoCo move timing, payload torque check (needs the sim group)
  agents/
    networks.py  shared-weight job encoder, pooled context, job scorer, value head
    ppo.py       MaskablePPO policy, train_ppo, load_ppo
    dqn.py       masked double dueling DQN, run_dqn, load_dqn
    common.py    validation shifts, best-checkpoint selection, learning-curve CSV
notebooks/
  experiments.ipynb   self-contained Part C experiments (upload alone to SageMaker)
scripts/
  baseline_check.py   rule results table and pre-training gap check
  train_ppo.py        train and test masked PPO from the command line
  train_dqn.py        train and test masked double dueling DQN from the command line
  calibrate_so101.py  writes assets/so101/cell_timing.json
assets/so101/         vendored SO101 MJCF and meshes, cell_timing.json
tests/                environment, agent and calibration tests
```

## Known limitations

- Only the eight oldest waiting jobs are visible; on a busy day newer jobs wait unseen.
- There is no explicit wait action: the arm always serves a job when one is waiting.
- The gripper is a hinge in the MJCF, while LeRobot drives it as a 0–100 linear value.
