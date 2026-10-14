/**
 * Replays one traced run of one shift. The trace records every pick; this module turns
 * it back into the cell state at any time t: who is waiting, where the arm is, what the
 * policy is doing, and the running totals.
 *
 * The reconstruction mirrors lvhm/env.py: jobs are admitted when arrival <= t, a job leaves
 * the queue at the end of the step that completes or scraps it, a dropped job stays and gets a
 * retry flag, and the arm rests at the tray of the job it served last (tray 0 at the start).
 */
import type { Dataset, Run, ShiftData } from '@/types/trace.types';

export const K_SLOTS = 8;
export const START_TRAY = 0;

export const enum Outcome {
  OnTime = 0,
  Late = 1,
  Drop = 2,
  Scrap = 3,
}
export const OUTCOME_LABEL = ['On time', 'Late', 'Dropped, retried', 'Dropped, scrapped'] as const;

export interface Job {
  id: number;
  arrival: number;
  deadline: number;
  tray: number;
  product: string;
  proc: number;
  weight: number;
  fragility: number;
}

export interface Step {
  index: number;
  job: number;
  start: number;
  end: number;
  travel: number;
  careful: boolean;
  outcome: Outcome;
  queueLen: number;
  reward: number;
  /** Tray the arm was at when this decision was made. */
  prevTray: number;
}

export interface QueueEntry {
  job: Job;
  /** Drops this job has already suffered. */
  attempts: number;
  /** Seconds since arrival. */
  waiting: number;
  /** Observation slot (0..K-1) or null when the job is beyond the visible window. */
  slot: number | null;
  /** Deadline already passed. */
  late: boolean;
}

export interface Stats {
  decisions: number;
  completed: number;
  onTime: number;
  late: number;
  drops: number;
  scrapped: number;
  careful: number;
  travel: number;
  score: number;
  unfinished: number;
}

export type Phase = 'waiting' | 'travel' | 'pick' | 'finished';

export interface ReplayState {
  t: number;
  phase: Phase;
  /** Step in progress (start <= t < end). */
  current: Step | null;
  /** Most recent step with start <= t. */
  lastDecision: Step | null;
  armFrom: number;
  armTo: number;
  /** 0..1 across the travel swing; 1 once the arm is at armTo. */
  armProgress: number;
  /** 0..1 across the processing part of the current pick. */
  pickProgress: number;
  queue: QueueEntry[];
  inHand: Job | null;
  stats: Stats;
  finished: boolean;
}

export function decodeJobs(ds: Dataset, shift: ShiftData): Job[] {
  return shift.jobs.map(([arrival, deadline, tray, product, proc], id) => {
    const feats = ds.products[product];
    const [weight, fragility] = feats ?? [Number.NaN, Number.NaN];
    return { id, arrival, deadline, tray, product, proc, weight, fragility };
  });
}

export function decodeSteps(run: Run, jobs: Job[]): Step[] {
  let prevTray = START_TRAY;
  return run.steps.map((s, index) => {
    const step: Step = {
      index,
      job: s[0],
      start: s[1],
      end: s[2],
      travel: s[3],
      careful: s[4] === 1,
      outcome: s[5] as Outcome,
      queueLen: s[6],
      reward: s[7],
      prevTray,
    };
    prevTray = jobs[s[0]].tray;
    return step;
  });
}

export class Replay {
  readonly jobs: Job[];
  readonly steps: Step[];
  readonly shiftLength: number;
  /** Time each job left the queue (completed or scrapped); Infinity if it never did. */
  private readonly removedAt: Float64Array;
  /** Times at which each job was dropped, ascending. */
  private readonly dropTimes: number[][];

  constructor(ds: Dataset, shift: ShiftData, run: Run) {
    this.jobs = decodeJobs(ds, shift);
    this.steps = decodeSteps(run, this.jobs);
    this.shiftLength = ds.shift_length;
    this.removedAt = new Float64Array(this.jobs.length).fill(Number.POSITIVE_INFINITY);
    this.dropTimes = this.jobs.map(() => []);
    for (const s of this.steps) {
      if (s.outcome === Outcome.Drop) this.dropTimes[s.job].push(s.end);
      else this.removedAt[s.job] = s.end;
    }
  }

  /** Time the episode ended: the last pick's end, or the shift length when nothing was picked. */
  get episodeEnd(): number {
    return this.steps.length ? this.steps[this.steps.length - 1].end : this.shiftLength;
  }

  /** End of the replay time axis: the last pick may run past the shift clock. */
  get horizon(): number {
    return Math.max(this.shiftLength, this.episodeEnd);
  }

  /** Jobs waiting at time t in arrival order, with the observation slot each occupies. */
  queueAt(t: number): QueueEntry[] {
    const out: QueueEntry[] = [];
    for (const job of this.jobs) {
      if (job.arrival > t) break; // jobs are stored in arrival order
      if (this.removedAt[job.id] <= t) continue;
      let attempts = 0;
      for (const d of this.dropTimes[job.id]) if (d <= t) attempts++;
      const slot = out.length < K_SLOTS ? out.length : null;
      out.push({ job, attempts, waiting: t - job.arrival, slot, late: t > job.deadline });
    }
    return out;
  }

  /**
   * The queue the policy saw when it made this decision (the served job is still in it).
   * The export rounds times to 1 ms, so a job that arrived a fraction of a millisecond after
   * the decision can look admitted; the recorded queue length settles those ties.
   */
  decisionQueue(step: Step): QueueEntry[] {
    const q = this.queueAt(step.start);
    while (q.length > step.queueLen && q[q.length - 1].job.arrival >= step.start - 1e-3) q.pop();
    return q;
  }

  stepAt(t: number): { current: Step | null; lastDecision: Step | null } {
    let current: Step | null = null;
    let lastDecision: Step | null = null;
    for (const s of this.steps) {
      if (s.start > t) break;
      lastDecision = s;
      if (t < s.end) current = s;
    }
    return { current, lastDecision };
  }

  statsAt(t: number): Stats {
    const st: Stats = { decisions: 0, completed: 0, onTime: 0, late: 0, drops: 0, scrapped: 0, careful: 0, travel: 0, score: 0, unfinished: 0 };
    for (const s of this.steps) {
      if (s.end > t) break;
      st.decisions++;
      st.score += s.reward;
      st.travel += s.travel;
      if (s.careful) st.careful++;
      switch (s.outcome) {
        case Outcome.OnTime: st.completed++; st.onTime++; break;
        case Outcome.Late: st.completed++; st.late++; break;
        case Outcome.Drop: st.drops++; break;
        case Outcome.Scrap: st.drops++; st.scrapped++; break;
      }
    }
    if (t >= this.episodeEnd) st.unfinished = this.queueAt(t).length;
    return st;
  }

  stateAt(tRaw: number): ReplayState {
    const t = Math.min(Math.max(0, tRaw), this.horizon);
    const { current, lastDecision } = this.stepAt(t);
    const queue = this.queueAt(t);
    const finished = t >= this.episodeEnd;

    let phase: Phase = finished ? 'finished' : 'waiting';
    let armFrom = START_TRAY;
    let armTo = START_TRAY;
    let armProgress = 1;
    let pickProgress = 0;
    let inHand: Job | null = null;

    if (current) {
      const job = this.jobs[current.job];
      armFrom = current.prevTray;
      armTo = job.tray;
      const travelEnd = current.start + current.travel;
      if (current.travel > 0 && t < travelEnd) {
        phase = 'travel';
        armProgress = (t - current.start) / current.travel;
      } else {
        phase = 'pick';
        armProgress = 1;
        const span = current.end - travelEnd;
        pickProgress = span > 0 ? Math.min(1, (t - travelEnd) / span) : 1;
        inHand = job;
      }
    } else if (lastDecision) {
      armFrom = armTo = this.jobs[lastDecision.job].tray;
    }

    return { t, phase, current, lastDecision, armFrom, armTo, armProgress, pickProgress, queue, inHand, stats: this.statsAt(t), finished };
  }

  /** Cumulative reward after each pick, as [time, score] points starting at the origin. */
  cumulative(): [number, number][] {
    const pts: [number, number][] = [[0, 0]];
    let c = 0;
    for (const s of this.steps) {
      c += s.reward;
      pts.push([s.end, c]);
    }
    return pts;
  }
}
