/**
 * The six dispatch rules of lvhm/baselines.py, applied to a reconstructed decision state so the
 * dashboard can show what each rule would have picked from the queue the active policy saw.
 * Ties go to the earliest arrival (lowest slot), as np.argmin does over arrival-ordered slots.
 */
import type { QueueEntry } from '@/lib/replay';

export const RULES = ['FCFS', 'SPT', 'LRF', 'EDF', 'NTF', 'Random'] as const;
export type Rule = (typeof RULES)[number];

export const RULE_LONG: Record<Rule, string> = {
  FCFS: 'First come, first served',
  SPT: 'Shortest processing time',
  LRF: 'Lowest failure risk first',
  EDF: 'Earliest deadline first',
  NTF: 'Nearest tray first',
  Random: 'Uniformly random valid slot',
};

export function isRule(name: string): name is Rule {
  return (RULES as readonly string[]).includes(name);
}

/** Small deterministic hash so the Random chip is stable for a given decision. */
function hash32(n: number): number {
  let x = (n + 0x9e3779b9) >>> 0;
  x ^= x >>> 16; x = Math.imul(x, 0x85ebca6b) >>> 0;
  x ^= x >>> 13; x = Math.imul(x, 0xc2b2ae35) >>> 0;
  x ^= x >>> 16;
  return x >>> 0;
}

/**
 * Index into `visible` (the first K queue entries) that the rule would serve, or -1 when the
 * queue is empty. `armTray` is where the arm rests; `seed` only matters for Random.
 */
export function rulePick(rule: Rule, visible: QueueEntry[], armTray: number, seed = 0): number {
  const n = visible.length;
  if (n === 0) return -1;
  const argmin = (key: (e: QueueEntry) => number): number => {
    let best = 0;
    for (let i = 1; i < n; i++) if (key(visible[i]) < key(visible[best])) best = i;
    return best;
  };
  switch (rule) {
    case 'FCFS': return argmin((e) => e.job.arrival);
    case 'SPT': return argmin((e) => e.job.proc);
    case 'LRF': return argmin((e) => e.job.fragility);
    case 'EDF': return argmin((e) => e.job.deadline);
    case 'NTF': return argmin((e) => (e.job.tray === armTray ? 0 : 1));
    case 'Random': return hash32(seed) % n;
  }
}
