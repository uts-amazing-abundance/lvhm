/**
 * The replay must reproduce what the environment recorded. Every traced step carries the
 * queue length the policy saw, and the rule-based runs were produced by the very rules this
 * app recomputes, so both are checked against the real export.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { K_SLOTS, Outcome, Replay } from '@/lib/replay';
import { isRule, rulePick } from '@/lib/dispatch-rules';
import type { Dataset } from '@/types/trace.types';

const ds = JSON.parse(readFileSync(new URL('../../public/data/job_sequences.json', import.meta.url), 'utf8')) as Dataset;

function* allRuns() {
  for (const cond of ds.conditions)
    for (const shift of cond.shifts)
      for (const policy of ds.policies) yield { cond, shift, policy, replay: new Replay(ds, shift, shift.runs[policy]) };
}

describe('queue reconstruction', () => {
  it('matches the queue length recorded at every decision', () => {
    let checked = 0;
    for (const { replay } of allRuns()) {
      for (const step of replay.steps) {
        const q = replay.decisionQueue(step);
        expect(q.length).toBe(step.queueLen);
        const slot = q.findIndex((e) => e.job.id === step.job);
        expect(slot).toBeGreaterThanOrEqual(0);
        expect(slot).toBeLessThan(K_SLOTS);
        checked++;
      }
    }
    expect(checked).toBeGreaterThan(1000);
  });

  it('travel happens exactly when the job is on the other tray', () => {
    for (const { replay } of allRuns()) {
      for (const step of replay.steps) {
        const needsTravel = replay.jobs[step.job].tray !== step.prevTray;
        expect(step.travel > 0).toBe(needsTravel);
      }
    }
  });

  it('reaches the recorded episode metrics at the end of the shift', () => {
    for (const { shift, policy, replay } of allRuns()) {
      const m = shift.runs[policy].metrics;
      const s = replay.statsAt(replay.horizon);
      expect(s.decisions).toBe(m.decisions);
      expect(s.completed).toBe(m.completed);
      expect(s.onTime).toBe(m.on_time);
      expect(s.late).toBe(m.late);
      expect(s.drops).toBe(m.drops);
      expect(s.scrapped).toBe(m.scrapped);
      expect(s.unfinished).toBe(m.unfinished);
      expect(s.careful).toBe(m.careful);
      expect(s.travel).toBeCloseTo(m.travel, 2);
      expect(s.score).toBeCloseTo(m.score, 2);
      expect(s.completedByTray[0] + s.completedByTray[1]).toBe(m.completed);
    }
  });

  it('keeps a dropped job in the queue with a retry flag and removes a scrapped one', () => {
    for (const { replay } of allRuns()) {
      for (const step of replay.steps) {
        const after = replay.queueAt(step.end);
        const entry = after.find((e) => e.job.id === step.job);
        if (step.outcome === Outcome.Drop) {
          expect(entry).toBeDefined();
          expect(entry!.attempts).toBeGreaterThan(0);
        } else {
          expect(entry).toBeUndefined();
        }
      }
    }
  });
});

describe('dispatch rules', () => {
  it('reproduce every pick of their own traced run', () => {
    const counts: Record<string, number> = {};
    for (const { shift, policy, replay } of allRuns()) {
      if (!isRule(policy) || policy === 'Random') continue;
      for (const step of replay.steps) {
        const visible = replay.decisionQueue(step).slice(0, K_SLOTS);
        const i = rulePick(policy, visible, step.prevTray);
        expect({ policy, shift: shift.shift, step: step.index, job: visible[i].job.id }).toEqual({ policy, shift: shift.shift, step: step.index, job: step.job });
        counts[policy] = (counts[policy] ?? 0) + 1;
      }
    }
    for (const r of ['FCFS', 'SPT', 'LRF', 'EDF', 'NTF']) expect(counts[r]).toBeGreaterThan(100);
  });

  it('never picks from an empty queue and always picks a visible slot', () => {
    expect(rulePick('EDF', [], 0)).toBe(-1);
    const { replay } = allRuns().next().value!;
    const visible = replay.decisionQueue(replay.steps[0]).slice(0, K_SLOTS);
    for (const r of ['FCFS', 'SPT', 'LRF', 'EDF', 'NTF', 'Random'] as const) {
      const i = rulePick(r, visible, 0, 7);
      expect(i).toBeGreaterThanOrEqual(0);
      expect(i).toBeLessThan(visible.length);
    }
  });
});

describe('tray pockets', () => {
  it('keep each job in one pocket for its whole stay and never share a pocket', () => {
    const shift = ds.conditions[3].shifts[0]; // busy day: trays overflow
    for (const policy of ['DQN', 'FCFS']) {
      const replay = new Replay(ds, shift, shift.runs[policy]);
      const seen = new Map<number, number>();
      for (let t = 0; t <= replay.horizon; t += 0.5) {
        const used = new Set<string>();
        for (const e of replay.queueAt(t)) {
          if (e.pocket == null) continue;
          expect(e.pocket).toBeLessThan(K_SLOTS);
          const key = `${e.job.tray}:${e.pocket}`;
          expect(used.has(key)).toBe(false);
          used.add(key);
          const prev = seen.get(e.job.id);
          if (prev != null) expect(e.pocket).toBe(prev);
          seen.set(e.job.id, e.pocket);
        }
      }
      expect(seen.size).toBeGreaterThan(20);
    }
  });
});
