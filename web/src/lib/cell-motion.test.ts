/** The arm motion derived from a trace must stay inside the calibrated waypoints and never jump. */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { poseFor, spot, type Waypoints } from '@/lib/cell-motion';
import { Replay } from '@/lib/replay';
import type { Dataset } from '@/types/trace.types';

const ds = JSON.parse(readFileSync(new URL('../../public/data/job_sequences.json', import.meta.url), 'utf8')) as Dataset;
const timing = JSON.parse(readFileSync(new URL('../../../assets/so101/cell_timing.json', import.meta.url), 'utf8')) as {
  layout: { radius: number; station_azimuth: number; spot_offset: number; gripper_open: number; gripper_closed: number };
  waypoints: Waypoints;
};
const grip = { open: timing.layout.gripper_open, closed: timing.layout.gripper_closed, hold: 0.25 };

describe('cell motion', () => {
  it('stays within the hull of the calibrated waypoints and moves continuously', () => {
    const lo = Array(6).fill(Infinity), hi = Array(6).fill(-Infinity);
    for (const q of Object.values(timing.waypoints)) q.forEach((v, i) => { lo[i] = Math.min(lo[i], v); hi[i] = Math.max(hi[i], v); });
    lo[5] = Math.min(lo[5], grip.closed); hi[5] = Math.max(hi[5], grip.open);
    const dt = 0.05;
    let samples = 0, maxJump = 0, outOfHull = 0;
    for (const cond of ds.conditions.slice(0, 1)) {
      for (const shift of cond.shifts.slice(0, 2)) {
        for (const policy of ['DQN', 'EDF']) {
          const replay = new Replay(ds, shift, shift.runs[policy]);
          let prev: number[] | null = null;
          for (let t = 0; t <= replay.horizon; t += dt) {
            const { q } = poseFor(replay.stateAt(t), timing.waypoints, grip);
            q.forEach((v, i) => { if (v < lo[i] - 1e-9 || v > hi[i] + 1e-9) outOfHull++; });
            if (prev) maxJump = Math.max(maxJump, ...q.map((v, i) => Math.abs(v - prev![i])));
            prev = q;
            samples++;
          }
        }
      }
    }
    expect(samples).toBeGreaterThan(10000);
    expect(outOfHull).toBe(0);
    expect(maxJump).toBeLessThan(0.6);
  });

  it('carries the part between lift and release, and drops it on a dropped pick', () => {
    const shift = ds.conditions[0].shifts[0];
    const replay = new Replay(ds, shift, shift.runs['DQN']);
    const ok = replay.steps.find((s) => s.outcome === 0)!;
    const mid = ok.start + ok.travel + (ok.end - ok.start - ok.travel) * 0.6;
    expect(poseFor(replay.stateAt(mid), timing.waypoints, grip).carrying).toBe(true);
    expect(poseFor(replay.stateAt(ok.end - 1e-3), timing.waypoints, grip).placed).toBe(true);
    const drop = replay.steps.find((s) => s.outcome === 2 || s.outcome === 3);
    if (drop) {
      const late = drop.start + drop.travel + (drop.end - drop.start - drop.travel) * 0.9;
      const pose = poseFor(replay.stateAt(late), timing.waypoints, grip);
      expect(pose.dropped).toBe(true);
      expect(pose.carrying).toBe(false);
    }
  });

  it('places stations where lvhm/so101.py puts them', () => {
    const [x, y] = spot(timing.layout, 0, 'tray', 0);
    expect(Math.hypot(x, y)).toBeCloseTo(timing.layout.radius, 6);
    expect((Math.atan2(y, x) * 180) / Math.PI).toBeCloseTo(timing.layout.station_azimuth + timing.layout.spot_offset, 6);
    const [x1, y1] = spot(timing.layout, 1, 'out', 0);
    expect((Math.atan2(y1, x1) * 180) / Math.PI).toBeCloseTo(-(timing.layout.station_azimuth - timing.layout.spot_offset), 6);
  });
});

describe('part leaves the tray exactly once per pick', () => {
  it('is lifted from the grip until the end of the step, and not before', () => {
    const shift = ds.conditions[0].shifts[0];
    const replay = new Replay(ds, shift, shift.runs['DQN']);
    const steps = [replay.steps.find((s) => s.outcome === 0)!, replay.steps.find((s) => s.outcome === 2 || s.outcome === 3)].filter(Boolean);
    for (const step of steps) {
      const t0 = step!.start + step!.travel, span = step!.end - t0;
      let lifted = false;
      for (let p = 0; p < 1; p += 0.01) {
        const pose = poseFor(replay.stateAt(t0 + span * p), timing.waypoints, grip);
        if (pose.lifted) lifted = true;
        else expect(lifted).toBe(false); // once lifted, the part never reappears in the tray during the step
      }
      expect(lifted).toBe(true);
      expect(poseFor(replay.stateAt(step!.end - 1e-3), timing.waypoints, grip).lifted).toBe(true);
      expect(poseFor(replay.stateAt(t0 + span * 0.1), timing.waypoints, grip).lifted).toBe(false);
    }
  });
});
