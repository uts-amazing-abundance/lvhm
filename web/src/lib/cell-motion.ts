/**
 * Maps the replay state onto SO-101 joint targets, following the pick cycle that
 * scripts/calibrate_so101.py timed in MuJoCo: swing to the tray, descend, close, lift, swing to
 * the station's output spot, descend, open, rise. Waypoints are the IK solutions stored in
 * assets/so101/cell_timing.json (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
 * wrist_roll, gripper).
 */
import { Outcome, type ReplayState } from '@/lib/replay';

export type Waypoints = Record<string, number[]>;

export interface GripperRange {
  open: number;
  /** Fully closed, fingers touching. */
  closed: number;
  /** Closed on a part: the fingers stop at the part width. */
  hold: number;
}

export interface ArmPose {
  q: number[];
  /** A part is in the gripper. */
  carrying: boolean;
  /** The part has been dropped during this pick. */
  dropped: boolean;
  /** The part has been released on the output spot during this pick. */
  placed: boolean;
  /** The part is no longer in its tray pocket: in the gripper, placed, or dropped. */
  lifted: boolean;
  /** 0 while the fingers close on the part in its pocket, 1 once it is seated in the gripper. */
  settle: number;
}

export const GRIPPER_JOINT = 5;

const lerp = (a: number[], b: number[], s: number): number[] => a.map((v, i) => v + (b[i] - v) * s);
const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));
const smooth = (x: number): number => { const s = clamp01(x); return s * s * (3 - 2 * s); };

function withGripper(q: number[], g: number): number[] {
  const out = q.slice();
  out[GRIPPER_JOINT] = g;
  return out;
}

/** Segment boundaries of a successful pick, as fractions of the processing part of the step. */
const SEG = { approach: 0.12, descend: 0.28, close: 0.36, lift: 0.48, carry: 0.7, lower: 0.8, open: 0.88 } as const;
/** A dropped pick aborts the carry part way through and returns to the station rest pose. */
const DROP_AT = 0.62;

export function poseFor(state: ReplayState, wp: Waypoints, grip: GripperRange): ArmPose {
  // Where the arm waits at a station: the hover above the last part it placed, else the calibrated spot.
  const rest = (tray: number) => withGripper(wp[`out${tray}_rest`] ?? wp[`out${tray}_hover`], grip.open);
  const none = (q: number[]): ArmPose => ({ q, carrying: false, dropped: false, placed: false, lifted: false, settle: 0 });

  if (state.phase === 'travel') {
    return none(lerp(rest(state.armFrom), withGripper(wp[`tray${state.armTo}_hover`], grip.open), smooth(state.armProgress)));
  }
  if (state.phase !== 'pick' || !state.current) return none(rest(state.armTo));

  const s = state.armTo;
  const p = state.pickProgress;
  const trayHover = withGripper(wp[`tray${s}_hover`], grip.open);
  const trayLow = withGripper(wp[`tray${s}_low`], grip.open);
  const trayLowClosed = withGripper(trayLow, grip.hold);
  const trayHoverClosed = withGripper(trayHover, grip.hold);
  const outHoverClosed = withGripper(wp[`out${s}_hover`], grip.hold);
  const outLowClosed = withGripper(wp[`out${s}_low`], grip.hold);
  const outLow = withGripper(outLowClosed, grip.open);
  const outHover = rest(s);
  const start = state.current.travel > 0 ? trayHover : outHover;
  const isDrop = state.current.outcome === Outcome.Drop || state.current.outcome === Outcome.Scrap;
  const seg = (a: number, b: number) => smooth((p - a) / (b - a));

  if (p < SEG.approach) return none(lerp(start, trayHover, seg(0, SEG.approach)));
  if (p < SEG.descend) return none(lerp(trayHover, trayLow, seg(SEG.approach, SEG.descend)));
  if (p < SEG.close) return none(lerp(trayLow, trayLowClosed, seg(SEG.descend, SEG.close)));
  if (p < SEG.lift) return { q: lerp(trayLowClosed, trayHoverClosed, seg(SEG.close, SEG.lift)), carrying: true, dropped: false, placed: false, lifted: true, settle: seg(SEG.close, SEG.lift) };
  if (isDrop) {
    // Lose the part part way across, finish the swing empty and come to rest at the station.
    const q = lerp(trayHoverClosed, withGripper(rest(s), grip.hold), seg(SEG.lift, 1));
    if (p < DROP_AT) return { q, carrying: true, dropped: false, placed: false, lifted: true, settle: 1 };
    const released = withGripper(q, grip.hold + (grip.open - grip.hold) * seg(DROP_AT, DROP_AT + 0.1));
    return { q: released, carrying: false, dropped: true, placed: false, lifted: true, settle: 1 };
  }
  if (p < SEG.carry) return { q: lerp(trayHoverClosed, outHoverClosed, seg(SEG.lift, SEG.carry)), carrying: true, dropped: false, placed: false, lifted: true, settle: 1 };
  if (p < SEG.lower) return { q: lerp(outHoverClosed, outLowClosed, seg(SEG.carry, SEG.lower)), carrying: true, dropped: false, placed: false, lifted: true, settle: 1 };
  if (p < SEG.open) return { q: lerp(outLowClosed, outLow, seg(SEG.lower, SEG.open)), carrying: false, dropped: false, placed: true, lifted: true, settle: 1 };
  return { q: lerp(outLow, outHover, seg(SEG.open, 1)), carrying: false, dropped: false, placed: true, lifted: true, settle: 1 };
}

/** World position of a station's tray or output spot, as CellLayout.spot() in lvhm/so101.py. */
export function spot(layout: { radius: number; station_azimuth: number; spot_offset: number }, station: number, kind: 'tray' | 'out', z: number): [number, number, number] {
  const sign = station === 0 ? 1 : -1;
  const offset = kind === 'tray' ? layout.spot_offset : -layout.spot_offset;
  const az = (sign * (layout.station_azimuth + offset) * Math.PI) / 180;
  return [layout.radius * Math.cos(az), layout.radius * Math.sin(az), z];
}
