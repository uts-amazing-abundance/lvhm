/**
 * Top-down inverse kinematics for the SO-101 in the browser: the damped least squares of
 * lvhm/so101.py on MuJoCo's site Jacobian. Positions the gripperframe site at a target while
 * keeping its approach axis pointing straight down. Only the four arm joints move; wrist roll
 * and the gripper keep the values of the seed.
 */
import type { MainModule, MjData, MjModel } from '@mujoco/mujoco';

const ARM_JOINTS = 4;
const DOWN = [0, 0, -1];
const ORI_WEIGHT = 0.3;
const DAMPING = 1e-4;
const MAX_ITERS = 300;
const POS_TOL = 1e-4;
const ORI_TOL = 3e-4;

interface DoubleBufferLike { GetView(): Float64Array; delete(): void }
interface Arrays { nv: number; jnt_range: Float64Array; site_xpos: Float64Array; site_xmat: Float64Array; qpos: Float64Array }

export interface IkSolver {
  /** Joint vector reaching `target` top-down from `seed`, or null when it is out of reach. */
  solve(target: readonly number[], seed: readonly number[]): number[] | null;
  dispose(): void;
}

export function createIkSolver(mujoco: MainModule, model: MjModel, data: MjData, siteId: number): IkSolver {
  const m = model as unknown as Arrays;
  const d = data as unknown as Arrays;
  const nv = m.nv;
  const Buffer = (mujoco as unknown as { DoubleBuffer: new (n: number) => DoubleBufferLike }).DoubleBuffer;
  const jacp = new Buffer(3 * nv);
  const jacr = new Buffer(3 * nv);
  const lo = Array.from({ length: ARM_JOINTS }, (_, j) => m.jnt_range[2 * j]);
  const hi = Array.from({ length: ARM_JOINTS }, (_, j) => m.jnt_range[2 * j + 1]);

  const pose = (q: number[]) => {
    for (let i = 0; i < q.length; i++) d.qpos[i] = q[i];
    mujoco.mj_forward(model, data);
    const p = siteId * 3, r = siteId * 9;
    const pos = [d.site_xpos[p], d.site_xpos[p + 1], d.site_xpos[p + 2]];
    const approach = [d.site_xmat[r], d.site_xmat[r + 3], d.site_xmat[r + 6]]; // first column: finger direction
    return { pos, approach };
  };

  const solve = (target: readonly number[], seed: readonly number[]): number[] | null => {
    const q = seed.slice();
    for (let iter = 0; iter < MAX_ITERS; iter++) {
      const { pos, approach } = pose(q);
      const cr = cross(approach, DOWN);
      const err = [target[0] - pos[0], target[1] - pos[1], target[2] - pos[2], ORI_WEIGHT * cr[0], ORI_WEIGHT * cr[1], ORI_WEIGHT * cr[2]];
      if (norm(err.slice(0, 3)) < POS_TOL && norm(err.slice(3)) < ORI_TOL) break;
      mujoco.mj_jacSite(model, data, jacp, jacr, siteId);
      const Jp = jacp.GetView(), Jr = jacr.GetView();
      // J: 6 × ARM_JOINTS, rows 0-2 position, rows 3-5 orientation (weighted)
      const J: number[][] = [];
      for (let r = 0; r < 3; r++) J.push(Array.from({ length: ARM_JOINTS }, (_, c) => Jp[r * nv + c]));
      for (let r = 0; r < 3; r++) J.push(Array.from({ length: ARM_JOINTS }, (_, c) => ORI_WEIGHT * Jr[r * nv + c]));
      // dq = J^T (J J^T + λI)^-1 err
      const A = J.map((row, i) => J.map((row2, k) => dot(row, row2) + (i === k ? DAMPING : 0)));
      const y = solveLinear(A, err);
      for (let c = 0; c < ARM_JOINTS; c++) {
        let dq = 0;
        for (let r = 0; r < 6; r++) dq += J[r][c] * y[r];
        q[c] = Math.min(hi[c], Math.max(lo[c], q[c] + dq));
      }
    }
    const { pos, approach } = pose(q);
    const posErr = norm([target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]]);
    const cosTilt = dot(approach, DOWN);
    return posErr <= 1e-3 && cosTilt >= Math.cos((2 * Math.PI) / 180) ? q : null;
  };

  return { solve, dispose: () => { jacp.delete(); jacr.delete(); } };
}

const dot = (a: readonly number[], b: readonly number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a: readonly number[]) => Math.sqrt(dot(a, a));
const cross = (a: readonly number[], b: readonly number[]) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Gaussian elimination with partial pivoting for a small dense system. */
function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const pivot = M[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / pivot;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / (M[r][r] || 1e-12);
  }
  return x;
}
