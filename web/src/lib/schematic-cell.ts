/**
 * 2D schematic of the cell: two input trays with the waiting jobs, an output area, and a
 * two-link arm that swings between trays and dips to pick. It implements CellRenderer so a
 * MuJoCo WASM renderer can take its place later with the same update() contract.
 */
import { cssVar, svg } from '@/lib/svg-dom';
import { Outcome, type Job, type ReplayState } from '@/lib/replay';

export interface CellRenderer {
  update(state: ReplayState, chosenJob: number | null): void;
}

const TRAY_X = [120, 360];
const TRAY_Y = 262;
const TRAY_W = 150;
const TRAY_H = 52;
/** One output spot per station, inboard of its tray, as in lvhm/so101.py. */
const OUT = [{ x: 185, y: 118 }, { x: 295, y: 118 }];
const SHOULDER = { x: 240, y: 212 };
const L1 = 92;
const L2 = 86;
const HOVER_Y = 168;
const DOWN_Y = 236;

export class SchematicCell implements CellRenderer {
  private readonly layer: { static: SVGGElement; jobs: SVGGElement; arm: SVGGElement; held: SVGGElement; text: SVGGElement };

  constructor(root: SVGSVGElement) {
    root.innerHTML = '';
    this.layer = {
      static: svg('g', {}, root),
      jobs: svg('g', {}, root),
      arm: svg('g', {}, root),
      held: svg('g', {}, root),
      text: svg('g', {}, root),
    };
    this.drawStatic();
  }

  private drawStatic(): void {
    const g = this.layer.static;
    // floor plate
    svg('polygon', { points: '30,300 450,300 420,226 60,226', fill: cssVar('--wash'), stroke: cssVar('--grid') }, g);
    TRAY_X.forEach((x, i) => {
      svg('rect', { x: x - TRAY_W / 2, y: TRAY_Y - TRAY_H / 2, width: TRAY_W, height: TRAY_H, rx: 6, fill: cssVar('--tray'), stroke: cssVar('--tray-edge') }, g);
      svg('text', { x, y: TRAY_Y + TRAY_H / 2 + 14, 'text-anchor': 'middle' }, g).textContent = `tray ${i}`;
    });
    OUT.forEach((o, i) => {
      svg('rect', { x: o.x - 34, y: o.y - 16, width: 68, height: 32, rx: 6, fill: 'none', stroke: cssVar('--tray-edge'), 'stroke-dasharray': '4 3' }, g);
      svg('text', { x: o.x, y: o.y - 22, 'text-anchor': 'middle' }, g).textContent = `output ${i}`;
    });
    // base
    svg('rect', { x: SHOULDER.x - 26, y: SHOULDER.y + 2, width: 52, height: 22, rx: 4, fill: cssVar('--ink-2') }, g);
    svg('circle', { cx: SHOULDER.x, cy: SHOULDER.y, r: 9, fill: cssVar('--ink') }, g);
  }

  /** x of the pocket holding the job being picked, so the arm dips where the part is. */
  private targetX: number | null = null;

  update(state: ReplayState, chosenJob: number | null): void {
    const picked = state.current ? state.queue.find((e) => e.job.id === state.current!.job) : undefined;
    this.targetX = picked?.pocket != null ? this.pocketPos(picked.job.tray, picked.pocket).x : null;
    this.drawJobs(state, chosenJob);
    this.drawArm(state);
    const placed = svg('text', { x: 240, y: 60, 'text-anchor': 'middle', style: `fill:${cssVar('--ink-2')};font-size:12px;font-weight:600` }, this.layer.text);
    placed.textContent = `${state.stats.completed} placed`;
  }

  private pocketPos(tray: number, k: number): { x: number; y: number } {
    const cols = 4;
    const col = k % cols, row = Math.floor(k / cols);
    return { x: TRAY_X[tray] - TRAY_W / 2 + 22 + col * 35, y: TRAY_Y - 11 + row * 22 };
  }

  private drawJobs(state: ReplayState, chosenJob: number | null): void {
    const g = this.layer.jobs;
    g.innerHTML = '';
    const overflow: number[] = [0, 0];
    for (const e of state.queue) {
      const tray = e.job.tray;
      const k = e.pocket;
      if (k == null) { overflow[tray]++; continue; }
      const carrying = state.inHand?.id === e.job.id && state.pickProgress > 0.3;
      if (carrying) continue;
      const { x, y } = this.pocketPos(tray, k);
      this.chip(g, x, y, e.job, {
        chosen: e.job.id === chosenJob,
        late: e.late,
        retry: e.attempts > 0,
        beyond: e.slot == null,
      });
    }
    overflow.forEach((n, tray) => {
      if (n > 0) svg('text', { x: TRAY_X[tray] + TRAY_W / 2 - 4, y: TRAY_Y - TRAY_H / 2 - 4, 'text-anchor': 'end' }, g).textContent = `+${n} more`;
    });
  }

  private chip(g: SVGGElement, x: number, y: number, job: Job, f: { chosen: boolean; late: boolean; retry: boolean; beyond: boolean }): void {
    const stroke = f.chosen ? cssVar('--brand') : f.late ? cssVar('--critical') : f.retry ? cssVar('--warning') : cssVar('--axis');
    const fill = f.chosen ? cssVar('--brand-wash') : cssVar('--surface');
    const grp = svg('g', { opacity: f.beyond ? 0.45 : 1 }, g);
    svg('rect', { x: x - 15, y: y - 9, width: 30, height: 18, rx: 3, fill, stroke, 'stroke-width': f.chosen ? 2 : 1 }, grp);
    const t = svg('text', { x, y: y + 4, 'text-anchor': 'middle', style: `fill:${f.chosen ? cssVar('--brand') : cssVar('--ink')};font-size:10px;font-weight:${f.chosen ? 700 : 500}` }, grp);
    t.textContent = `#${job.id}`;
  }

  private drawArm(state: ReplayState): void {
    const g = this.layer.arm;
    const held = this.layer.held;
    const text = this.layer.text;
    g.innerHTML = ''; held.innerHTML = ''; text.innerHTML = '';

    // wrist target
    let wx = TRAY_X[state.armTo], wy = HOVER_Y;
    let carrying: Job | null = null;
    let dropped = false;
    if (state.phase === 'travel') {
      const p = easeInOut(state.armProgress);
      wx = TRAY_X[state.armFrom] + (TRAY_X[state.armTo] - TRAY_X[state.armFrom]) * p;
      wy = HOVER_Y - 18 * Math.sin(Math.PI * p);
    } else if (state.phase === 'pick' && state.current) {
      const p = state.pickProgress;
      const isDrop = state.current.outcome === Outcome.Drop || state.current.outcome === Outcome.Scrap;
      const tx = this.targetX ?? TRAY_X[state.armTo];
      const out = OUT[state.armTo];
      if (p < 0.3) { // descend
        wx = tx; wy = HOVER_Y + (DOWN_Y - HOVER_Y) * easeInOut(p / 0.3);
      } else if (p < 0.45) { // lift with the part
        wx = tx; wy = DOWN_Y - (DOWN_Y - HOVER_Y) * easeInOut((p - 0.3) / 0.15); carrying = state.inHand;
      } else if (p < 0.8) { // carry to the output area
        const q = easeInOut((p - 0.45) / 0.35);
        if (isDrop && q > 0.5) { dropped = true; wx = tx + (out.x - tx) * 0.5; wy = HOVER_Y + (out.y + 10 - HOVER_Y) * 0.5; }
        else { wx = tx + (out.x - tx) * q; wy = HOVER_Y + (out.y + 10 - HOVER_Y) * q; carrying = isDrop && q > 0.5 ? null : state.inHand; }
      } else { // return to hover above the tray
        const q = easeInOut((p - 0.8) / 0.2);
        const fx = isDrop ? tx + (out.x - tx) * 0.5 : out.x, fy = isDrop ? HOVER_Y + (out.y + 10 - HOVER_Y) * 0.5 : out.y + 10;
        wx = fx + (tx - fx) * q; wy = fy + (HOVER_Y - fy) * q; dropped = isDrop;
      }
    } else if (state.phase === 'finished') {
      wx = SHOULDER.x; wy = HOVER_Y - 30;
    }

    const { ex, ey } = ik(wx, wy);
    const armCol = cssVar('--arm');
    svg('line', { x1: SHOULDER.x, y1: SHOULDER.y, x2: ex, y2: ey, stroke: armCol, 'stroke-width': 12, 'stroke-linecap': 'round' }, g);
    svg('line', { x1: ex, y1: ey, x2: wx, y2: wy, stroke: armCol, 'stroke-width': 9, 'stroke-linecap': 'round' }, g);
    svg('circle', { cx: ex, cy: ey, r: 7, fill: cssVar('--ink-2') }, g);
    svg('circle', { cx: SHOULDER.x, cy: SHOULDER.y, r: 7, fill: cssVar('--ink-2') }, g);
    // gripper
    const open = carrying ? 6 : 10;
    svg('path', { d: `M${wx - open},${wy + 4} l0,12 M${wx + open},${wy + 4} l0,12 M${wx - open},${wy + 4} L${wx + open},${wy + 4}`, stroke: cssVar('--ink'), 'stroke-width': 3, fill: 'none', 'stroke-linecap': 'round' }, g);
    if (carrying) this.chip(held, wx, wy + 18, carrying, { chosen: true, late: false, retry: false, beyond: false });
    if (dropped && state.current) {
      const tx = TRAY_X[state.armTo];
      const dx = tx + (OUT[state.armTo].x - tx) * 0.5, dy = TRAY_Y - 30;
      svg('text', { x: dx, y: dy, 'text-anchor': 'middle', style: `fill:${cssVar('--critical')};font-size:12px;font-weight:700` }, text).textContent =
        state.current.outcome === Outcome.Scrap ? `#${state.current.job} dropped · scrapped` : `#${state.current.job} dropped · re-queued`;
    }
    if (state.current?.careful && state.phase === 'pick') {
      svg('text', { x: wx, y: wy - 14, 'text-anchor': 'middle', style: `fill:${cssVar('--ink-2')};font-size:11px` }, text).textContent = 'careful';
    }
    if (state.phase === 'waiting') {
      svg('text', { x: SHOULDER.x, y: 40, 'text-anchor': 'middle' }, text).textContent = 'queue empty · waiting for arrivals';
    } else if (state.phase === 'finished') {
      svg('text', { x: SHOULDER.x, y: 40, 'text-anchor': 'middle' }, text).textContent = 'shift over';
    }
  }
}

function easeInOut(p: number): number {
  const q = Math.min(1, Math.max(0, p));
  return q < 0.5 ? 2 * q * q : 1 - Math.pow(-2 * q + 2, 2) / 2;
}

/** Two-link planar IK, elbow up. */
function ik(wx: number, wy: number): { ex: number; ey: number } {
  const dx = wx - SHOULDER.x, dy = wy - SHOULDER.y;
  let d = Math.hypot(dx, dy);
  const maxD = L1 + L2 - 2, minD = Math.abs(L1 - L2) + 2;
  d = Math.min(maxD, Math.max(minD, d));
  const base = Math.atan2(dy, dx);
  const cosA = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d);
  const a = Math.acos(Math.min(1, Math.max(-1, cosA)));
  // elbow up = the solution whose elbow has the smaller y
  const c1 = { ex: SHOULDER.x + L1 * Math.cos(base - a), ey: SHOULDER.y + L1 * Math.sin(base - a) };
  const c2 = { ex: SHOULDER.x + L1 * Math.cos(base + a), ey: SHOULDER.y + L1 * Math.sin(base + a) };
  return c1.ey <= c2.ey ? c1 : c2;
}
