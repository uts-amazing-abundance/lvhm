import { useSize } from 'ahooks';
import { cn } from 'cn';
import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { SectionCard } from '@/components/section-card';
import { fmt } from '@/lib/format-number';
import { niceStep } from '@/lib/nice-step';
import { cssVar } from '@/lib/svg-dom';
import { recipes } from '@/lib/ui-recipes';

export interface Series {
  name: string;
  pts: [number, number][];
  role: 'active' | 'best' | 'other';
}

export function valueAt(pts: [number, number][], t: number): number {
  let v = 0;
  for (const p of pts) { if (p[0] <= t) v = p[1]; else break; }
  return v;
}

const LEFT = 44, RIGHT = 110, TOP = 12, BOTTOM = 28;

interface Props {
  title: string;
  description: string;
  series: Series[];
  horizon: number;
  t: number;
  /** Draw each series as a step function (value holds until the next point) instead of a line. */
  step?: boolean;
  height?: number;
  decimals?: number;
}

export function PolicyLineChart({ title, description, series, horizon, t, step: stepped = false, height: H = 260, decimals = 1 }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const W = Math.max(360, useSize(wrap)?.width ?? 600);
  const [theme, setTheme] = useState(0);
  const [hover, setHover] = useState<{ x: number; y: number; t: number } | null>(null);
  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const bump = () => setTheme((x) => x + 1);
    mq.addEventListener('change', bump);
    return () => mq.removeEventListener('change', bump);
  }, []);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const colours = useMemo(() => ({ brand: cssVar('--brand'), ink2: cssVar('--ink-2'), axis: cssVar('--axis'), grid: cssVar('--grid') }), [theme]);
  const iw = W - LEFT - RIGHT, ih = H - TOP - BOTTOM;
  const { y0, y1, step } = useMemo(() => {
    const ys = series.flatMap((s) => s.pts.map((p) => p[1]));
    let lo = Math.min(0, ...ys), hi = Math.max(1, ...ys);
    const st = niceStep((hi - lo) / 5);
    lo = Math.floor(lo / st) * st; hi = Math.ceil(hi / st) * st;
    return { y0: lo, y1: hi, step: st };
  }, [series]);
  const x = (tt: number) => LEFT + (Math.min(tt, horizon) / horizon) * iw;
  const y = (v: number) => TOP + (1 - (v - y0) / (y1 - y0)) * ih;
  const paths = useMemo(() => {
    const order = [...series.filter((s) => s.role === 'other'), ...series.filter((s) => s.role === 'best'), ...series.filter((s) => s.role === 'active')];
    return order.map((s) => ({
      s,
      d: s.pts.map((p, i) => {
        const px = x(p[0]).toFixed(1), py = y(p[1]).toFixed(1);
        if (i === 0) return `M${px},${py}`;
        return stepped ? `H${px}V${py}` : `L${px},${py}`;
      }).join('') + (stepped ? `H${x(horizon).toFixed(1)}` : ''),
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, horizon, W, y0, y1, stepped]);
  const colour = (r: Series['role']) => (r === 'active' ? colours.brand : r === 'best' ? colours.ink2 : colours.axis);
  const active = series.find((s) => s.role === 'active');
  const best = series.find((s) => s.role === 'best');
  const gridLines: number[] = [];
  for (let v = y0; v <= y1 + 1e-9; v += step) gridLines.push(v);
  const ticks: number[] = [];
  for (let tt = 0; tt <= horizon; tt += 120) ticks.push(tt);

  const onMove = (ev: MouseEvent<SVGRectElement>) => {
    const r = ev.currentTarget.ownerSVGElement!.getBoundingClientRect();
    const tt = (((ev.clientX - r.left) * (W / r.width)) - LEFT) / iw * horizon;
    setHover({ x: ev.clientX, y: ev.clientY, t: tt });
  };
  const hoverVals = hover ? series.map((s) => [s.name, valueAt(s.pts, hover.t)] as const).sort((a, b) => b[1] - a[1]) : [];

  return (
    <SectionCard title={title} action={<span className={recipes.note}>{description}</span>}>
      <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[13px] text-ink-2 mb-2">
        {active && <span className={recipes.legendItem}><span className={recipes.swatch} style={{ background: colours.brand }} />{active.name}</span>}
        {best && <span className={recipes.legendItem}><span className={cn(recipes.swatch, 'h-[2px]')} style={{ background: colours.ink2 }} />best dispatch rule ({best.name})</span>}
        <span className={recipes.legendItem}><span className={cn(recipes.swatch, 'h-[2px]')} style={{ background: colours.axis }} />other policies</span>
      </div>
      <div ref={wrap} className="overflow-x-auto">
        <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Cumulative score of every policy over the shift">
          {gridLines.map((v) => (
            <g key={v}>
              <line x1={LEFT} x2={LEFT + iw} y1={y(v)} y2={y(v)} stroke={v === 0 ? colours.axis : colours.grid} />
              <text x={LEFT - 6} y={y(v) + 4} textAnchor="end">{Math.round(v)}</text>
            </g>
          ))}
          {ticks.map((tt) => <text key={tt} x={x(tt)} y={H - 8} textAnchor="middle">{tt} s</text>)}
          <text x={LEFT + iw + 8} y={H - 8}>shift time →</text>
          {paths.map(({ s, d }) => (
            <path key={s.name} d={d} fill="none" stroke={colour(s.role)} strokeWidth={s.role === 'active' ? 2.5 : s.role === 'best' ? 1.8 : 1.2} strokeDasharray={s.role === 'best' ? '5 4' : undefined} strokeLinejoin="round" />
          ))}
          {active && <text x={LEFT + iw + 8} y={TOP + 14} style={{ fill: colours.brand, fontWeight: 650 }}>{active.name} {fmt(valueAt(active.pts, t), decimals)}</text>}
          {best && <text x={LEFT + iw + 8} y={TOP + 30} style={{ fill: colours.ink2 }}>{best.name} {fmt(valueAt(best.pts, t), decimals)}</text>}
          <line x1={x(t)} x2={x(t)} y1={TOP} y2={TOP + ih} stroke={colours.brand} strokeWidth={1.5} strokeDasharray="2 3" />
          <rect x={LEFT} y={TOP} width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
        </svg>
      </div>
      {hover && (
        <div role="tooltip" className="fixed pointer-events-none z-10 bg-surface text-ink border border-line rounded-lg px-2.5 py-2 text-[13px] shadow-lg max-w-[300px]"
          style={{ left: Math.min(hover.x + 14, innerWidth - 220), top: Math.min(hover.y + 14, innerHeight - 260) }}>
          <b className="font-semibold">{fmt(hover.t, 0)} s into the shift</b>
          {hoverVals.map(([p, v]) => <div key={p} className={recipes.tipRow}><span>{p}</span><span>{fmt(v, decimals + 1)}</span></div>)}
        </div>
      )}
    </SectionCard>
  );
}
