import { useSize } from 'ahooks';
import { cn } from 'cn';
import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { SectionCard } from '@/components/section-card';
import { Button } from '@/components/ui/button';
import { fmt } from '@/lib/format-number';
import { niceStep } from '@/lib/nice-step';
import { OUTCOME_LABEL, type Replay, type Step } from '@/lib/replay';
import { cssVar } from '@/lib/svg-dom';
import { recipes } from '@/lib/ui-recipes';

interface Props {
  policies: string[];
  replays: Record<string, Replay>;
  scores: Record<string, number>;
  policy: string;
  t: number;
  horizon: number;
  onSeek: (t: number) => void;
}

const LEFT = 108, RIGHT = 12, TOP = 20, BOTTOM = 22, LANE = 30, BAR = 18;

/**
 * Every pick of every policy on this shift, one lane per policy, on the shift's time axis: the
 * active policy on top, the rest by final score. Picks after the playhead are faded, so the
 * lanes show how far each policy had got at the same moment. Click a bar to follow that job
 * across lanes, drag to zoom, click the axis to jump the replay there.
 */
export function PickGantt({ policies, replays, scores, policy, t, horizon, onSeek }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const W = Math.max(480, useSize(wrap)?.width ?? 640);
  const [follow, setFollow] = useState<number | null>(null);
  const [view, setView] = useState<[number, number] | null>(null);
  const [brush, setBrush] = useState<{ x0: number; x1: number } | null>(null);
  const [hover, setHover] = useState<{ x: number; y: number; html: string[] } | null>(null);
  const [theme, setTheme] = useState(0);
  useEffect(() => {
    const mq = matchMedia('(prefers-color-scheme: dark)');
    const bump = () => setTheme((x) => x + 1);
    mq.addEventListener('change', bump);
    return () => mq.removeEventListener('change', bump);
  }, []);
  useEffect(() => { setFollow(null); setView(null); }, [replays]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const c = useMemo(() => ({
    brand: cssVar('--brand'), ink: cssVar('--ink'), ink2: cssVar('--ink-2'), ink3: cssVar('--ink-3'), grid: cssVar('--grid'), axis: cssVar('--axis'),
    wash: cssVar('--wash'), travel: cssVar('--ink-3'), good: cssVar('--good'), warning: cssVar('--warning'), arm: cssVar('--arm'), critical: cssVar('--critical'),
    focus: cssVar('--focus'), surface: cssVar('--surface'),
  }), [theme]);
  const outcomeColour = [c.good, c.warning, c.arm, c.critical];

  const lanes = useMemo(() => {
    const rest = policies.filter((p) => p !== policy).sort((a, b) => (scores[b] ?? 0) - (scores[a] ?? 0));
    return [policy, ...rest];
  }, [policies, policy, scores]);
  const jobs = replays[policy].jobs;
  const [t0, t1] = view ?? [0, horizon];
  const iw = W - LEFT - RIGHT;
  const H = TOP + lanes.length * LANE + BOTTOM;
  const x = (tt: number) => LEFT + ((tt - t0) / (t1 - t0)) * iw;
  const toT = (px: number) => t0 + (Math.min(Math.max(px, LEFT), LEFT + iw) - LEFT) / iw * (t1 - t0);
  const step = niceStep((t1 - t0) / 8);
  const ticks: number[] = [];
  for (let tt = Math.ceil(t0 / step) * step; tt <= t1 + 1e-9; tt += step) ticks.push(tt);

  const fill = (s: Step) => outcomeColour[s.outcome];
  const svgX = (ev: MouseEvent<SVGElement>) => { const r = ev.currentTarget.getBoundingClientRect(); return (ev.clientX - r.left) * (W / r.width); };
  const tip = (p: string, s: Step) => {
    const job = jobs[s.job];
    const lateBy = s.end - job.deadline;
    return [
      `<b class="font-semibold">${p}</b> · pick ${s.index + 1}: job #${s.job}`,
      `Product|${job.product}, tray ${job.tray}`, `Outcome|${OUTCOME_LABEL[s.outcome]}`, `Mode|${s.careful ? 'careful' : 'normal'}`,
      `Arrived / deadline|${fmt(job.arrival)} s / ${fmt(job.deadline)} s`, `Picked|${fmt(s.start)} to ${fmt(s.end)} s`,
      `Waited before pick|${fmt(s.start - job.arrival)} s`, `${lateBy > 0 ? 'Late by' : 'Slack left'}|${fmt(Math.abs(lateBy))} s`,
      `Travel swing|${fmt(s.travel, 2)} s`, `Jobs waiting|${s.queueLen}`, `Reward this step|${fmt(s.reward, 3)}`,
    ];
  };

  const legend: [string, string][] = OUTCOME_LABEL.map((n, i) => [n, outcomeColour[i]]);

  return (
    <SectionCard
      title="Pick sequence"
      description="Every pick of every policy on this shift. Click a bar to follow a job, drag to zoom, click the axis to jump."
      action={
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={() => setView(null)} disabled={!view}>Reset zoom</Button>
          <Button size="sm" variant="outline" onClick={() => setFollow(null)} disabled={follow == null}>Clear job</Button>
        </div>
      }
    >
      <div className="flex flex-wrap gap-x-4 gap-y-1.5 text-[13px] text-ink-2 mb-2">
        {legend.map(([n, col]) => <span key={n} className={recipes.legendItem}><span className="inline-block w-3.5 h-2.5 rounded-sm" style={{ background: col }} />{n}</span>)}
        <span className={recipes.legendItem}><span className="inline-block w-3.5 h-2.5 rounded-sm" style={{ background: c.travel }} />tray switch</span>
        <span className={recipes.legendItem}><svg width="14" height="10" aria-hidden="true"><rect width="14" height="10" rx="2" fill={c.good} /><rect width="14" height="10" rx="2" fill="url(#gantt-hatch)" /></svg>careful</span>
        <span className={recipes.legendItem}><span className="inline-block w-3.5 h-2.5 rounded-sm opacity-35" style={{ background: c.good }} />after playhead</span>
      </div>
      <div ref={wrap} className="overflow-x-auto">
        <svg
          width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Pick sequence of every policy over the shift" className="cursor-crosshair select-none"
          onMouseDown={(ev) => { if ((ev.target as Element).closest('[data-pick]')) return; ev.preventDefault(); const px = svgX(ev); setBrush({ x0: px, x1: px }); }}
          onMouseMove={(ev) => { if (brush) setBrush({ ...brush, x1: svgX(ev) }); }}
          onMouseUp={(ev) => {
            if (!brush) return;
            const a = toT(Math.min(brush.x0, brush.x1)), b = toT(Math.max(brush.x0, brush.x1));
            setBrush(null);
            if (b - a > 2) setView([a, b]);
            else onSeek(toT(svgX(ev))); // a click: jump the replay there
          }}
          onMouseLeave={() => { setBrush(null); setHover(null); }}
        >
          <defs>
            <pattern id="gantt-hatch" width={5} height={5} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <line x1={0} y1={0} x2={0} y2={5} stroke={c.ink} strokeOpacity={0.55} strokeWidth={1.6} />
            </pattern>
            <clipPath id="gantt-clip"><rect x={LEFT} y={0} width={iw} height={H} /></clipPath>
          </defs>
          {ticks.map((tt) => (
            <g key={tt}>
              <line x1={x(tt)} x2={x(tt)} y1={TOP - 4} y2={H - BOTTOM + 2} stroke={c.grid} />
              <text x={x(tt)} y={H - 8} textAnchor="middle">{Math.round(tt)} s</text>
            </g>
          ))}
          <text x={LEFT} y={12}>time in shift</text>
          <g clipPath="url(#gantt-clip)">
            {follow != null && (() => {
              const j = jobs[follow];
              return (
                <g>
                  <rect x={x(j.arrival)} y={TOP - 4} width={Math.max(1, x(j.deadline) - x(j.arrival))} height={lanes.length * LANE} fill={c.wash} />
                  <line x1={x(j.deadline)} x2={x(j.deadline)} y1={TOP - 4} y2={TOP + lanes.length * LANE - 4} stroke={c.critical} strokeDasharray="3 3" />
                  <text x={x(j.deadline) + 4} y={TOP + 6} style={{ fill: c.ink2 }}>job #{follow} deadline</text>
                </g>
              );
            })()}
            {lanes.map((p, li) => {
              const y0 = TOP + li * LANE;
              const steps = replays[p].steps;
              return (
                <g key={p}>
                  <line x1={LEFT} x2={LEFT + iw} y1={y0 + BAR + 1} y2={y0 + BAR + 1} stroke={c.grid} />
                  {steps.map((s) => {
                    if (s.end < t0 || s.start > t1) return null;
                    const future = s.start > t;
                    const dim = follow != null && s.job !== follow;
                    const xs = x(s.start + s.travel), w = Math.max(1, x(s.end) - xs - 1);
                    const opacity = dim ? 0.18 : future ? 0.35 : 1;
                    return (
                      <g key={s.index} data-pick opacity={opacity} className="cursor-pointer"
                        onMouseMove={(ev) => setHover({ x: ev.clientX, y: ev.clientY, html: tip(p, s) })}
                        onMouseLeave={() => setHover(null)}
                        onMouseDown={(ev) => ev.preventDefault()}
                        onClick={(ev) => { ev.stopPropagation(); setFollow(follow === s.job ? null : s.job); }}
                      >
                        {s.travel > 0 && <rect x={x(s.start)} y={y0 + 6} width={Math.max(0.5, x(s.start + s.travel) - x(s.start))} height={BAR - 12} fill={c.travel} />}
                        <rect x={xs} y={y0} width={w} height={BAR} rx={Math.min(3, w / 2)} fill={fill(s)} stroke={follow === s.job ? c.focus : 'none'} strokeWidth={2} />
                        {s.careful && <rect x={xs} y={y0} width={w} height={BAR} rx={Math.min(3, w / 2)} fill="url(#gantt-hatch)" pointerEvents="none" />}
                        {w > 18 && <text x={xs + w / 2} y={y0 + BAR / 2 + 4} textAnchor="middle" pointerEvents="none" style={{ fill: '#0b0b0b', fontSize: 10 }}>{s.job}</text>}
                      </g>
                    );
                  })}
                </g>
              );
            })}
            {t >= t0 && t <= t1 && <line x1={x(t)} x2={x(t)} y1={TOP - 4} y2={TOP + lanes.length * LANE - 4} stroke={c.brand} strokeWidth={1.5} strokeDasharray="2 3" />}
            {brush && <rect x={Math.min(brush.x0, brush.x1)} y={TOP - 4} width={Math.abs(brush.x1 - brush.x0)} height={lanes.length * LANE} fill={c.wash} stroke={c.axis} />}
          </g>
          {lanes.map((p, li) => {
            const y0 = TOP + li * LANE;
            const active = p === policy;
            return (
              <g key={p}>
                <text x={LEFT - 10} y={y0 + BAR / 2 + 4} textAnchor="end">
                  <tspan style={{ fill: active ? c.brand : c.ink, fontSize: 12, fontWeight: 600 }}>{p}</tspan>
                  <tspan style={{ fill: c.ink2 }}> {fmt(scores[p], 1)}</tspan>
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      {hover && (
        <div role="tooltip" className="fixed pointer-events-none z-10 bg-surface text-ink border border-line rounded-lg px-2.5 py-2 text-[13px] shadow-lg max-w-[300px]"
          style={{ left: Math.min(hover.x + 14, innerWidth - 310), top: Math.min(hover.y + 14, innerHeight - 260) }}>
          <div dangerouslySetInnerHTML={{ __html: hover.html[0] }} />
          {hover.html.slice(1).map((row) => { const [k, v] = row.split('|'); return <div key={k} className={cn(recipes.tipRow)}><span>{k}</span><span>{v}</span></div>; })}
        </div>
      )}
    </SectionCard>
  );
}

