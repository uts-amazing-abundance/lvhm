import { cn } from 'cn';
import { SectionCard } from '@/components/section-card';
import { clock } from '@/lib/format-clock';
import { fmt, signed } from '@/lib/format-number';
import { OUTCOME_LABEL, type ReplayState } from '@/lib/replay';
import { recipes } from '@/lib/ui-recipes';

interface Props {
  state: ReplayState;
  shiftLength: number;
  policy: string;
  activeScore: number;
  best: { name: string; score: number } | null;
}

/** Shift clock, running KPIs and the score cards of the active policy versus the best rule. */
export function StatusPanel({ state, shiftLength, policy, activeScore, best }: Props) {
  const s = state.stats;
  const tiles: { k: string; v: string; tone?: 'good' | 'bad' }[] = [
    { k: 'On time', v: String(s.onTime), tone: 'good' },
    { k: 'Late', v: String(s.late), tone: s.late ? 'bad' : undefined },
    { k: 'Travel', v: `${fmt(s.travel, 0)} s` },
    { k: 'Drops', v: String(s.drops), tone: s.drops ? 'bad' : undefined },
    { k: 'Careful', v: String(s.careful) },
    { k: 'Queue', v: String(state.queue.length) },
  ];
  if (state.finished && s.unfinished) tiles.push({ k: 'Unfinished', v: String(s.unfinished), tone: 'bad' });
  const lastPick = state.lastDecision ? `${OUTCOME_LABEL[state.lastDecision.outcome].toLowerCase()} on the last pick` : '';

  return (
    <SectionCard title="Shift clock" action={<span className={cn(recipes.mono, 'text-xl font-semibold')}>{clock(state.t)} / {clock(shiftLength)}</span>}>
      <div className="h-2 bg-grid rounded overflow-hidden mb-4" role="progressbar" aria-valuemin={0} aria-valuemax={shiftLength} aria-valuenow={Math.min(state.t, shiftLength)} aria-label="Shift progress">
        <div className="h-full bg-brand transition-[width] duration-75 ease-linear motion-reduce:transition-none" style={{ width: `${Math.min(100, (state.t / shiftLength) * 100)}%` }} />
      </div>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(84px,1fr))] gap-2">
        {tiles.map((tile) => (
          <div key={tile.k} className={recipes.kpi}>
            <div className={recipes.kpiKey}>{tile.k}</div>
            <div className={cn(recipes.kpiVal, tile.tone === 'good' && 'text-good', tile.tone === 'bad' && 'text-critical')}>{tile.v}</div>
          </div>
        ))}
      </div>
      <h3 className="text-base font-semibold mt-4 mb-2">Score</h3>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(140px,1fr))] gap-2">
        <div className={cn(recipes.score, recipes.scoreActive)}>
          <div className="text-[13px] text-ink-2">{policy}</div>
          <div className="text-3xl font-bold leading-none text-brand">{fmt(activeScore, 1)}</div>
          <div className="text-xs text-ink-3">{lastPick}</div>
        </div>
        {best && (
          <div className={recipes.score}>
            <div className="text-[13px] text-ink-2">best rule ({best.name})</div>
            <div className="text-3xl font-bold leading-none">{fmt(best.score, 1)}</div>
            <div className="text-xs text-ink-3">{policy} {signed(activeScore - best.score, 1)} so far</div>
          </div>
        )}
      </div>
    </SectionCard>
  );
}
