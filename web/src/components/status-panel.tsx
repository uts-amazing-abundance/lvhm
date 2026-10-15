import { cn } from 'cn';
import { SectionCard } from '@/components/section-card';
import { Slider } from '@/components/ui/slider';
import { clock } from '@/lib/format-clock';
import { fmt, signed } from '@/lib/format-number';
import { OUTCOME_LABEL, type ReplayState } from '@/lib/replay';
import { recipes } from '@/lib/ui-recipes';

interface Props {
  state: ReplayState;
  shiftLength: number;
  horizon: number;
  onSeek: (t: number) => void;
  policy: string;
  activeScore: number;
  best: { name: string; score: number } | null;
}

/** Shift clock, running KPIs and the score cards of the active policy versus the best rule. */
export function StatusPanel({ state, shiftLength, horizon, onSeek, policy, activeScore, best }: Props) {
  const s = state.stats;
  const tiles: { k: string; v: string; tone?: 'good' | 'bad' }[] = [
    { k: 'On time', v: String(s.onTime), tone: 'good' },
    { k: 'Late', v: String(s.late), tone: s.late ? 'bad' : undefined },
    { k: 'Travel', v: `${fmt(s.travel, 0)} s` },
    { k: 'Drops', v: String(s.drops), tone: s.drops ? 'bad' : undefined },
    { k: 'Careful', v: String(s.careful) },
    { k: 'Queue', v: String(state.queue.length) },
  ];
  // Always present so the tile grid keeps its height; it only gets a value once the shift is over.
  tiles.push({ k: 'Unfinished', v: state.finished ? String(s.unfinished) : '–', tone: state.finished && s.unfinished ? 'bad' : undefined });
  const lastPick = state.lastDecision ? `${OUTCOME_LABEL[state.lastDecision.outcome].toLowerCase()} on the last pick` : '';

  return (
    <SectionCard title="Shift clock" action={<span className={cn(recipes.mono, 'text-xl font-semibold')}>{clock(state.t)} / {clock(shiftLength)}</span>}>
      {/* The clock's progress bar doubles as the scrubber: click or drag to move the replay. */}
      <Slider value={[state.t]} min={0} max={horizon} step={0.1} onValueChange={(v) => onSeek(Array.isArray(v) ? v[0] : v)} aria-label="Shift time" className="mb-4 min-h-6" />
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
            <div className="text-[13px] text-ink-2">best baseline ({best.name})</div>
            <div className="text-3xl font-bold leading-none">{fmt(best.score, 1)}</div>
            <div className="text-xs text-ink-3">{policy} {signed(activeScore - best.score, 1)} so far</div>
          </div>
        )}
      </div>
    </SectionCard>
  );
}
