import { useEventListener, useMemoizedFn, useRafInterval, useRequest } from 'ahooks';
import { useMemo, useRef, useState } from 'react';
import { CellView } from '@/components/cell-view';
import { ChosenNowPanel } from '@/components/chosen-now-panel';
import { PenaltyWeights, type DeltaOption } from '@/components/penalty-weights';
import { PlaybackToolbar } from '@/components/playback-toolbar';
import { QueueTable } from '@/components/queue-table';
import { ScoreChart, valueAt, type Series } from '@/components/score-chart';
import { StatusPanel } from '@/components/status-panel';
import { isRule } from '@/lib/dispatch-rules';
import { Replay } from '@/lib/replay';
import { recipes } from '@/lib/ui-recipes';
import type { Dataset } from '@/types/trace.types';

async function loadTrace(): Promise<Dataset> {
  const url = `${import.meta.env.BASE_URL}data/job_sequences.json`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`could not load ${url}: ${res.status}`);
  return (await res.json()) as Dataset;
}

export function App() {
  const { data: ds, error } = useRequest(loadTrace);

  return (
    <main className="max-w-[1440px] mx-auto px-4 pt-4 pb-12">
      <header className="mb-4">
        <h1 className="text-2xl font-semibold tracking-tight m-0 mb-1">Work cell dashboard</h1>
        <p className="text-ink-2 m-0 max-w-[80ch]">
          {error
            ? `Could not load the traced shifts: ${error.message}. Run the export cell in notebooks/experiments_phase2_1.ipynb and copy job_sequences.json to web/public/data/.`
            : ds
              ? `Replays every pick each policy made on the pre-drawn test shifts of the ${ds.profile} run. ${ds.conditions[0].shifts.length} shifts per condition, ${ds.shift_length} s per shift. Pick a policy, press play, and compare its choices with the rules.`
              : 'Loading traced shifts…'}
        </p>
      </header>
      {ds && <Dashboard ds={ds} />}
    </main>
  );
}

/** Selection and playback state; everything derived from the trace is memoised below. */
function Dashboard({ ds }: { ds: Dataset }) {
  const [cond, setCond] = useState(0);
  const [shiftIdx, setShiftIdx] = useState(0);
  const [policy, setPolicy] = useState(ds.policies.includes('DQN') ? 'DQN' : ds.policies[0]);
  const [t, setTState] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const tRef = useRef(0);
  const setT = useMemoizedFn((v: number) => { tRef.current = v; setTState(v); });

  const condition = ds.conditions[cond];
  const shift = condition.shifts[Math.min(shiftIdx, condition.shifts.length - 1)];
  const replays = useMemo(() => Object.fromEntries(ds.policies.map((p) => [p, new Replay(ds, shift, shift.runs[p])])) as Record<string, Replay>, [ds, shift]);
  const replay = replays[policy];
  const horizon = replay.horizon;

  const bestRule = useMemo(() => {
    const rules = ds.policies.filter((p) => isRule(p) && p !== policy);
    return rules.length ? rules.reduce((a, p) => (shift.runs[p].metrics.score > shift.runs[a].metrics.score ? p : a), rules[0]) : null;
  }, [ds, shift, policy]);
  const series = useMemo<Series[]>(
    () => ds.policies.map((p) => ({ name: p, pts: replays[p].cumulative(), role: p === policy ? 'active' : p === bestRule ? 'best' : 'other' })),
    [ds, replays, policy, bestRule],
  );
  const deltaOptions = useMemo<DeltaOption[]>(() => {
    // Conditions whose shifts are identical except for the damage weight.
    const sig = (i: number) => JSON.stringify([ds.conditions[i].arrival_rate, ds.conditions[i].shifts[0]?.jobs]);
    const mine = sig(cond);
    return ds.conditions
      .map((c, i) => ({ value: c.weights[2], condIndex: i, ok: sig(i) === mine && c.weights[0] === condition.weights[0] && c.weights[1] === condition.weights[1] }))
      .filter((o) => o.ok)
      .sort((a, b) => a.value - b.value);
  }, [ds, cond, condition]);

  const clampedT = Math.min(t, horizon);
  const state = useMemo(() => replay.stateAt(clampedT), [replay, clampedT]);
  const chosenJob = state.current ? state.current.job : null;
  const activeScore = valueAt(series.find((s) => s.role === 'active')?.pts ?? [], clampedT);
  const best = bestRule ? { name: bestRule, score: valueAt(series.find((s) => s.name === bestRule)?.pts ?? [], clampedT) } : null;

  // ---- playback: advance by wall-clock time scaled by speed, one update per animation frame
  const lastFrame = useRef<number | null>(null);
  useRafInterval(() => {
    const now = performance.now();
    if (lastFrame.current != null) {
      const next = Math.min(horizon, tRef.current + ((now - lastFrame.current) / 1000) * speed);
      setT(next);
      if (next >= horizon) setPlaying(false);
    }
    lastFrame.current = now;
  }, playing ? 0 : undefined);
  if (!playing) lastFrame.current = null;

  const togglePlay = useMemoizedFn(() => {
    if (!playing && tRef.current >= horizon) setT(0);
    setPlaying(!playing);
  });
  useEventListener('keydown', (ev) => {
    // Leave keys to form controls and to the sliders' and selects' own keyboard handling.
    if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLSelectElement || ev.target instanceof HTMLButtonElement) return;
    const focused = document.activeElement;
    if (focused && focused !== document.body && focused.closest('input, select, button, [role="slider"], [data-slot="slider"], [role="combobox"], [role="listbox"]')) return;
    if (ev.key === ' ') { ev.preventDefault(); togglePlay(); }
    if (ev.key === 'ArrowRight') setT(Math.min(horizon, tRef.current + 5));
    if (ev.key === 'ArrowLeft') setT(Math.max(0, tRef.current - 5));
  }, { target: document });

  return (
    <>
      <PlaybackToolbar
        ds={ds} policy={policy} cond={cond} shift={shiftIdx} speed={speed} playing={playing} t={clampedT} horizon={horizon}
        onPolicy={setPolicy}
        onCond={(i) => { setCond(i); setShiftIdx(0); }}
        onShift={(i) => { setShiftIdx(i); setT(0); }}
        onSpeed={setSpeed}
        onPlay={togglePlay}
        onReset={() => { setPlaying(false); setT(0); }}
        onScrub={setT}
      />
      <div className="grid grid-cols-1 lg:grid-cols-[3fr_2fr] gap-x-4">
        <div className="min-w-0">
          <PenaltyWeights weights={condition.weights} deltaOptions={deltaOptions} arrivalRate={condition.arrival_rate} onDelta={setCond} />
          <QueueTable state={state} policy={policy} chosenJob={chosenJob} />
          <ChosenNowPanel state={state} policy={policy} replay={replay} />
          <ScoreChart series={series} horizon={horizon} t={clampedT} />
        </div>
        <div className="min-w-0">
          <CellView state={state} chosenJob={chosenJob} />
          <StatusPanel state={state} shiftLength={ds.shift_length} policy={policy} activeScore={activeScore} best={best} />
        </div>
      </div>
      <p className={recipes.note}>
        Condition "{condition.name}": weights (α, β, δ) = ({condition.weights.join(', ')}), arrival rate {condition.arrival_rate} jobs/s.
        PPO and DQN replay their seed-0 checkpoint. Random draws its own choices in this trace, so its picks differ from the results-table run.
        Rule chips under "Chosen now" are recomputed in the browser from the queue the active policy saw.
      </p>
    </>
  );
}
