import { ConditionInfoDialog, type DeltaOption } from '@/components/condition-info-dialog';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { isRule } from '@/lib/dispatch-rules';
import { recipes } from '@/lib/ui-recipes';
import type { Dataset } from '@/types/trace.types';

interface Props {
  ds: Dataset;
  policy: string; cond: number; shift: number; speed: number; playing: boolean;
  onPolicy: (p: string) => void; onCond: (i: number) => void; onShift: (i: number) => void; onSpeed: (s: number) => void;
  onPlay: () => void; onReset: () => void;
  deltaOptions: DeltaOption[]; onDelta: (condIndex: number) => void;
}

const SPEEDS = [0.5, 1, 2, 4, 8, 16];

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className={recipes.field}>{label}{children}</label>;
}

/** Policy, condition, shift and speed pickers, play/reset, and the shift-time scrubber. */
export function PlaybackToolbar({ ds, policy, cond, shift, speed, playing, onPolicy, onCond, onShift, onSpeed, onPlay, onReset, deltaOptions, onDelta }: Props) {
  const shifts = ds.conditions[cond].shifts;
  const policyLabel = (p: string) => (isRule(p) ? p : `${p} ${p === 'PPO' || p === 'DQN' ? '(learned)' : '(solver)'}`);
  const policyItems = Object.fromEntries(ds.policies.map((p) => [p, policyLabel(p)]));
  const condItems = Object.fromEntries(ds.conditions.map((c, i) => [String(i), shortCondition(c.name, c.description)]));
  const shiftItems = Object.fromEntries(shifts.map((s, i) => [String(i), `Shift ${s.shift} (${s.jobs.length} jobs)`]));
  const speedItems = Object.fromEntries(SPEEDS.map((s) => [String(s), `${s}×`]));

  return (
    <Card className="mb-4 ring-line" aria-label="Playback controls">
      <CardContent className="flex flex-wrap gap-x-4 gap-y-3 items-end">
        <Field label="Policy">
          <Select value={policy} items={policyItems} onValueChange={(v) => v != null && onPolicy(v)}>
            <SelectTrigger className="h-10 min-w-40 bg-surface"><SelectValue /></SelectTrigger>
            <SelectContent>{ds.policies.map((p) => <SelectItem key={p} value={p}>{policyLabel(p)}</SelectItem>)}</SelectContent>
          </Select>
        </Field>
        <div className={recipes.field}>
          <span className="inline-flex items-center gap-1 leading-none">Test condition <ConditionInfoDialog condition={ds.conditions[cond]} deltaOptions={deltaOptions} onDelta={onDelta} /></span>
          <Select value={String(cond)} items={condItems} onValueChange={(v) => v != null && onCond(Number(v))}>
            <SelectTrigger className="h-10 w-64 bg-surface" aria-label="Test condition"><SelectValue /></SelectTrigger>
            <SelectContent>{ds.conditions.map((c, i) => <SelectItem key={c.name} value={String(i)}>{c.description}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <Field label="Shift">
          <Select value={String(Math.min(shift, shifts.length - 1))} items={shiftItems} onValueChange={(v) => v != null && onShift(Number(v))}>
            <SelectTrigger className="h-10 min-w-36 bg-surface"><SelectValue /></SelectTrigger>
            <SelectContent>{shifts.map((s, i) => <SelectItem key={s.shift} value={String(i)}>Shift {s.shift} ({s.jobs.length} jobs)</SelectItem>)}</SelectContent>
          </Select>
        </Field>
        <Field label="Speed">
          <Select value={String(speed)} items={speedItems} onValueChange={(v) => v != null && onSpeed(Number(v))}>
            <SelectTrigger className="h-10 min-w-20 bg-surface"><SelectValue /></SelectTrigger>
            <SelectContent>{SPEEDS.map((s) => <SelectItem key={s} value={String(s)}>{s}×</SelectItem>)}</SelectContent>
          </Select>
        </Field>
        <div className="flex gap-2">
          <Button size="lg" className="h-10 min-w-26 text-base" onClick={onPlay} aria-label={playing ? 'Pause' : 'Play'}>{playing ? '❚❚ Pause' : '▶ Play'}</Button>
          <Button size="lg" variant="outline" className="h-10 text-base" onClick={onReset}>Reset shift</Button>
        </div>
      </CardContent>
    </Card>
  );
}

/** Short trigger label per condition; the full description stays in the dropdown and the info dialog. */
const SHORT_CONDITION: Record<string, string> = {
  familiar: '1. Normal day, P1–P4',
  unseen_p5: '2a. Unseen P5 (inside range)',
  unseen_p6: '2b. Unseen P6 (outside range)',
  busy: '3. Busy day, 2× arrivals',
  damage_hi: '4. Damage δ at range top',
  damage_oor: '5. Damage δ outside range',
};
const shortCondition = (name: string, description: string): string => SHORT_CONDITION[name] ?? description;
