import { InfoIcon } from 'lucide-react';
import { useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import { fmt } from '@/lib/format-number';
import type { ConditionData } from '@/types/trace.types';

export interface DeltaOption { value: number; condIndex: number }

interface Props {
  condition: ConditionData;
  deltaOptions: DeltaOption[];
  onDelta: (condIndex: number) => void;
}

/** (i) next to the condition picker: what this test condition is and the penalty weights it runs under. */
export function ConditionInfoDialog({ condition, deltaOptions, onDelta }: Props) {
  const [open, setOpen] = useState(false);
  const linked = deltaOptions.length > 1;
  const [alpha, beta, delta] = condition.weights;
  const jobsPerShift = Math.round(condition.arrival_rate * 480);
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger render={<Button variant="ghost" size="icon-xs" className="text-ink-3 hover:text-ink -my-1" aria-label="About this test condition and its penalty weights" />}>
        <InfoIcon className="size-3.5" />
      </DialogTrigger>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>{condition.description}</DialogTitle>
          <DialogDescription>
            Condition <code className="font-mono text-[13px]">{condition.name}</code> · arrival rate {condition.arrival_rate} jobs/s (≈{jobsPerShift} jobs per 480 s shift)
            {linked ? ' · the same pre-drawn shifts as the other damage-weight conditions' : ''}
          </DialogDescription>
        </DialogHeader>

        <section>
          <h3 className="text-sm font-semibold mb-1">Reward per decision</h3>
          <p className="text-sm text-ink-2 m-0">
            +1 for a job finished by its deadline · −α per second any job in the system is late · −β per second of tray-switch travel ·
            −δ per dropped part (re-queued once, scrapped on the second drop) · −1 per job still waiting when the shift ends.
            α, β, δ are sampled per episode in training and passed to the policy in the observation, which is what lets a condition change them without retraining.
          </p>
        </section>

        <section className="grid grid-cols-[auto_1fr_auto] gap-x-3 gap-y-3 items-center">
          <WeightRow label="α tardiness" hint="per late second" value={alpha} max={0.05} range="training range 0.005–0.03" />
          <WeightRow label="β travel" hint="per second of switching" value={beta} max={0.5} range="training range 0.05–0.30" />
          {linked
            ? <SnapRow label="δ damage" hint="per drop" value={delta} options={deltaOptions} onPick={(i) => { onDelta(i); }} />
            : <WeightRow label="δ damage" hint="per drop" value={delta} max={2.5} range="training range 0.2–1.5" />}
          <p className="col-span-full text-[13px] text-ink-3 m-0">
            {linked
              ? `This shift was traced at ${deltaOptions.length} damage weights (${deltaOptions.map((o) => o.value).join(', ')}). Move δ to switch between them: the jobs stay identical and only the policies' responses change. 1.5 is the top of the training range; 2.4 is outside it.`
              : 'This condition was traced at one set of weights, so the sliders are read-only.'}
          </p>
        </section>

        <section>
          <h3 className="text-sm font-semibold mb-1">Products in this condition</h3>
          <p className="text-sm text-ink-2 m-0">
            Jobs are drawn uniformly from P1–P4 (training products){condition.name.includes('p5') ? ' plus P5, a held-out product inside the training feature range' : ''}
            {condition.name.includes('p6') ? ' plus P6, a held-out product outside the training feature range' : ''}. The policy never sees product identity, only weight and fragility.
          </p>
        </section>
      </DialogContent>
    </Dialog>
  );
}

function WeightRow({ label, hint, value, max, range }: { label: string; hint: string; value: number; max: number; range: string }) {
  const id = useId();
  return (
    <>
      <span id={id} className="text-sm whitespace-nowrap"><span className="text-ink">{label}</span> <span className="text-ink-3 text-[12px]">{hint}</span></span>
      <div><Slider value={[value]} min={0} max={max} step={0.001} disabled aria-labelledby={id} /><span className="text-[11px] text-ink-3">{range}</span></div>
      <output aria-labelledby={id} className="text-sm min-w-[4ch] text-right tabular-nums">{fmt(value, 3)}</output>
    </>
  );
}

function SnapRow({ label, hint, value, options, onPick }: { label: string; hint: string; value: number; options: DeltaOption[]; onPick: (i: number) => void }) {
  const id = useId();
  const values = options.map((o) => o.value);
  const snap = (v: number) => options.reduce((a, b) => (Math.abs(b.value - v) < Math.abs(a.value - v) ? b : a));
  const [drag, setDrag] = useState<number | null>(null);
  const shown = drag ?? value;
  return (
    <>
      <span id={id} className="text-sm whitespace-nowrap"><span className="text-ink">{label}</span> <span className="text-ink-3 text-[12px]">{hint}</span></span>
      <div>
        <Slider
          value={[shown]} min={Math.min(...values)} max={Math.max(...values)} step={0.01}
          aria-labelledby={id} aria-valuetext={`${shown}; snaps to ${values.join(', ')}`}
          onValueChange={(v) => setDrag(Array.isArray(v) ? v[0] : v)}
          onValueCommitted={(v) => { setDrag(null); const n = snap(Array.isArray(v) ? v[0] : v); if (n.value !== value) onPick(n.condIndex); }}
        />
        <span className="text-[11px] text-ink-3">snaps to {values.join(' · ')}</span>
      </div>
      <output aria-labelledby={id} className="text-sm min-w-[4ch] text-right tabular-nums">{fmt(shown, 3)}</output>
    </>
  );
}
