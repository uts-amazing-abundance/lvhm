import { useId, useState } from 'react';
import { SectionCard } from '@/components/section-card';
import { Slider } from '@/components/ui/slider';
import { fmt } from '@/lib/format-number';

export interface DeltaOption { value: number; condIndex: number }

interface Props {
  weights: [number, number, number];
  deltaOptions: DeltaOption[];
  arrivalRate: number;
  onDelta: (condIndex: number) => void;
}

/** α, β, δ of the selected condition; δ snaps between the traced damage conditions when they share shifts. */
export function PenaltyWeights({ weights, deltaOptions, arrivalRate, onDelta }: Props) {
  const linked = deltaOptions.length > 1;
  return (
    <SectionCard
      title="Penalty weights"
      description="α, β and δ of the selected condition. Where the traced data holds the same shifts under several δ values, the δ slider switches between them."
    >
      <div className="grid grid-cols-[auto_1fr_auto] gap-x-3 gap-y-3 items-center">
        <WeightRow label="α tardiness" value={weights[0]} max={0.05} />
        <WeightRow label="β travel" value={weights[1]} max={0.5} />
        {linked
          ? <SnapRow label="δ damage" value={weights[2]} options={deltaOptions} onPick={onDelta} />
          : <WeightRow label="δ damage" value={weights[2]} max={2.0} />}
        <p className="col-span-full text-[13px] text-ink-3 m-0">
          {linked
            ? `Arrival rate ${arrivalRate} jobs/s. The same ${deltaOptions.length} traced δ values share identical shifts, so moving δ keeps the jobs and swaps the policy's response.`
            : `Arrival rate ${arrivalRate} jobs/s. This condition was traced at one δ only, so the sliders are read-only.`}
        </p>
      </div>
    </SectionCard>
  );
}

function WeightRow({ label, value, max }: { label: string; value: number; max: number }) {
  const id = useId();
  return (
    <>
      <span id={id} className="text-sm text-ink-2 whitespace-nowrap">{label}</span>
      <Slider value={[value]} min={0} max={max} step={0.001} disabled aria-labelledby={id} />
      <output aria-labelledby={id} className="text-sm min-w-[4ch] text-right tabular-nums">{fmt(value, 3)}</output>
    </>
  );
}

function SnapRow({ label, value, options, onPick }: { label: string; value: number; options: DeltaOption[]; onPick: (i: number) => void }) {
  const id = useId();
  const values = options.map((o) => o.value);
  const snap = (v: number) => options.reduce((a, b) => (Math.abs(b.value - v) < Math.abs(a.value - v) ? b : a));
  // The thumb follows the drag; the condition only switches once the drag is released.
  const [drag, setDrag] = useState<number | null>(null);
  const shown = drag ?? value;
  return (
    <>
      <span id={id} className="text-sm text-ink-2 whitespace-nowrap">{label}</span>
      <Slider
        value={[shown]} min={Math.min(...values)} max={Math.max(...values)} step={0.01}
        aria-labelledby={id} aria-valuetext={`${shown}; snaps to ${values.join(', ')}`}
        onValueChange={(v) => setDrag(Array.isArray(v) ? v[0] : v)}
        onValueCommitted={(v) => { setDrag(null); const n = snap(Array.isArray(v) ? v[0] : v); if (n.value !== value) onPick(n.condIndex); }}
      />
      <output aria-labelledby={id} className="text-sm min-w-[4ch] text-right tabular-nums">{fmt(shown, 3)}</output>
    </>
  );
}
