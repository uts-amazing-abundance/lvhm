import { InfoIcon } from 'lucide-react';
import { useId, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Slider } from '@/components/ui/slider';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { fmt } from '@/lib/format-number';
import type { ConditionData } from '@/types/trace.types';

export interface DeltaOption { value: number; condIndex: number }

interface Props {
  condition: ConditionData;
  deltaOptions: DeltaOption[];
  onDelta: (condIndex: number) => void;
}

const REWARD_TERMS: [term: string, sign: string, weight: string, when: string][] = [
  ['On-time bonus', '+', '1.0', 'the job finishes before its deadline'],
  ['Tardiness', '−', 'α × late seconds', 'any job in the system is past its deadline'],
  ['Travel', '−', 'β × seconds', 'the arm switches trays'],
  ['Damage', '−', 'δ per drop', 'a part is dropped (re-queued once, scrapped on the second drop)'],
  ['Unfinished job', '−', '1.0 per job', 'the shift ends with jobs still waiting'],
];

/** (i) next to the condition picker: what this test condition is and the penalty weights it runs under. */
export function ConditionInfoDialog({ condition, deltaOptions, onDelta }: Props) {
  const [open, setOpen] = useState(false);
  const linked = deltaOptions.length > 1;
  const [alpha, beta, delta] = condition.weights;
  const jobsPerShift = Math.round(condition.arrival_rate * 480);
  const products = condition.name.includes('p5')
    ? 'P1–P4 plus P5, a held-out product inside the training feature range'
    : condition.name.includes('p6')
      ? 'P1–P4 plus P6, a held-out product outside the training feature range'
      : 'P1–P4, the training products';

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger render={<Button variant="ghost" size="icon-xs" className="text-ink-3 hover:text-ink -my-1" aria-label="About this test condition and its penalty weights" />}>
        <InfoIcon className="size-3.5" />
      </DialogTrigger>
      <DialogContent className="sm:max-w-2xl gap-5">
        <DialogHeader>
          <DialogTitle>{condition.description}</DialogTitle>
          <DialogDescription>Test condition <code className="font-mono text-[13px] text-ink">{condition.name}</code></DialogDescription>
        </DialogHeader>

        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
          <dt className="text-ink-2">Arrivals</dt><dd className="m-0">{condition.arrival_rate} jobs/s, about {jobsPerShift} jobs per 480 s shift</dd>
          <dt className="text-ink-2">Products</dt><dd className="m-0">{products}. The policy never sees product identity, only weight and fragility.</dd>
          <dt className="text-ink-2">Shifts</dt><dd className="m-0">{linked ? 'The same pre-drawn shifts as the other damage-weight conditions, so only δ differs.' : 'Pre-drawn from a fixed seed, identical for every policy.'}</dd>
        </dl>

        <section>
          <h3 className="text-sm font-semibold mb-2">Reward per decision</h3>
          <Table>
            <TableHeader>
              <TableRow><TableHead>Term</TableHead><TableHead className="text-center">Sign</TableHead><TableHead>Weight</TableHead><TableHead>Charged when</TableHead></TableRow>
            </TableHeader>
            <TableBody>
              {REWARD_TERMS.map(([term, sign, weight, when]) => (
                <TableRow key={term}>
                  <TableCell className="font-medium">{term}</TableCell>
                  <TableCell className="text-center tabular-nums">{sign}</TableCell>
                  <TableCell className="whitespace-nowrap">{weight}</TableCell>
                  <TableCell className="whitespace-normal">{when}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <p className="text-[13px] text-ink-3 m-0 mt-2">α, β and δ are drawn per episode in training and are part of the observation, so a test condition can change them without retraining.</p>
        </section>

        <section>
          <h3 className="text-sm font-semibold mb-2">Weights in this condition</h3>
          <div className="grid grid-cols-[auto_1fr_auto] gap-x-3 gap-y-3 items-center">
            <WeightRow label="α tardiness" hint="per late second" value={alpha} max={0.05} range="training range 0.005–0.03" />
            <WeightRow label="β travel" hint="per second of switching" value={beta} max={0.5} range="training range 0.05–0.30" />
            {linked
              ? <SnapRow label="δ damage" hint="per drop" value={delta} options={deltaOptions} onPick={onDelta} />
              : <WeightRow label="δ damage" hint="per drop" value={delta} max={2.5} range="training range 0.2–1.5" />}
          </div>
          <p className="text-[13px] text-ink-3 m-0 mt-2">
            {linked
              ? `Drag δ to switch between the traced damage weights (${deltaOptions.map((o) => o.value).join(', ')}). The jobs stay identical; only the policies' responses change. 1.5 is the top of the training range, 2.4 lies outside it.`
              : 'This condition was traced at one set of weights, so the sliders are read-only.'}
          </p>
        </section>

        <section>
          <h3 className="text-sm font-semibold mb-1">Notes on this replay</h3>
          <ul className="text-[13px] text-ink-2 m-0 pl-4 list-disc space-y-0.5">
            <li>PPO and DQN replay their seed-0 checkpoint.</li>
            <li>Random draws its own choices in this trace, so its picks differ from the results-table run.</li>
            <li>The rule chips under "Chosen now" are recomputed in the browser from the queue the active policy saw.</li>
          </ul>
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
