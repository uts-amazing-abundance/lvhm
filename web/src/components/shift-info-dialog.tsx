import { InfoIcon } from 'lucide-react';
import { useMemo, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { fmt } from '@/lib/format-number';
import { fragilityLabel } from '@/lib/fragility-label';
import type { Dataset, ShiftData } from '@/types/trace.types';

interface Props {
  ds: Dataset;
  shift: ShiftData;
  /** How many shifts of this condition the trace carries (the dropdown's length). */
  traced: number;
}

/** Shifts per condition behind the results tables; the trace replays only the first few of them. */
const REPORT_SHIFTS: Record<string, number> = { smoke: 3, partc: 30, full: 100 };

/** (i) next to the shift picker: what a shift is, how the test shifts are drawn, and what is in this one. */
export function ShiftInfoDialog({ ds, shift, traced }: Props) {
  const [open, setOpen] = useState(false);
  const reportShifts = REPORT_SHIFTS[ds.profile];
  const summary = useMemo(() => summarise(ds, shift), [ds, shift]);

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger render={<Button variant="ghost" size="icon-xs" className="text-ink-3 hover:text-ink -my-1" aria-label="About shifts and this shift's jobs" />}>
        <InfoIcon className="size-3.5" />
      </DialogTrigger>
      <DialogContent className="sm:max-w-2xl gap-5">
        <DialogHeader>
          <DialogTitle>Shift {shift.shift}: {shift.jobs.length} jobs in {ds.shift_length} s</DialogTitle>
          <DialogDescription>One shift is one episode: an {ds.shift_length / 60}-minute run of the cell from an empty queue to the end of the clock.</DialogDescription>
        </DialogHeader>

        <section>
          <h3 className="text-sm font-semibold mb-1">What a shift is</h3>
          <ul className="text-[13px] text-ink-2 m-0 pl-4 list-disc space-y-0.5">
            <li>Jobs arrive at random moments during the {ds.shift_length} s (a Poisson stream), each with a tray, a product and a deadline. The policy only sees what has arrived.</li>
            <li>Every time the arm is free, the policy picks one waiting job and whether to handle it carefully. A shift has about 70 such decisions.</li>
            <li>The score of a shift is the sum of the rewards of its decisions. A pick that starts before the clock runs out finishes, so the last few seconds can overrun.</li>
          </ul>
        </section>

        <section>
          <h3 className="text-sm font-semibold mb-1">How the test shifts are drawn</h3>
          <ul className="text-[13px] text-ink-2 m-0 pl-4 list-disc space-y-0.5">
            <li>They are generated once from a fixed seed, so every policy faces exactly the same arrivals, deadlines and products. A score difference between two policies on the same shift is therefore a paired observation, not noise.</li>
            <li>{reportShifts ? `The results tables average ${reportShifts} such shifts per condition; this trace replays the first ${traced} of them.` : `This trace replays ${traced} shifts per condition.`}</li>
            <li>Conditions differ in the product mix, the arrival rate or the penalty weights, not in how shifts are drawn. The damage-weight conditions even share the same shifts.</li>
            <li>Training never sees these shifts: it draws fresh ones from a different seed, and validation uses a third set.</li>
          </ul>
        </section>

        <section>
          <h3 className="text-sm font-semibold mb-2">This shift</h3>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
            <dt className="text-ink-2">Jobs</dt><dd className="m-0">{shift.jobs.length} total, tray 0 / tray 1 = {summary.byTray[0]} / {summary.byTray[1]}</dd>
            <dt className="text-ink-2">Arrivals</dt><dd className="m-0">first at {fmt(summary.firstArrival, 0)} s, last at {fmt(summary.lastArrival, 0)} s, busiest minute has {summary.peakPerMinute} arrivals</dd>
            <dt className="text-ink-2">Deadlines</dt><dd className="m-0">{fmt(summary.meanSlack, 0)} s after arrival on average, tightest {fmt(summary.minSlack, 0)} s</dd>
            <dt className="text-ink-2">Processing</dt><dd className="m-0">{fmt(summary.totalProc, 0)} s of nominal pick time in a {ds.shift_length} s shift ({fmt((summary.totalProc / ds.shift_length) * 100, 0)}% load before travel and careful handling)</dd>
          </dl>
          <Table className="mt-3">
            <TableHeader>
              <TableRow><TableHead>Product</TableHead><TableHead className="text-right">Jobs</TableHead><TableHead className="text-right">Weight</TableHead><TableHead>Fragility</TableHead></TableRow>
            </TableHeader>
            <TableBody>
              {summary.products.map((p) => (
                <TableRow key={p.name}>
                  <TableCell className="font-medium">{p.name}</TableCell>
                  <TableCell className="text-right tabular-nums">{p.count}</TableCell>
                  <TableCell className="text-right tabular-nums">{fmt(p.weight * 1000, 0)} g</TableCell>
                  <TableCell>{fragilityLabel(p.fragility)} ({fmt(p.fragility, 2)})</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          <p className="text-[13px] text-ink-3 m-0 mt-2">The policy never sees the product name, only its weight and fragility.</p>
        </section>
      </DialogContent>
    </Dialog>
  );
}

function summarise(ds: Dataset, shift: ShiftData) {
  const jobs = shift.jobs;
  const byTray = [0, 0];
  const counts = new Map<string, number>();
  const perMinute = new Array(Math.ceil(ds.shift_length / 60)).fill(0) as number[];
  let totalProc = 0, slackSum = 0, minSlack = Infinity;
  for (const [arrival, deadline, tray, product, proc] of jobs) {
    byTray[tray] = (byTray[tray] ?? 0) + 1;
    counts.set(product, (counts.get(product) ?? 0) + 1);
    const m = Math.min(perMinute.length - 1, Math.floor(arrival / 60));
    perMinute[m] += 1;
    totalProc += proc;
    const slack = deadline - arrival;
    slackSum += slack;
    if (slack < minSlack) minSlack = slack;
  }
  const products = [...counts.entries()]
    .map(([name, count]) => ({ name, count, weight: ds.products[name]?.[0] ?? 0, fragility: ds.products[name]?.[1] ?? 0 }))
    .sort((a, b) => a.name.localeCompare(b.name));
  return {
    byTray,
    products,
    firstArrival: jobs.length ? Math.min(...jobs.map((j) => j[0])) : 0,
    lastArrival: jobs.length ? Math.max(...jobs.map((j) => j[0])) : 0,
    peakPerMinute: Math.max(0, ...perMinute),
    totalProc,
    meanSlack: jobs.length ? slackSum / jobs.length : 0,
    minSlack: jobs.length ? minSlack : 0,
  };
}
