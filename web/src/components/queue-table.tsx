import { cn } from 'cn';
import { SectionCard } from '@/components/section-card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { fmt } from '@/lib/format-number';
import { fragilityLabel } from '@/lib/fragility-label';
import type { ReplayState } from '@/lib/replay';
import { recipes } from '@/lib/ui-recipes';

const FRAG_CLASS = { low: 'text-good', med: 'text-warning', high: 'text-critical' } as const;
/** Column label, numeric alignment, and a fixed width share so the table never reflows as cells change. */
const HEAD: [string, boolean, string][] = [
  ['Job', false, 'w-[11%]'], ['Product', false, 'w-[10%]'], ['Processing', false, 'w-[19%]'], ['Deadline', true, 'w-[11%]'],
  ['Fragility', false, 'w-[10%]'], ['Weight', true, 'w-[9%]'], ['Tray', true, 'w-[7%]'], ['', false, 'w-[23%]'],
];

/** The jobs waiting at time t in arrival order, with the active policy's pick marked. */
export function QueueTable({ state, chosenJob }: { state: ReplayState; chosenJob: number | null }) {
  const maxProc = Math.max(1, ...state.queue.map((e) => e.job.proc));
  return (
    <SectionCard
      title="Queue (arrival order)"
      action={<span className={recipes.note}>{state.queue.length} waiting · {Math.min(state.queue.length, 8)} visible to the policy</span>}
      className="flex-1 min-h-0"
      contentClassName="flex-1 min-h-[280px] max-h-[560px] overflow-auto"
    >
      <Table className="tabular-nums table-fixed">
        <TableHeader>
          <TableRow>{HEAD.map(([n, num, w], i) => <TableHead key={i} className={cn(recipes.th, w, num && recipes.num)}>{n}</TableHead>)}</TableRow>
        </TableHeader>
        <TableBody>
          {state.queue.length === 0 && (
            <TableRow><TableCell colSpan={8} className="text-ink-3">{state.finished ? 'Shift over.' : 'Queue empty, waiting for arrivals.'}</TableCell></TableRow>
          )}
          {state.queue.map((e) => {
            const chosen = e.job.id === chosenJob;
            const beyond = e.slot == null && !chosen;
            const ttd = e.job.deadline - state.t;
            const fl = fragilityLabel(e.job.fragility);
            return (
              <TableRow key={e.job.id} data-state={chosen ? 'selected' : undefined} className={cn(chosen && 'bg-brand-wash font-semibold hover:bg-brand-wash', beyond && 'text-ink-3')}>
                <TableCell>#{e.job.id}{e.attempts > 0 && <span className={cn(recipes.badge, 'text-warning')}>retry</span>}</TableCell>
                <TableCell>{e.job.product}</TableCell>
                <TableCell>
                  <span className="inline-block h-2 rounded-sm bg-brand align-middle mr-1.5" style={{ width: Math.round((e.job.proc / maxProc) * 56) }} />
                  {fmt(e.job.proc, 1)} s
                </TableCell>
                <TableCell className={cn(recipes.num, ttd < 0 ? 'text-critical' : ttd < 10 && 'text-warning')}>{ttd < 0 ? '−' : ''}{fmt(Math.abs(ttd), 0)} s</TableCell>
                <TableCell className={FRAG_CLASS[fl]}>{fl}</TableCell>
                <TableCell className={recipes.num}>{fmt(e.job.weight * 1000, 0)} g</TableCell>
                <TableCell className={recipes.num}>T{e.job.tray}</TableCell>
                <TableCell>
                  {chosen
                    ? <span className="text-brand font-semibold">◀ picking{state.current?.careful ? ' · careful' : ''}</span>
                    : e.slot == null ? 'beyond the 8 visible slots' : null}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </SectionCard>
  );
}
