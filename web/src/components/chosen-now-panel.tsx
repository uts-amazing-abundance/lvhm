import { cn } from 'cn';
import { SectionCard } from '@/components/section-card';
import { RULE_LONG, RULES, rulePick } from '@/lib/dispatch-rules';
import { clock } from '@/lib/format-clock';
import type { Replay, ReplayState } from '@/lib/replay';
import { recipes } from '@/lib/ui-recipes';

/**
 * The Part B "Chosen now" panel: at the latest decision, the job the active policy picked next to
 * the job each dispatch rule would have picked from the very same queue. Agreement shows whether
 * the learned policy is doing something a simple rule would also do, or something none of them do.
 */
export function ChosenNowPanel({ state, policy, replay }: { state: ReplayState; policy: string; replay: Replay }) {
  const decision = state.lastDecision;
  const visible = decision ? replay.decisionQueue(decision).slice(0, 8) : [];
  const picks = decision
    ? RULES.filter((r) => r !== policy).map((rule) => {
      const i = rulePick(rule, visible, decision.prevTray, decision.index);
      return { rule, job: i >= 0 ? visible[i].job.id : null };
    })
    : [];
  const agree = picks.filter((p) => decision && p.job === decision.job).length;
  const live = state.phase === 'pick' || state.phase === 'travel';
  const note = !decision
    ? 'No decision yet.'
    : `${live ? 'Decision' : 'Last decision'} ${decision.index + 1} at ${clock(decision.start)} · ${visible.length} job${visible.length === 1 ? '' : 's'} visible · ${agree} of ${picks.length} rules agree`;

  return (
    <SectionCard
      title="Chosen now"
      description="The job the policy picked at this decision, next to the job each dispatch rule would have picked from the same queue. Rules that agree are outlined."
      action={<span className={cn(recipes.note, 'whitespace-nowrap')}>{note}</span>}
    >
      {decision && (
        <div className="grid grid-cols-7 gap-2">
          <Cell who={policy} job={decision.job} sub={decision.careful ? 'careful' : 'normal'} className="border-2 border-brand bg-brand-wash" title={`${policy}'s actual pick`} />
          {picks.map((p) => (
            <Cell key={p.rule} who={p.rule} job={p.job} sub={p.job === decision.job ? 'agrees' : undefined}
              className={p.job === decision.job ? 'border-brand' : 'border-axis'} title={`${RULE_LONG[p.rule]} would pick this from the same queue`} />
          ))}
        </div>
      )}
    </SectionCard>
  );
}

function Cell({ who, job, sub, className, title }: { who: string; job: number | null; sub?: string; className?: string; title: string }) {
  return (
    <div className={cn('rounded-lg border px-2 py-1.5 text-center min-w-0', className)} title={title}>
      <div className="text-[11px] text-ink-2 truncate">{who}</div>
      <div className="text-base font-semibold tabular-nums leading-tight">{job == null ? '–' : `#${job}`}</div>
      <div className="text-[11px] text-ink-3 h-4">{sub ?? ''}</div>
    </div>
  );
}
