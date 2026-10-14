import { cn } from 'cn';
import { SectionCard } from '@/components/section-card';
import { RULE_LONG, RULES, rulePick } from '@/lib/dispatch-rules';
import { clock } from '@/lib/format-clock';
import type { Replay, ReplayState } from '@/lib/replay';
import { recipes } from '@/lib/ui-recipes';

/** The active policy's pick at the latest decision, next to what each dispatch rule would have picked. */
export function ChosenNowPanel({ state, policy, replay }: { state: ReplayState; policy: string; replay: Replay }) {
  const decision = state.lastDecision;
  const visible = decision ? replay.decisionQueue(decision).slice(0, 8) : [];
  const picks = decision
    ? RULES.filter((r) => r !== policy).map((rule) => {
      const i = rulePick(rule, visible, decision.prevTray, decision.index);
      return { rule, job: i >= 0 ? visible[i].job.id : null };
    })
    : [];
  const agree = picks.filter((p) => decision && p.job === decision.job).map((p) => p.rule);
  const live = state.phase === 'pick' || state.phase === 'travel';
  const at = decision ? `decision ${decision.index + 1} at ${clock(decision.start)}, arm at tray ${decision.prevTray}, job tray ${replay.jobs[decision.job].tray}` : '';
  const note = !decision ? 'No decision yet.' : live ? `${at} · ${agree.length ? `agrees with ${agree.join(', ')}` : 'no rule picks the same job'}` : `last ${at}`;

  return (
    <SectionCard title="Chosen now" description={<span className="block min-h-[2.6em]">{note}</span>}>
      {decision && (
        <div className="flex flex-wrap gap-2 min-h-9">
          <Chip who={policy} job={decision.job} mode={decision.careful ? 'careful' : 'normal'} className={recipes.chipActive} title={`${policy}'s actual pick at ${clock(decision.start)}`} />
          {picks.map((p) => (
            <Chip key={p.rule} who={p.rule} job={p.job} className={p.job === decision.job ? recipes.chipAgree : undefined} title={`${RULE_LONG[p.rule]} would pick this from the same queue`} />
          ))}
        </div>
      )}
    </SectionCard>
  );
}

function Chip({ who, job, mode, className, title }: { who: string; job: number | null; mode?: string; className?: string; title: string }) {
  return (
    <div className={cn(recipes.chip, className)} title={title}>
      <span className="text-ink-2">{who} →</span>
      <span>{job == null ? '–' : `#${job}`}</span>
      {mode && <span className="text-[11px] text-ink-3">{mode}</span>}
    </div>
  );
}
