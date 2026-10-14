import { useMount, useUnmount } from 'ahooks';
import { cn } from 'cn';
import { useEffect, useRef, useState } from 'react';
import { SectionCard } from '@/components/section-card';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';
import type { MujocoCell } from '@/lib/mujoco-cell';
import type { ReplayState } from '@/lib/replay';
import { SchematicCell, type CellRenderer } from '@/lib/schematic-cell';
import { recipes } from '@/lib/ui-recipes';

type Mode = '3d' | '2d';

/**
 * The SO-ARM101 cell: MuJoCo WASM + three.js when it loads (the Part B plan's default), the
 * SVG schematic meanwhile and whenever the 3D view is unavailable. Both renderers are
 * imperative and share the CellRenderer contract, so React only owns the toggle and the handoff.
 */
export function CellView({ state, chosenJob }: { state: ReplayState; chosenJob: number | null }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const divRef = useRef<HTMLDivElement>(null);
  const schematic = useRef<SchematicCell | null>(null);
  const mujoco = useRef<MujocoCell | null>(null);
  const [mode, setMode] = useState<Mode>('3d');
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState<string | null>('loading MuJoCo…');

  useMount(() => { if (svgRef.current && !schematic.current) schematic.current = new SchematicCell(svgRef.current); });

  // One load per mounted view, shared across effect re-runs (StrictMode mounts twice in dev).
  const loading = useRef<Promise<MujocoCell> | null>(null);
  useEffect(() => {
    if (mode !== '3d' || mujoco.current || !divRef.current) return;
    let active = true;
    const container = divRef.current;
    loading.current ??= import('@/lib/mujoco-cell').then(({ MujocoCell }) =>
      MujocoCell.create(container, `${import.meta.env.BASE_URL}so101/`, (msg) => { if (active) setStatus(`${msg}…`); }));
    loading.current.then(
      (cell) => { mujoco.current = cell; if (active) { setStatus(null); setReady(true); } },
      (err: unknown) => {
        loading.current = null;
        if (!active) return;
        setStatus(`MuJoCo view unavailable (${err instanceof Error ? err.message : String(err)}); showing the schematic.`);
        setMode('2d');
      },
    );
    return () => { active = false; };
  }, [mode]);

  useUnmount(() => {
    // Dispose only a finished renderer. A load still in flight is left to finish so a StrictMode
    // remount (dev) can adopt it instead of fetching the 26 MB model twice.
    if (mujoco.current) { mujoco.current.dispose(); mujoco.current = null; loading.current = null; }
  });

  const show3d = mode === '3d' && ready;
  useEffect(() => {
    const view: CellRenderer | null = show3d ? mujoco.current : schematic.current;
    view?.update(state, chosenJob);
  }, [state, chosenJob, show3d]);

  const offTray = state.queue.filter((e) => e.pocket == null).length;
  const note = (state.phase === 'travel' ? `swinging to tray ${state.armTo}`
    : state.phase === 'pick' ? `${state.current?.careful ? 'careful' : 'normal'} pick of #${state.current?.job}`
    : state.phase === 'waiting' ? 'idle, waiting for arrivals' : 'shift over')
    + (offTray ? ` · ${offTray} more waiting off-tray (every pocket is taken)` : '');

  return (
    <SectionCard
      title="SO-ARM101 — cell view"
      action={
        <ToggleGroup value={[mode]} onValueChange={(v) => { if (v[0]) setMode(v[0] as Mode); }} variant="outline" size="sm" spacing={0} aria-label="Cell view mode">
          <ToggleGroupItem value="3d">MuJoCo 3D</ToggleGroupItem>
          <ToggleGroupItem value="2d">Schematic</ToggleGroupItem>
        </ToggleGroup>
      }
    >
      <div className="relative">
        <svg ref={svgRef} viewBox="0 0 480 320" className={cn('w-full h-auto aspect-[480/320]', show3d && 'hidden')} role="img" aria-label="Schematic of the arm, the two input trays and the waiting jobs" />
        <div ref={divRef} className={cn('w-full aspect-[3/2] rounded-lg overflow-hidden bg-tray', !show3d && 'hidden')} aria-label="MuJoCo rendering of the SO-ARM101 cell" />
        {status && mode === '3d' && <p className="absolute left-2 bottom-2 m-0 text-[12px] text-ink-2 bg-surface/85 rounded px-2 py-0.5" aria-live="polite">{status}</p>}
        {status && mode === '2d' && <p className={cn(recipes.note, 'mt-1')} aria-live="polite">{status}</p>}
      </div>
      <p className={cn(recipes.note, 'mt-2')}>{note}</p>
    </SectionCard>
  );
}
