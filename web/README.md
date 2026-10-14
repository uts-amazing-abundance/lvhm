| `src/lib/replay.ts` | Rebuilds the cell state at any time t from a traced run (queue, arm, phase, running totals) |
| `src/lib/dispatch-rules.ts` | The six dispatch rules, for the counterfactual "chosen now" chips |
| `src/lib/cell-motion.ts` | Maps the replay state to SO-101 joint targets along the calibrated pick cycle |
| `src/lib/mujoco-cell.ts` | MuJoCo WASM + three.js renderer of the cell |
| `src/lib/schematic-cell.ts` | 2D SVG schematic with the same `CellRenderer` contract |
| `src/components/*.tsx` | React components: playback toolbar, penalty weights, queue table, chosen-now panel, score chart, cell view, status panel |
| `src/components/ui/*` | shadcn/ui primitives (Base UI): button, select, slider, table, toggle group, card |
| `src/app.tsx` | Selection and playback state |
| `src/lib/ui-recipes.ts` | Tailwind recipes for the bits shadcn does not cover |
| `src/lib/*.test.ts` | Vitest, colocated with the module under test |
# LVHM web demo

Browser front end for the work cell, following the two-panel layout of the Part B plan: the
dashboard on the left (policy, penalty weights, live queue, chosen-now comparison, cumulative
score) and the cell view on the right (SO-ARM101 in MuJoCo, shift clock, KPIs, score cards).

It replays the traced shifts that `gui/dashboard_export_cell.py` writes from the experiment
notebook, so it needs no server and no Python at run time. The rule chips under "Chosen now" are
recomputed in the browser from the queue the active policy saw (`src/rules.ts` mirrors
`lvhm/baselines.py`); the learned and solver policies show their recorded pick.

The 3D view loads the vendored SO-101 MJCF and meshes (`assets/so101`) into the official MuJoCo
WebAssembly bindings, drives the joints along the calibrated waypoints from `cell_timing.json`,
and draws the geoms with three.js. No dynamics are stepped: the trace fixes what happened and
MuJoCo only provides the kinematics. If WebGL or the model fails, the SVG schematic stays.

Stack: Vite, React 19, TypeScript, Tailwind CSS v4, shadcn/ui on Base UI, ahooks, bun. Imports use the `@/` alias and files are kebab-case per `.agents/rules/frontend.md`.

## Run

```bash
cd web
bun install
bun run dev        # http://localhost:5173
bun run typecheck
bun run test       # vitest: replay reconstruction, rule checks and arm motion against the real export
```

## Data

`public/data/job_sequences.json` is the export from the notebook (currently extracted from
`gui/job_sequence_dashboard.html`). After a new experiment run:

1. run the export cell at the end of `notebooks/experiments_phase2_1.ipynb`
2. copy `notebooks/results/<profile>/job_sequences.json` to `web/public/data/`

The SO-101 model is served from `../assets/so101` by a small Vite plugin (`/so101/...`), so the
meshes are not duplicated in this folder.

## Layout

| File | Role |
|---|---|
| `src/replay.ts` | Rebuilds the cell state at any time t from a traced run (queue, arm, phase, running totals) |
| `src/rules.ts` | The six dispatch rules, for the counterfactual "chosen now" chips |
| `src/cellMotion.ts` | Maps the replay state to SO-101 joint targets along the calibrated pick cycle |
| `src/ui/mujocoCell.ts` | MuJoCo WASM + three.js renderer of the cell |
| `src/ui/workcell.ts` | 2D SVG schematic with the same `CellView` contract |
| `src/components/*` | React components: toolbar, weights, queue, chosen-now, score chart, cell view, status |
| `src/App.tsx` | Selection and playback state |
| `src/ui/cls.ts`, `src/lib/cn.ts` | Tailwind recipes and the class composer |

## Not yet

- Live α/β sliders: the trace holds one α and β; δ switches between the traced damage conditions
- Hardware replay through LeRobot (Part B plan) is outside this app
