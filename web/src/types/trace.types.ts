/** Shape of notebooks/results/<profile>/job_sequences.json (written by gui/dashboard_export_cell.py). */

/** [arrival s, deadline s, tray, product name, nominal processing s] */
export type JobTuple = [number, number, number, string, number];
/** [job, start s, end s, travel s, careful 0|1, outcome, queue length before the pick, reward] */
export type StepTuple = [number, number, number, number, number, number, number, number];

export interface RunMetrics {
  decisions: number;
  completed: number;
  on_time: number;
  late: number;
  drops: number;
  scrapped: number;
  unfinished: number;
  careful: number;
  tardiness: number;
  travel: number;
  score: number;
}

export interface Run {
  steps: StepTuple[];
  metrics: RunMetrics;
}

export interface ShiftData {
  shift: number;
  jobs: JobTuple[];
  runs: Record<string, Run>;
}

export interface ConditionData {
  name: string;
  description: string;
  weights: [number, number, number];
  arrival_rate: number;
  shifts: ShiftData[];
}

export interface Dataset {
  profile: string;
  shift_length: number;
  eval_weights: [number, number, number];
  products: Record<string, [number, number]>;
  policies: string[];
  step_fields: string[];
  outcomes: string[];
  conditions: ConditionData[];
}
