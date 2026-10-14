export const fmt = (x: number | null | undefined, d = 1): string =>
  x == null || Number.isNaN(x) ? '–' : x.toFixed(d);

export const signed = (x: number, d = 1): string => (x >= 0 ? '+' : '') + x.toFixed(d);
