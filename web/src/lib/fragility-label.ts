export type FragilityLabel = 'low' | 'med' | 'high';

export function fragilityLabel(f: number): FragilityLabel {
  return f < 0.3 ? 'low' : f < 0.7 ? 'med' : 'high';
}
