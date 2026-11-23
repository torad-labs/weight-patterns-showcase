export type RunTurn = { id: string; sourceId: string; ordinal: number; promptTokens: number; cachedTokens: number; generatedTokens: number; cumulativeTokens: number; hostNs: number; prefillNs?: number; prefillRows?: number; gpuNs: number | null; decodeTokens: number | null; finish: string; request: string; response: string };
export type RunMetric = { id: string; sourceId: string; launchId: string; name: string; unit: string; mean: number; min: number; max: number; samples: number };
type UnavailableCapture = { id: string; sourceId: string; position: number; label: string; reason: string };
export type RunEvidence = { schema: "qgre.run-evidence.v1"; id: string; label: string; targetTokens: number; generatedTokens: number; sources: { id: string; method: string; artifact: string; binarySha256: string | null; warnings: string[] }[]; turns: RunTurn[]; launches: { id: string; sourceId: string; name: string; startNs: number; endNs: number; registers: number; grid: number; threads: number; sharedBytes: number }[]; metrics: RunMetric[]; captures: { id: string; sourceId?: string; position: number; durationNs: number; profiling: string; slidingTailNs: number; fullTailNs: number }[]; unavailableCaptures?: UnavailableCapture[]; counterPasses?: { id: string; sourceId: string; timelineBinding: null; name: string; metrics: { name: string; unit: string; value: string }[] }[]; gaps: { area: string; missing: string }[]; observations: { claim: string; evidenceIds: string[]; next: string }[] };

export function readRunEvidence(value: unknown): RunEvidence {
  const r = value as RunEvidence;
  if (!r || r.schema !== "qgre.run-evidence.v1" || !Array.isArray(r.turns) || !Array.isArray(r.sources) || !Array.isArray(r.launches) || !Array.isArray(r.metrics) || !Array.isArray(r.captures) || !Array.isArray(r.gaps) || !Array.isArray(r.observations)) throw Error("Unsupported run evidence.");
  const count = (n: number) => Number.isSafeInteger(n) && n >= 0;
  if (!count(r.generatedTokens) || !count(r.targetTokens)) throw Error("Invalid token count.");
  let cumulative = 0;
  for (const t of r.turns) {
    if (![t.promptTokens, t.cachedTokens, t.generatedTokens, t.cumulativeTokens, t.hostNs].every(count) || t.cachedTokens > t.promptTokens) throw Error("Invalid turn population.");
    cumulative += t.generatedTokens;
    if (t.cumulativeTokens !== cumulative) throw Error("The conversation token counts do not reconcile.");
    if (t.gpuNs !== null && (!count(t.gpuNs) || !t.decodeTokens || !count(t.decodeTokens))) throw Error("Decode time needs its actual token denominator.");
  }
  if (cumulative !== r.generatedTokens) throw Error("The run total does not match its turns.");
  const ids = new Set(r.sources.map(s => s.id)), launches = new Map(r.launches.map(l => [l.id, l]));
  for (const t of r.turns) if (!ids.has(t.sourceId)) throw Error("Missing conversation source.");
  for (const l of r.launches) if (!ids.has(l.sourceId) || l.endNs < l.startNs) throw Error("Invalid launch source or interval.");
  for (const m of r.metrics) if (!ids.has(m.sourceId) || launches.get(m.launchId)?.sourceId !== m.sourceId || ![m.mean, m.min, m.max].every(Number.isFinite) || m.min > m.mean || m.mean > m.max) throw Error("Counter and launch must belong to the same pass.");
  for (const c of r.counterPasses ?? []) if (!ids.has(c.sourceId) || c.timelineBinding !== null) throw Error("An independent counter pass needs its source and cannot claim a timeline binding.");
  for (const c of r.captures) if (c.sourceId && !ids.has(c.sourceId)) throw Error("Missing diagnostic source.");
  return r;
}

export const turnMeanMs = (t: RunTurn) => t.gpuNs !== null && t.decodeTokens ? t.gpuNs / t.decodeTokens / 1e6 : null;
