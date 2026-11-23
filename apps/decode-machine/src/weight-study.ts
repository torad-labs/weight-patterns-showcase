export type Span = { replyId: string; start: number; end: number };
export type StudySource = { id: string; path: string; sha256: string; kind: string };
export type StudyReply = { id: string; ordinal: number; sourceId: string; requestSourceId?: string; promptTail?: string; promptTokens: number; text: string; tokenIds: number[] };
export type StoredTensor = { id: string; sourceId: string; name: string; dtype: string; shape: number[]; offsets: number[]; dataStart: number };
export type Matrix = { id: string; name: string; layer: number | null; rows: number; columns: number; tensorIds: string[]; input: string | null; output: string | null; recordId: string | null; sharedFrom?: string; blockSize: number; basis: string };
export type StudyCapture = { id: string; position: number; sourceId: string; timingId: string | null; replyId: string; inputIndex: number; outputIndex: number; inputMatches: boolean; outputMatches: boolean; inputToken: number; outputToken: number };
export type ActivationSummary = { id: string; captureId: string; sourceId: string; stage: string; layer: number; activeWidth: number; counts: number[]; zeroBlocks: number; nanBlocks: number; codeOffset: number; scaleOffset: number; codePath: string; scalePath: string; codeSha256: string; scaleSha256: string };
export type WeightStudy = { schema: "qgre.weight-investigation.v1"; id: string; runId: string; fingerprint: string; sources: StudySource[]; replies: StudyReply[]; vocabulary: Record<string, string>; captures: StudyCapture[]; matrices: Matrix[]; tensors: StoredTensor[]; activations: ActivationSummary[]; inventory: { storedTensors: number; routedMatrices: number; sharedKvMatrices?: number; unroutedMatrices: number }; provenance: { checkpointBinding: string; anatomyBinding: string }; };
export type WeightCase = { id: string; studyId: string; primary: Span; comparison: Span | null; matrixId: string; row: number; block: number; hypothesis: string; prediction: string; refutation: string; change: string; strengths: string };

export function validateSpan(span: Span, replies: { id: string; tokenIds: number[] }[]) {
  const reply = replies.find(r => r.id === span.replyId);
  if (!reply || !Number.isInteger(span.start) || !Number.isInteger(span.end) || span.start < 0 || span.end <= span.start || span.end > reply.tokenIds.length) throw Error("Select a nonempty token range inside this reply.");
  return true;
}

export function captureMatch(reply: { id: string; promptTokens: number; tokenIds: number[] }, position: number, input: number, output: number) {
  const index = position - reply.promptTokens;
  return { replyId: reply.id, inputIndex: index, outputIndex: index + 1, inputMatches: index >= 0 && reply.tokenIds[index] === input, outputMatches: index + 1 >= 0 && reply.tokenIds[index + 1] === output };
}

export function matrixRoute(name: string) {
  const match = name.match(/\.layers\.(\d+)\.(.+)$/);
  const layer = match ? Number(match[1]) : null;
  const routes: Record<string, [string, string, string]> = {
    "self_attn.q_proj": ["n1", "q_raw", "mixer:0"], "self_attn.k_proj": ["n1", "k_raw", "mixer:0"],
    "self_attn.v_proj": ["n1", "v_normed", "mixer:0"], "self_attn.o_proj": ["attn_out", "o_proj", "main:0"],
    "mlp.gate_proj": ["n2", "geglu", "gate-up:0"], "mlp.up_proj": ["n2", "geglu", "gate-up:0"],
    "mlp.down_proj": ["geglu", "mlp_down", "main:10"],
  };
  const route = match && routes[match[2]];
  return { layer, input: route ? route[0] : null, output: route ? route[1] : null, recordId: route ? `l${layer}:${route[2]}` : null };
}

export function readWeightStudy(value: unknown): WeightStudy {
  const s = value as WeightStudy;
  if (!s || s.schema !== "qgre.weight-investigation.v1" || !s.fingerprint || ![s.sources, s.replies, s.captures, s.matrices, s.tensors, s.activations].every(Array.isArray)) throw Error("Choose a weight investigation bundle.");
  const ids = new Set<string>();
  for (const table of [s.sources, s.replies, s.captures, s.matrices, s.tensors, s.activations]) for (const r of table) { if (!r.id || ids.has(r.id)) throw Error("Duplicate or missing study identity."); ids.add(r.id); }
  const sources = new Set(s.sources.map(r => r.id)), tensors = new Set(s.tensors.map(t => t.id));
  for (const row of [...s.replies, ...s.captures, ...s.tensors, ...s.activations]) if (!sources.has(row.sourceId)) throw Error("Missing evidence source.");
  for (const r of s.replies) if (r.requestSourceId && !sources.has(r.requestSourceId)) throw Error("Missing prompt source.");
  for (const r of s.replies) if (!r.tokenIds.length || r.tokenIds.some(n => !Number.isSafeInteger(n) || n < 0) || typeof r.text !== "string") throw Error("Invalid reply tokens.");
  for (const m of s.matrices) if (!Number.isSafeInteger(m.rows) || !Number.isSafeInteger(m.columns) || m.rows <= 0 || m.columns <= 0 || m.tensorIds.some(id => !tensors.has(id))) throw Error("Invalid matrix shape or storage reference.");
  for (const m of s.matrices) if (m.sharedFrom && !s.matrices.some(x => x.id === m.sharedFrom)) throw Error("Missing K/V owner matrix.");
  for (const c of s.captures) {
    const r = s.replies.find(r => r.id === c.replyId); if (!r) throw Error("Missing capture reply.");
    const check = captureMatch(r, c.position, c.inputToken, c.outputToken);
    if (Object.keys(check).some(k => check[k as keyof typeof check] !== c[k as keyof StudyCapture])) throw Error("Capture token correspondence does not match the recorded tokens.");
  }
  for (const a of s.activations) if (!s.captures.some(c => c.id === a.captureId) || a.counts.some(n => !Number.isSafeInteger(n) || n < 0) || a.activeWidth !== a.counts.reduce((x, y) => x + y, 0) * 32) throw Error("Invalid activation population or capture.");
  return s;
}

export function initialCase(s: WeightStudy): WeightCase {
  const r = s.replies[0], m = s.matrices.find(m => m.recordId && m.name.endsWith("mlp.down_proj")) ?? s.matrices[0];
  return { id: `case:${s.id}`, studyId: s.id, primary: { replyId: r.id, start: 0, end: Math.min(32, r.tokenIds.length) }, comparison: null, matrixId: m.id, row: 0, block: 0, hypothesis: "", prediction: "", refutation: "", change: "", strengths: "" };
}

export function validateCase(c: WeightCase, s: WeightStudy) {
  if (!c || c.studyId !== s.id) throw Error("This case belongs to a different study.");
  validateSpan(c.primary, s.replies); if (c.comparison) validateSpan(c.comparison, s.replies);
  const m = s.matrices.find(m => m.id === c.matrixId);
  if (!m || !Number.isInteger(c.row) || c.row < 0 || c.row >= m.rows || !Number.isInteger(c.block) || c.block < 0 || c.block >= Math.ceil(m.columns / 32)) throw Error("Weight slice is outside its matrix.");
  for (const k of ["hypothesis", "prediction", "refutation", "change", "strengths"] as const) if (typeof c[k] !== "string") throw Error("Invalid proposal text.");
  return c;
}

export function capturesInSpan(s: WeightStudy, span: Span) {
  return s.captures.filter(c => c.replyId === span.replyId && c.outputIndex >= span.start && c.outputIndex < span.end);
}

export function caseSignature(c: WeightCase) {
  return JSON.stringify([c.studyId, c.primary, c.comparison, c.matrixId, c.row, c.block, c.hypothesis, c.prediction, c.refutation, c.change, c.strengths]);
}

export function exportWeightCase(s: WeightStudy, c: WeightCase) {
  validateCase(c, s);
  const matrix = s.matrices.find(m => m.id === c.matrixId)!;
  const spans = [{ id: "span:primary", ...c.primary }, ...(c.comparison ? [{ id: "span:comparison", ...c.comparison }] : [])];
  const layers = [...new Set(s.matrices.flatMap(m => m.layer === null ? [] : [m.layer]))].map(index => ({ id: `model-layer:${index}`, index }));
  const mechanisms = [...new Set(s.matrices.flatMap(m => m.recordId ? [m.recordId] : []))].map(recordId => ({ id: `mechanism:${recordId}`, sourceRecordId: recordId, layerId: `model-layer:${Number(recordId.match(/^l(\d+):/)![1])}`, binding: "architectural operation; timings remain in the separate capture" }));
  const slices = [{ id: `slice:${c.id}`, matrixId: matrix.id, rows: [c.row, c.row + 1], columns: [c.block * 32, Math.min(matrix.columns, (c.block + 1) * 32)], basis: matrix.basis }];
  const relations: { from: string; to: string; kind: string; basis: string }[] = [];
  const link = (from: string, to: string, kind: string, basis: string) => relations.push({ from, to, kind, basis });
  for (const r of s.replies) { link(r.id, r.sourceId, "recorded_in", "recorded"); if (r.requestSourceId) link(r.id, r.requestSourceId, "responded_to", "recorded request hash"); }
  for (const t of s.tensors) link(t.id, t.sourceId, "stored_in", "recorded header");
  for (const m of s.matrices) {
    for (const t of m.tensorIds) link(m.id, t, "encoded_by", "checkpoint metadata");
    if (m.layer !== null) link(m.id, `model-layer:${m.layer}`, "belongs_to", "model hierarchy");
    if (m.recordId) link(m.id, `mechanism:${m.recordId}`, "parameter_of", "architecture; not individual access telemetry");
    if (m.sharedFrom) link(m.id, m.sharedFrom, "schedule_reuses_kv_from", "captured layer geometry; not a weight-access observation");
  }
  for (const op of mechanisms) link(op.id, op.layerId, "belongs_to", "model hierarchy");
  for (const a of s.activations) { link(a.id, a.captureId, "observed_in", "recorded"); link(a.id, a.sourceId, "reported_by", "native tap-range reader"); if (layers.some(l => l.index === a.layer)) link(a.id, `model-layer:${a.layer}`, "sampled_at", "recorded layer index"); }
  for (const cap of s.captures) { link(cap.id, cap.sourceId, "recorded_in", "recorded"); link(cap.id, cap.replyId, "compared_token_ids_with", `${cap.inputMatches && cap.outputMatches ? "two token IDs agree" : "token mismatch"}; separate execution`); }
  for (const span of spans) { link(span.id, span.replyId, "selects", "analyst selection"); for (const cap of capturesInSpan(s, span)) link(span.id, cap.id, "intersects_output_index", "derived position; token agreement remains explicit"); }
  for (const m of s.matrices) for (const a of s.activations.filter(a => a.layer === m.layer && [m.input, m.output].includes(a.stage))) link(m.id, a.id, a.stage === m.input ? "expected_input_boundary" : "expected_output_boundary", "architecture; no measured transfer or causal attribution");
  link(c.id, slices[0].id, "proposes_inspection_of", "proposed"); link(slices[0].id, matrix.id, "selects_coordinates_of", "stored coordinates"); for (const span of spans) link(c.id, span.id, "motivated_by", "analyst selection, not causal proof");
  return { schema: "qgre.weight-case.v1", studyFingerprint: s.fingerprint, case: c, story: [{ question: "What happened?", refs: spans.map(x => x.id) }, { question: "What internals were captured?", refs: spans.flatMap(span => capturesInSpan(s, span).map(x => x.id)) }, { question: "Which parameters are under investigation?", refs: [matrix.id, slices[0].id, ...matrix.tensorIds] }, { question: "What would test the explanation?", refs: [c.id] }], entities: { sources: s.sources, replies: s.replies, captures: s.captures, layers, mechanisms, matrices: s.matrices, tensors: s.tensors, activations: s.activations, spans, slices, proposals: [{ ...c, status: "proposed" }] }, relations,
    captureRequest: {
      schema: "qgre.weight-probes.v1", execution: "not-run", studyFingerprint: s.fingerprint, caseId: c.id,
      proposalSignature: caseSignature(c),
      proposal: { hypothesis: c.hypothesis, intervention: c.change, prediction: c.prediction, refutation: c.refutation, strengths: c.strengths },
      spans,
      matrix: { id: matrix.id, name: matrix.name, row: c.row, columns: [c.block * 32, Math.min(matrix.columns, (c.block + 1) * 32)], basis: matrix.basis },
      measurements: ["token-aligned residual before/after operation", "MLP down-projection input with original coordinate basis", "attention probabilities where the hypothesis requires them", "exact original and changed codes plus block scales"],
      controls: ["same prompt and sampling seed", "same engine, policy and precision", "unedited baseline", "strength sweep and norm-matched control", "independent confirmation examples"],
      missing: [...(c.hypothesis ? [] : ["hypothesis supplied by analyst"]), "generation approval after hypothesis review", "new capture data; no new inference performed"],
    } };
}
