import { readFile, writeFile, mkdir, copyFile } from "node:fs/promises";
import { resolve, dirname, basename } from "node:path";
import { createHash } from "node:crypto";
import { tensorHeader, sha256 } from "./weight-storage";
import { captureMatch, matrixRoute, readWeightStudy, type WeightStudy } from "../src/weight-study";

// Offline only: reads saved outputs and checkpoint metadata; never imports the inference engine.
const runRoot = resolve(process.argv[2]), destination = resolve(process.argv[3]);
const json = async (path: string) => JSON.parse(await readFile(path, "utf8"));
const run = await json(resolve(runRoot, "run-evidence.json"));
const replay = await json(resolve(destination, "gemma4-e2b.json"));
await mkdir(resolve(destination, "activation-planes"), { recursive: true });
const study: WeightStudy = { schema: "qgre.weight-investigation.v1", id: `weights:${run.id}`, runId: run.id, fingerprint: "pending", sources: [], replies: [], vocabulary: {}, captures: [], matrices: [], tensors: [], activations: [], inventory: { storedTensors: 0, routedMatrices: 0, unroutedMatrices: 0 }, provenance: { checkpointBinding: "Checkpoint files rehashed against this run's recorded hashes.", anatomyBinding: "Expected boundary relationships; source snapshot is not verified as the saved binary." } };
const source = async (path: string, kind: string, expected?: string) => {
  path = resolve(path); const known = study.sources.find(x => x.path === path); if (known) return known;
  const hash = await sha256(path); if (expected && hash !== expected) throw Error(`Recorded artifact changed: ${path}`);
  const s = { id: `source:${study.sources.length}`, path, sha256: hash, kind }; study.sources.push(s); return s;
};
await source(resolve(destination, "gemma4-e2b.json"), "replay-index");
for (const item of run.configuration.checkpoint) {
  const p = resolve(item.path); if (!p.endsWith(".safetensors")) continue;
  const s = await source(p, "checkpoint", item.sha256), { header, dataStart } = await tensorHeader(p);
  const metadata = header.__metadata__ ?? {};
  for (const [name, value] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const v = value as { dtype: string; shape: number[]; data_offsets: number[] };
    study.tensors.push({ id: `tensor:${study.tensors.length}`, sourceId: s.id, name, dtype: v.dtype, shape: v.shape, offsets: v.data_offsets, dataStart });
  }
  for (const key of Object.keys(metadata).filter(k => k.endsWith(".in_dim"))) {
    const name = key.slice(0, -7), route = matrixRoute(name);
    const owner = route.layer === null ? null : replay.captures[0].layers.find((x: { id: number }) => x.id === route.layer)?.owner;
    const shared = owner !== null && owner !== undefined && owner !== route.layer && /self_attn\.[kv]_proj$/.test(name);
    study.matrices.push({ id: `matrix:${name}`, name, ...route, ...(shared ? { sharedFrom: `matrix:${name.replace(/\.layers\.\d+\./, `.layers.${owner}.`)}`, input: null, output: null, recordId: null } : {}), rows: Number(metadata[`${name}.out_dim`]), columns: Number(metadata[key]), blockSize: Number(metadata[`${name}.blocksize`]), tensorIds: study.tensors.filter(t => t.sourceId === s.id && t.name.startsWith(`${name}.`)).map(t => t.id), basis: `Stored ${metadata[`${name}.indices_semantic`]} coordinates; WHT seed ${metadata[`${name}.wht_seed`]}. Relation to unrotated neuron axes needs the runtime transform.` });
  }
}
study.matrices.sort((a, b) => (a.layer ?? -1) - (b.layer ?? -1) || a.name.localeCompare(b.name));
const checkpoint = study.sources.find(s => s.kind === "checkpoint")!;
const tokenizerPath = resolve(dirname(checkpoint.path), "tokenizer.json"), tokenizerSource = await source(tokenizerPath, "tokenizer");
const tokenizer = await json(tokenizerPath);
for (const t of run.turns) {
  const s = await source(t.response, "production-response"), reply = await json(t.response), choice = reply.choices[0];
  const requestSource = await source(t.request, "production-request"), request = await json(t.request);
  if (choice.token_ids.length !== t.generatedTokens) throw Error("Response token count does not match the saved run.");
  study.replies.push({ id: `reply:${t.ordinal}`, ordinal: t.ordinal, sourceId: s.id, requestSourceId: requestSource.id, promptTail: String(request.prompt ?? "").slice(-800), promptTokens: t.promptTokens, tokenIds: choice.token_ids, text: choice.text });
}
const used = new Set(study.replies.flatMap(r => r.tokenIds));
for (const [piece, id] of Object.entries(tokenizer.model.vocab)) if (used.has(Number(id))) study.vocabulary[String(id)] = piece;
for (const t of tokenizer.added_tokens) if (used.has(t.id)) study.vocabulary[String(t.id)] = t.content;
// Vocabulary pieces are displayed verbatim; no approximate token-to-character offsets.
void tokenizerSource;
for (const name of ["first-turn-tapes", "later-context/tapes"]) {
  const folder = resolve(runRoot, name);
  let steps: { position: number; token_in: number; token_out: number }[];
  try { steps = (await readFile(resolve(folder, "steps.jsonl"), "utf8")).trim().split("\n").map(x => JSON.parse(x)); } catch (e) { if ((e as NodeJS.ErrnoException).code === "ENOENT") continue; throw e; }
  const ss = await source(resolve(folder, "steps.jsonl"), "diagnostic-steps");
  const ordinal = name === "first-turn-tapes" ? 1 : Number((await readFile(resolve(runRoot, "later-context/source-request.txt"), "utf8")).match(/turn-(\d+)/)?.[1]);
  const reply = study.replies.find(r => r.ordinal === ordinal)!;
  for (const step of steps) {
    const rpath = resolve(runRoot, `range-${step.position}.json`);
    let range; try { range = await json(rpath); } catch (e) { if ((e as NodeJS.ErrnoException).code === "ENOENT") continue; throw e; }
    const s = await source(rpath, "admitted-activation-summary");
    const id = `capture:${name}:${step.position}`, timing = replay.captures.find((c: { position: number }) => c.position === step.position);
    study.captures.push({ id, position: step.position, sourceId: ss.id, timingId: timing?.id ?? null, inputToken: step.token_in, outputToken: step.token_out, ...captureMatch(reply, step.position, step.token_in, step.token_out) });
    const paths = [range.codes_plane, range.scale_plane].map((p: string) => resolve(p));
    const files = await Promise.all(paths.map(p => source(p, "encoded-activation-plane")));
    const exported = paths.map(p => `activation-planes/${name === "first-turn-tapes" ? "early" : "later"}-${basename(p)}`);
    for (let i = 0; i < paths.length; i++) await copyFile(paths[i], resolve(destination, exported[i]));
    for (const stage of range.stages) for (const row of stage.rows) {
      study.activations.push({ id: `activation:${id}:${stage.stage}:${row.layer}`, captureId: id, sourceId: s.id, stage: stage.stage, layer: Number(row.layer), activeWidth: Number(row.active_blocks) * 32, counts: [Number(row.zero_scale_active_blocks), ...row.exponent_histogram.map(Number)], zeroBlocks: Number(row.zero_scale_active_blocks), nanBlocks: Number(row.nan_blocks), codeOffset: Number(stage.base_elem) + Number(row.layer) * Number(stage.layer_stride_elems), scaleOffset: (Number(stage.base_elem) + Number(row.layer) * Number(stage.layer_stride_elems)) / 32, codePath: exported[0], scalePath: exported[1], codeSha256: files[0].sha256, scaleSha256: files[1].sha256 });
    }
  }
}
study.inventory = { storedTensors: study.tensors.length, routedMatrices: study.matrices.filter(m => m.recordId).length, sharedKvMatrices: study.matrices.filter(m => m.sharedFrom).length, unroutedMatrices: study.matrices.filter(m => !m.recordId && !m.sharedFrom).length };
study.fingerprint = createHash("sha256").update(JSON.stringify({ sources: study.sources, matrices: study.matrices, captures: study.captures })).digest("hex");
readWeightStudy(study);
await writeFile(resolve(destination, "weight-study.json"), JSON.stringify(study));
console.log(JSON.stringify({ replies: study.replies.length, tokens: study.replies.reduce((s, r) => s + r.tokenIds.length, 0), captures: study.captures.length, activations: study.activations.length, ...study.inventory, output: resolve(destination, "weight-study.json") }));
