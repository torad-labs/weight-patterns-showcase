import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { capturesInSpan, readWeightStudy, validateCase } from "../src/weight-study.ts";
import { readWeightBlock, sha256 } from "./weight-storage.ts";

// Extracts saved observations only. It has no model runner or checkpoint writer.
const [studyFile, caseFile, outputFile] = process.argv.slice(2);
if (!studyFile || !caseFile || !outputFile) throw Error("Usage: bun tools/extract-weight-probe.ts <weight-study.json> <case.json> <output.json>");
const s = readWeightStudy(JSON.parse(await readFile(studyFile, "utf8")));
const exported = JSON.parse(await readFile(caseFile, "utf8"));
if (exported.schema !== "qgre.weight-case.v1" || exported.studyFingerprint !== s.fingerprint) throw Error("Case and recorded study do not match.");
const c = validateCase(exported.case, s), matrix = s.matrices.find(m => m.id === c.matrixId)!;
const spans = [c.primary, ...(c.comparison ? [c.comparison] : [])];
const captures = [...new Set(spans.flatMap(span => capturesInSpan(s, span).map(x => x.id)))];
const evidence = s.activations.filter(a => captures.includes(a.captureId) && a.layer === matrix.layer && [matrix.input, matrix.output, "post_attn", "layer_out"].includes(a.stage));
const planes = new Map<string, Buffer>();
for (const a of evidence) for (const [path, hash] of [[a.codePath, a.codeSha256], [a.scalePath, a.scaleSha256]]) {
  if (planes.has(path)) continue;
  const file = resolve(dirname(studyFile), path);
  if (await sha256(file) !== hash) throw Error(`Recorded plane changed: ${path}`);
  planes.set(path, await readFile(file));
}
const observations = evidence.map(a => ({ id: a.id, captureId: a.captureId, stage: a.stage, layer: a.layer, axes: ["stored element"], shape: [a.activeWidth], representation: "e4m3 codes with one ue8m0 scale byte per 32 elements", coordinateBinding: "Runtime tap boundary; no unrotated parameter-axis equivalence asserted", codes: [...planes.get(a.codePath)!.subarray(a.codeOffset, a.codeOffset + a.activeWidth)], scales: [...planes.get(a.scalePath)!.subarray(a.scaleOffset, a.scaleOffset + a.activeWidth / 32)], source: { codeSha256: a.codeSha256, scaleSha256: a.scaleSha256, codeOffset: a.codeOffset, scaleOffset: a.scaleOffset } }));
const weights = await readWeightBlock(s, matrix.id, c.row, c.block);
const result = { schema: "qgre.weight-probe.v1", studyFingerprint: s.fingerprint, caseId: c.id, execution: "existing-artifacts-only", spans, captures: s.captures.filter(x => captures.includes(x.id)), matrix, weights, observations, missing: ["token log-probabilities", "attention probabilities", "verified unrotated parameter-to-activation coordinate binding", ...(observations.length ? [] : ["internal capture for selected spans"])], relations: observations.map(a => ({ from: a.id, to: a.captureId, kind: "recorded_in" })) };
await mkdir(dirname(resolve(outputFile)), { recursive: true });
if ([studyFile, caseFile, ...s.sources.map(x => x.path)].some(p => resolve(p) === resolve(outputFile))) throw Error("Output cannot replace evidence inputs.");
await writeFile(outputFile, JSON.stringify(result, null, 2), { flag: "wx" });
console.log(JSON.stringify({ output: resolve(outputFile), observations: observations.length, captures: captures.length, execution: result.execution }));
