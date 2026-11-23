import { expect, test } from "bun:test";
import { mkdtemp, writeFile, rm, readFile } from "node:fs/promises";
import { join } from "node:path";
import { sha256, tensorHeader, readWeightBlock } from "../tools/weight-storage";
import { readWeightStudy, initialCase, exportWeightCase, type WeightStudy } from "./weight-study";

test("a weight block reader returns exact original bytes without altering the checkpoint", async () => {
  const dir = await mkdtemp("/tmp/weight-reader-test-");
  try {
    const file = join(dir, "fixture.safetensors");
    const header = { "m.indices": { dtype: "U8", shape: [2, 16], data_offsets: [0, 32] }, "m.absmax_e8m0": { dtype: "U8", shape: [2], data_offsets: [32, 34] } };
    const text = Buffer.from(JSON.stringify(header)), size = Buffer.alloc(8); size.writeBigUInt64LE(BigInt(text.length));
    await writeFile(file, Buffer.concat([size, text, Buffer.from(Array.from({ length: 34 }, (_, i) => i))]));
    const original = await readFile(file), hash = await sha256(file), h = await tensorHeader(file);
    const s: WeightStudy = { schema: "qgre.weight-investigation.v1", id: "s", runId: "r", fingerprint: "fixture", sources: [{ id: "src", path: file, sha256: hash, kind: "checkpoint" }], replies: [{ id: "reply", ordinal: 1, sourceId: "src", promptTokens: 10, text: "saved", tokenIds: [1, 2] }], vocabulary: {}, captures: [], activations: [], matrices: [{ id: "m", name: "m", layer: 0, rows: 2, columns: 32, blockSize: 32, tensorIds: ["i", "s"], input: "geglu", output: "mlp_down", recordId: "l0:main:10", basis: "fixture" }], tensors: Object.entries(header).map(([name, t], i) => ({ id: i ? "s" : "i", name, sourceId: "src", dtype: t.dtype, shape: t.shape, offsets: t.data_offsets, dataStart: h.dataStart })), inventory: { storedTensors: 2, routedMatrices: 1, unroutedMatrices: 0 }, provenance: { checkpointBinding: "fixture", anatomyBinding: "fixture" } };
    readWeightStudy(s);
    const result = await readWeightBlock(s, "m", 1, 0);
    expect(result.chunks[0].bytes).toEqual(Array.from({ length: 16 }, (_, i) => i + 16));
    expect(result.chunks[1].bytes).toEqual([33]);
    expect(await readFile(file)).toEqual(original);
    await expect(readWeightBlock(s, "m", 2, 0)).rejects.toThrow("inside");
    const exported = exportWeightCase(s, initialCase(s));
    const ids = new Set(Object.values(exported.entities).flat().map(x => x.id));
    for (const relation of exported.relations) { expect(ids.has(relation.from)).toBe(true); expect(ids.has(relation.to)).toBe(true); }
    expect(exported.captureRequest.execution).toBe("not-run");
    const drafted = { ...initialCase(s), hypothesis: "A selected column contributes to the observed repetition.", prediction: "The chosen span changes without harming the comparison." };
    const request = exportWeightCase(s, drafted).captureRequest;
    expect(request.proposal.hypothesis).toBe(drafted.hypothesis);
    expect(request.proposal.prediction).toBe(drafted.prediction);
    expect(request.missing).not.toContain("hypothesis supplied by analyst");
    await writeFile(file, Buffer.concat([original.subarray(0, original.length - 1), Buffer.from([250])]));
    await expect(readWeightBlock(s, "m", 1, 0)).rejects.toThrow("changed");
  } finally { await rm(dir, { recursive: true, force: true }); }
});
