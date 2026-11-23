import { open, stat } from "node:fs/promises";
import { createReadStream } from "node:fs";
import { createHash } from "node:crypto";
import type { WeightStudy } from "../src/weight-study.ts";

export async function sha256(path: string) {
  const h = createHash("sha256"); for await (const chunk of createReadStream(path)) h.update(chunk); return h.digest("hex");
}

export async function tensorHeader(path: string) {
  const f = await open(path, "r");
  try {
    const size = (await f.stat()).size, length = Buffer.alloc(8);
    if ((await f.read(length, 0, 8, 0)).bytesRead !== 8) throw Error("Truncated checkpoint header.");
    const n = Number(length.readBigUInt64LE());
    if (!Number.isSafeInteger(n) || n < 2 || n > 32 * 1024 * 1024 || n + 8 > size) throw Error("Invalid checkpoint header size.");
    const data = Buffer.alloc(n); if ((await f.read(data, 0, n, 8)).bytesRead !== n) throw Error("Truncated checkpoint header.");
    const header = JSON.parse(data.toString());
    for (const [key, t] of Object.entries(header)) {
      if (key === "__metadata__") continue;
      const v = t as { data_offsets: number[]; shape: number[]; dtype: string };
      if (!Array.isArray(v.shape) || v.shape.some(x => !Number.isSafeInteger(x) || x < 0) || !Array.isArray(v.data_offsets) || v.data_offsets.length !== 2 || v.data_offsets.some(x => !Number.isSafeInteger(x) || x < 0) || v.data_offsets[0] > v.data_offsets[1] || v.data_offsets[1] + n + 8 > size) throw Error(`Invalid tensor bounds: ${key}`);
    }
    return { header, dataStart: n + 8 };
  } finally { await f.close(); }
}

const verified = new Map<string, string>();
export async function readWeightBlock(s: WeightStudy, matrixId: string, row: number, block: number) {
  const m = s.matrices.find(m => m.id === matrixId);
  if (!m || !Number.isInteger(row) || row < 0 || row >= m.rows || !Number.isInteger(block) || block < 0 || (block + 1) * 32 > m.columns || m.blockSize !== 32) throw Error("Select a complete 32-weight block inside this matrix.");
  const planes = s.tensors.filter(t => m.tensorIds.includes(t.id));
  const indices = planes.find(t => t.name.endsWith(".indices"));
  if (!indices || indices.dtype !== "U8" || indices.shape[0] !== m.rows || indices.shape[1] !== m.columns / 2) throw Error("This matrix has no supported packed E2M1 storage.");
  const source = s.sources.find(x => x.id === indices.sourceId)!;
  if (source.kind !== "checkpoint" || planes.some(p => p.sourceId !== source.id)) throw Error("All matrix planes must belong to the same checkpoint source.");
  const before = await stat(source.path), key = `${source.path}:${before.size}:${before.mtimeMs}:${before.ino}`;
  if (verified.get(key) !== source.sha256) {
    if (await sha256(source.path) !== source.sha256) throw Error("Checkpoint changed since this study was prepared.");
    verified.set(key, source.sha256);
  }
  const h = await tensorHeader(source.path);
  const f = await open(source.path, "r");
  try {
    const chunks = [];
    for (const t of planes) {
      const actual = h.header[t.name];
      if (!actual || JSON.stringify(actual.data_offsets) !== JSON.stringify(t.offsets) || JSON.stringify(actual.shape) !== JSON.stringify(t.shape) || actual.dtype !== t.dtype || h.dataStart !== t.dataStart) throw Error("Tensor header does not match this study.");
      const blocks = m.rows * m.columns / 32, ordinal = row * m.columns / 32 + block;
      const length = t.offsets[1] - t.offsets[0];
      const bytesPerBlock = t.name.endsWith(".indices") ? 16 : length / blocks;
      if (!Number.isInteger(bytesPerBlock) || bytesPerBlock < 1 || bytesPerBlock > 16) continue;
      const offset = t.dataStart + t.offsets[0] + ordinal * bytesPerBlock;
      if (offset + bytesPerBlock > t.dataStart + t.offsets[1]) throw Error("Weight block crosses storage bounds.");
      const bytes = Buffer.alloc(bytesPerBlock); if ((await f.read(bytes, 0, bytesPerBlock, offset)).bytesRead !== bytesPerBlock) throw Error("Truncated weight block.");
      chunks.push({ tensorId: t.id, name: t.name, fileOffset: offset, bytes: [...bytes] });
    }
    const after = await f.stat();
    if (after.mtimeMs !== before.mtimeMs || after.size !== before.size || after.ino !== before.ino) throw Error("Checkpoint changed during the read.");
    return { matrixId, row, columns: [block * 32, (block + 1) * 32], checkpointSha256: source.sha256, basis: m.basis, chunks };
  } finally { await f.close(); }
}
