import type { Plugin } from "vite";
import { readFile, realpath } from "node:fs/promises";
import { resolve, sep } from "node:path";
import { readWeightStudy } from "../src/weight-study.ts";
import { readWeightBlock } from "./weight-storage.ts";

/** Local, bounded, read-only access. No inference or mutation route exists. */
export function weightApi(): Plugin {
  return { name: "weight-evidence-read-only", configureServer(server) {
    server.middlewares.use("/__weight-block", async (req, res) => {
      res.setHeader("Content-Type", "application/json"); res.setHeader("Cache-Control", "no-store");
      try {
        if (req.method !== "GET") throw Error("Only read requests are supported.");
        if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}`) throw Error("Use the local workbench origin.");
        const u = new URL(req.url!, "http://localhost"), rel = u.searchParams.get("study") ?? "";
        if (!rel.startsWith("/evidence/") || !rel.endsWith("/weight-study.json")) throw Error("Choose a prepared local study.");
        const publicRoot = await realpath(server.config.publicDir), p = await realpath(resolve(publicRoot, `.${rel}`));
        if (!p.startsWith(publicRoot + sep)) throw Error("Study path is outside the workbench.");
        const s = readWeightStudy(JSON.parse(await readFile(p, "utf8")));
        const models = await realpath(resolve(server.config.root, "../../models"));
        // Historical records retain their original paths. Read their relocated bytes
        // from this research store, while keeping the same containment and hash checks.
        for (const source of s.sources.filter(x => x.kind === "checkpoint")) {
          const relModel = source.path.match(/(?:^|\/)models\/(.+)$/);
          if (relModel && !source.path.startsWith(models + sep)) source.path = resolve(models, relModel[1]);
          if (!(await realpath(source.path)).startsWith(models + sep)) throw Error("Checkpoint is outside this project's models.");
        }
        const result = await readWeightBlock(s, u.searchParams.get("matrix") ?? "", Number(u.searchParams.get("row")), Number(u.searchParams.get("block")));
        res.end(JSON.stringify(result));
      } catch (e) { res.statusCode = 400; res.end(JSON.stringify({ error: e instanceof Error ? e.message : String(e) })); }
    });
  } };
}
