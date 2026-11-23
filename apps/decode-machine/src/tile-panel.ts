import { $, setText } from "./replay-dom";
import type { HardwareModel } from "./hardware-model";
import type { MechanicalWorld } from "./replay-world";
import type { TileWorld } from "./tile-world";
import {
  tileStations,
  type TileStation,
  type ToolKind,
  type Pressure,
} from "./tile-flow";

export class TilePanel {
  constructor(
    private host: MechanicalWorld,
    private model: HardwareModel,
    private world: TileWorld,
  ) {
    const section = document.createElement("section");
    section.id = "tile-controls";
    section.hidden = true;
    section.innerHTML = `
      <p class="evidence-tag">Illustrative machinery · shared playback</p>
      <label for="tile-station">Inspect equipment<select id="tile-station"></select></label>
      <p id="tile-description" class="small-copy"></p>
      <div class="equipment-actions"><button id="tile-watch" class="solid-button">Watch one tile handoff</button><button id="tile-focus" class="outline-button">Look closer</button></div>
      <p id="tile-state" class="hardware-status"></p>
      <div class="tile-slots" aria-label="Shared tray ownership"></div>
      <dl class="material-flow"><div><dt>Working crew</dt><dd><div id="tile-crew-shape" class="tile-turns" aria-hidden="true"><i></i><i></i><i></i></div><span id="tile-issued"></span></dd></div><div><dt>Inside the press</dt><dd><div id="tile-sum-shape" class="tile-segments" aria-hidden="true"><i></i><i></i><i></i><i></i></div><span id="tile-accumulation"></span></dd></div><div><dt>Receiving latch</dt><dd><div id="tile-latch-shape" class="tile-latch" aria-hidden="true"><i></i><i></i><i></i><b></b></div><span id="tile-completion"></span></dd></div></dl>
      <details class="tile-choices"><summary>Try the machinery</summary>
        <p class="small-copy">Teaching choices change this cutaway, never the recording or engine.</p>
        <label for="tile-cache">Where is the copy?<select id="tile-cache"><option value="miss">Fetch from the weight reservoir</option><option value="hit">Already in the L2 cupboard</option></select></label>
        <label for="tile-tool">Which instruction?<select id="tile-tool"><option value="matrix">Matrix multiply · tensor press</option><option value="arithmetic">Ordinary addition · lever</option><option value="compare">Maximum · selector</option><option value="exponential">Exponential · shaped cam</option></select></label>
        <label for="tile-pressure">Supply &amp; workspace<select id="tile-pressure"><option value="normal">Normal illustrative flow</option><option value="supply">Supply road occupied</option><option value="execution">Execution unit occupied</option><option value="registers">More registers per block</option><option value="shared">More shared memory per block</option></select></label>
        <p id="tile-capacity" class="small-copy"></p>
      </details>
      <details class="hardware-explanation"><summary>What is known here</summary><p id="tile-evidence" class="footnote"></p></details>`;
    $("hardware-ready").append(section);
    for (const [id, name] of tileStations) {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = name;
      $("tile-station").append(option);
    }
    for (let i = 0; i < 3; i++) {
      const e = document.createElement("div");
      e.innerHTML = `<strong>Tray ${"ABC"[i]}</strong><div id="tile-tray-${i}" class="tile-tray" aria-hidden="true"><i></i><i></i><i></i><b></b></div><span id="tile-slot-${i}"></span>`;
      section.querySelector(".tile-slots")!.append(e);
    }
    $("tile-station").addEventListener("change", () =>
      this.select($<HTMLSelectElement>("tile-station").value as TileStation),
    );
    $("tile-cache").addEventListener("change", () => {
      model.tile.cacheHit = $<HTMLSelectElement>("tile-cache").value === "hit";
      this.select("cache");
    });
    $("tile-tool").addEventListener("change", () => {
      model.tile.tool = $<HTMLSelectElement>("tile-tool").value as ToolKind;
      this.select("compute");
    });
    $("tile-pressure").addEventListener("change", () => {
      model.tile.pressure = $<HTMLSelectElement>("tile-pressure")
        .value as Pressure;
      this.select(
        model.tile.pressure === "registers" || model.tile.pressure === "shared"
          ? "capacity"
          : model.tile.pressure === "supply"
            ? "loader"
            : "scheduler",
      );
    });
    $("tile-watch").addEventListener("click", () => this.watch());
    // A closer view is an explicit spatial action, independent of time.
    $("tile-focus").onclick = () => {
      host.frameDetail(() => world.focus());
    };
    world.onSelect = (station) => this.select(station);
  }
  select(station: TileStation) {
    this.model.tile.station = station;
    this.update();
    if (innerWidth <= 650) {
      this.host.frameDetail(() => this.world.focus());
    }
  }
  watch() {
    const p = this.model.playback;
    if (p.illustrated) {
      p.inspect(p.journey.timeline.layerIndex, "qkv");
      p.seek(0);
    } else {
      const work = this.host.recorded.workers[this.host.selectedBlock].filter(
        (x) => x.step.kind !== "wait",
      );
      const next = work.find((x) => x.end > p.time) ?? work[0];
      if (!next) return;
      p.clock.seekTime(next.start);
    }
    p.playing = true;
  }
  update() {
    const tile = this.model.depth === "tile";
    $("tile-controls").hidden = !tile;
    $("hardware-basic").hidden = tile;
    $("hardware-open-tile").hidden = tile;
    $("hardware-ready").querySelector<HTMLElement>(".hardware-limit")!.hidden =
      tile;
    if (!tile) return;
    const s = this.model.tileSnapshot(this.host.selectedBlock),
      m = this.model.tile;
    const idleExample =
      this.model.playback.illustrated && this.host.selectedBlock % 8 >= 6;
    $<HTMLButtonElement>("tile-watch").disabled = idleExample;
    setText(
      "tile-watch",
      idleExample ? "Idle in this illustration" : "Watch one tile handoff",
    );
    setText("inspector-title", "From weight to working result");
    setText(
      "inspector-copy",
      `Block ${String(this.host.selectedBlock).padStart(2, "0")} · copies travel; original weights stay in storage.`,
    );
    setText("view-note", "Storage → shared trays → working crew → result");
    setText("world-note", s.evidence);
    $<HTMLSelectElement>("tile-station").value = m.station;
    $<HTMLSelectElement>("tile-cache").value = m.cacheHit ? "hit" : "miss";
    setText(
      "tile-description",
      tileStations.find((x) => x[0] === m.station)![2],
    );
    setText("tile-state", s.cause);
    $("tile-state").dataset.state =
      s.held ||
      s.waiting ||
      (s.active && !s.loading && !s.using && !s.gateOpen)
        ? "waiting"
        : s.active
          ? "working"
          : "idle";
    for (const slot of s.slots) {
      $(`tile-tray-${slot.slot}`).dataset.state = slot.state;
      setText(
        `tile-slot-${slot.slot}`,
        `${slot.tile === null ? "" : `Tile ${slot.tile + 1} · `}${slot.state}`,
      );
    }
    $("tile-crew-shape").dataset.crew = s.issued ?? "none";
    $("tile-sum-shape").hidden = m.tool !== "matrix";
    for (const [i, cell] of Array.from($("tile-sum-shape").children).entries())
      cell.classList.toggle("filled", i < s.completed);
    $("tile-latch-shape").dataset.open = String(s.gateOpen);
    for (const [i, peg] of Array.from(
      $("tile-latch-shape").querySelectorAll("i"),
    ).entries())
      peg.classList.toggle("filled", s.contributions[i]);
    setText(
      "tile-issued",
      s.issued
        ? `${s.issued[0].toUpperCase() + s.issued.slice(1)} crew`
        : "Waiting for a ready crew",
    );
    setText(
      "tile-accumulation",
      m.tool === "matrix"
        ? `${s.completed} of 4 contributions`
        : `${m.tool === "arithmetic" ? "Adding lever" : m.tool === "compare" ? "Maximum selector" : "Exponential cam"} · alternative instruction path`,
    );
    setText(
      "tile-completion",
      `${s.contributions.filter(Boolean).length} of 3 results ready · ${s.gateOpen ? "open" : "held"}`,
    );
    setText(
      "tile-capacity",
      `Teaching budget: 8 register units and 8 shared units. Each block reserves ${s.capacity.registersPerBlock} + ${s.capacity.sharedPerBlock}; ${s.resident} resident, ${s.queued} queued. Other hardware limits are omitted.`,
    );
    setText(
      "tile-evidence",
      `${s.evidence}. Slot count, cache outcome, issue choice, producer count and resource budgets are examples, not capture measurements.`,
    );
  }
}
