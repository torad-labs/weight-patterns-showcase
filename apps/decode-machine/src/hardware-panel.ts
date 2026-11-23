import type { ReplayPlayback, View } from "./replay-playback";
import type { MechanicalWorld } from "./replay-world";
import type { SourceInspector } from "./source-inspector";
import { HardwareModel } from "./hardware-model";
import { HardwareWorld } from "./hardware-world";
import { $, setText } from "./replay-dom";
import { TilePanel } from "./tile-panel";
import { tileSnapshotFor, type TileStation } from "./tile-flow";

export class HardwarePanel {
  model?: HardwareModel;
  world?: HardwareWorld;
  private pending?: Promise<void>;
  private blockCount = 0;
  private tilePanel?: TilePanel;
  private returnView: View = "engine";
  constructor(
    public playback: ReplayPlayback,
    public host: MechanicalWorld,
    private source: SourceInspector,
    private show: (view: View) => void,
    private selectBlock: (block: number) => void,
  ) {
    const section = document.createElement("section");
    section.id = "hardware-controls";
    section.hidden = true;
    section.innerHTML = `
      <nav class="hardware-navigation" aria-label="Hardware inspection depth"><button id="hardware-floor" class="text-button">Worker floor</button><button id="hardware-sm" class="text-button" hidden>SM bay</button></nav>
      <p id="hardware-loading" role="status"></p><button id="hardware-retry" class="outline-button" hidden>Retry source anatomy</button>
      <div id="hardware-ready" hidden>
        <p class="hardware-limit"><span aria-hidden="true">○</span> Representative SM · internal timing illustrative</p>
        <button id="hardware-open-tile" class="solid-button">Follow a tile through the machinery</button>
        <div id="hardware-basic">
        <div class="equipment-actions supply-actions"><button id="sm-l2" class="outline-button">Open shared L2</button><button id="sm-tma" class="outline-button">Open TMA crane</button></div>
        <label for="hardware-block">Worker block<select id="hardware-block"></select></label>
        <p id="hardware-status" class="hardware-status"></p>
        <p id="hardware-operation" class="small-copy"></p>
        <label for="hardware-warp">Warp group<select id="hardware-warp"></select></label>
        <div class="equipment-actions"><button id="hardware-expand" class="solid-button">Open 32 lanes</button><button id="hardware-watch" class="outline-button">Watch this block</button></div>
        <p id="hardware-role" class="small-copy"></p>
        <div id="hardware-lanes" hidden><h3>32 lanes in this warp</h3><div id="hardware-lane-grid" aria-label="Select a lane"></div><p id="hardware-lane-reading" class="small-copy"></p></div>
        <details class="hardware-explanation"><summary>What the machinery represents</summary><p class="small-copy">The SM is a physical bay with shared execution hardware. A block is a group of work using that bay; its warps use the same hardware. Each warp has 32 logical lanes, not 32 dedicated cores. The staging rack belongs to the block; each lane has private register storage.</p></details>
        <details class="hardware-explanation"><summary>What is known here</summary><p id="hardware-evidence" class="small-copy"></p></details>
        </div>
      </div>`;
    $("replay-inspector").insertBefore(section, $("source-details"));
    $("open-hardware").addEventListener("click", () => void this.open());
    host.supplies.poseForBlock = (block) =>
      tileSnapshotFor(playback, host.supplies.flow, host.recorded, block);
    host.onSupply = (station, origin) => {
      host.nextInspectionOrigin = origin;
      void this.open("tile", station);
    };
    for (const id of ["floor-l2", "sm-l2"])
      $(id).onclick = () => void this.open("tile", "cache");
    for (const id of ["floor-tma", "sm-tma"])
      $(id).onclick = () => void this.open("tile", "loader");
    $("floor-cache").onchange = () => {
      host.supplies.flow.cacheHit =
        $<HTMLSelectElement>("floor-cache").value === "hit";
      this.update();
    };
    for (const id of ["open-tile", "equipment-tile", "material-tile"])
      $(id).addEventListener("click", () => void this.open("tile"));
    $("hardware-open-tile").addEventListener("click", () =>
      this.changeDepth("tile"),
    );
    $("hardware-retry").addEventListener("click", () => void this.open());
    $("hardware-floor").addEventListener("click", () => {
      this.host.nextInspectionDirection = "out";
      this.show(this.returnView);
    });
    $("hardware-sm").addEventListener("click", () => {
      if (this.model) {
        this.changeDepth("sm");
      }
    });
    $("hardware-block").addEventListener("change", () => {
      this.selectBlock(Number($<HTMLSelectElement>("hardware-block").value));
      this.update();
    });
    $("hardware-warp").addEventListener("change", () =>
      this.chooseWarp(
        Number($<HTMLSelectElement>("hardware-warp").value),
        false,
      ),
    );
    $("hardware-expand").addEventListener("click", () =>
      this.chooseWarp(this.model!.selectedWarp, true),
    );
    $("hardware-watch").addEventListener("click", () => this.watch());
    for (let lane = 0; lane < 32; lane++) {
      const button = document.createElement("button");
      button.textContent = String(lane);
      button.setAttribute("aria-label", `Inspect lane ${lane}`);
      button.dataset.lane = String(lane);
      button.addEventListener("click", () => this.chooseLane(lane));
      $("hardware-lane-grid").append(button);
    }
    host.onHardwareWarp = (warp) => this.chooseWarp(warp, true);
    host.onHardwareLane = (warp, lane) => {
      this.chooseWarp(warp, true);
      this.chooseLane(lane);
    };
  }
  async open(depth: "sm" | "tile" = "sm", station?: TileStation) {
    if (this.host.view !== "hardware") this.returnView = this.host.view;
    if (station) this.host.supplies.flow.station = station;
    if (this.world) {
      if (this.host.view === "hardware") this.changeDepth(depth);
      else {
        this.model!.depth = depth;
        this.host.nextInspectionDirection = "in";
        this.show("hardware");
        this.world.update();
        this.host.fit();
        this.update();
      }
      if (station) {
        this.update();
        this.host.frameDetail(
          () => this.world!.tile.focus(),
          !this.host.inspection.active,
        );
      }
      return;
    }
    if (this.pending) return this.pending;
    setText("hardware-loading", "Opening the source anatomy…");
    $("hardware-retry").hidden = true;
    this.pending = (async () => {
      try {
        const anatomy = await this.source.read();
        this.model = new HardwareModel(
          this.playback,
          anatomy,
          this.host.supplies.flow,
        );
        this.world = new HardwareWorld(this.host, this.model);
        this.host.hardware = this.world;
        this.tilePanel = new TilePanel(this.host, this.model, this.world.tile);
        this.model.depth = depth;
        this.world.onWarp = (warp) => this.chooseWarp(warp, true);
        this.world.onLane = (lane) => this.chooseLane(lane);
        $("hardware-warp").replaceChildren(
          ...this.model.groups.map((warp, i) => {
            const option = document.createElement("option");
            option.value = String(i);
            option.textContent = warp.name + " · 32 lanes";
            return option;
          }),
        );
        setText("hardware-loading", "");
        $("hardware-ready").hidden = false;
        if (this.host.view === this.returnView) {
          this.host.nextInspectionDirection = "in";
          this.show("hardware");
        }
        this.update();
        if (this.host.view === "hardware") {
          this.world.update();
          this.host.fit();
          if (station)
            this.host.frameDetail(() => this.world!.tile.focus(), false);
        }
      } catch (error) {
        if (this.host.view === this.returnView) this.show("hardware");
        setText("hardware-loading", String(error));
        $("hardware-retry").hidden = false;
      }
    })().finally(() => {
      this.pending = undefined;
    });
    return this.pending;
  }
  chooseWarp(warp: number, expand: boolean) {
    if (!this.model) return;
    const opening = expand && this.model.depth !== "warp";
    if (opening)
      this.host.beginInspection(
        "in",
        this.world!.groups[warp]?.root.getAbsolutePosition(),
      );
    this.model.selectedWarp = Math.max(
      0,
      Math.min(this.model.groups.length - 1, warp),
    );
    if (expand) this.model.depth = "warp";
    this.update();
    this.world!.update();
    if (expand) this.host.fit();
    if (opening) this.host.inspection.finish();
  }
  changeDepth(depth: "sm" | "warp" | "tile") {
    if (!this.model || this.model.depth === depth) return;
    this.host.inspectChange(
      depth === "sm" ? "out" : "in",
      () => {
        this.model!.depth = depth;
        this.world!.update();
        this.update();
      },
      this.host.nextInspectionOrigin,
    );
    this.host.nextInspectionOrigin = undefined;
    if (depth === "tile" && innerWidth <= 650) this.world!.tile.focus();
  }
  chooseLane(lane: number) {
    if (this.model) {
      this.model.selectedLane = lane;
      this.update();
      this.world!.update();
    }
  }
  watch() {
    if (!this.model) return;
    const p = this.playback,
      block = this.host.selectedBlock;
    if (p.illustrated) {
      p.inspect(p.journey.timeline.layerIndex);
      p.seek(0.15);
    } else {
      const work = this.host.recorded.workers[block].filter(
        (x) => x.step.kind !== "wait",
      );
      const next = work.find((x) => x.end > p.time) ?? work[0];
      if (!next) return;
      p.clock.seekTime(next.start);
    }
    p.playing = true;
  }
  update() {
    $<HTMLSelectElement>("floor-cache").value = this.host.supplies.flow.cacheHit
      ? "hit"
      : "miss";
    const visible = this.host.view === "hardware";
    $("hardware-controls").hidden = !visible;
    if (!visible) return;
    setText("inspector-title", "Inside an SM");
    setText(
      "inspector-copy",
      "A representative hardware bay around the selected worker. The recording does not locate this block on a physical SM.",
    );
    setText("view-note", "GPU → SM → block → warp → 32 lanes");
    setText(
      "world-note",
      "SM = streaming multiprocessor · warps group 32 logical lanes · motion is illustrative",
    );
    if (!this.model) return;
    if (this.blockCount !== this.playback.clock.capture.grid) {
      this.blockCount = this.playback.clock.capture.grid;
      $("hardware-block").replaceChildren(
        ...Array.from({ length: this.blockCount }, (_, i) => {
          const option = document.createElement("option");
          option.value = String(i);
          option.textContent = "Block " + String(i).padStart(2, "0");
          return option;
        }),
      );
    }
    const m = this.model,
      s = m.snapshot(this.host.selectedBlock),
      expanded = m.depth === "warp";
    $<HTMLSelectElement>("hardware-block").value = String(s.block);
    $<HTMLSelectElement>("hardware-warp").value = String(m.selectedWarp);
    setText(
      "inspector-title",
      expanded
        ? `${s.warp.name} · 32 lanes`
        : `SM bay · block ${String(s.block).padStart(2, "0")}`,
    );
    setText(
      "inspector-copy",
      expanded
        ? "32 logical lanes share an instruction group and the SM’s execution hardware."
        : `${m.groups.length} warp groups share this block’s workspace inside a representative SM.`,
    );
    setText(
      "hardware-status",
      `${s.status} · ${s.illustrated ? "illustrated block" : "recorded block interval"}`,
    );
    $("hardware-status").dataset.state = s.status.toLowerCase();
    setText(
      "hardware-operation",
      s.layer === null
        ? "No layer interval yet"
        : `Layer ${s.layer + 1} · ${s.activity.step?.name ?? (s.activity.active ? "serving the same question" : "no active work")}`,
    );
    setText(
      "hardware-role",
      s.warp.role === "controller"
        ? "Controller: coordinates the block’s next work and readiness."
        : s.warp.role === "loader"
          ? "Loader: arranges operand delivery to the block’s staging space."
          : "Consumer: uses staged operands and shared execution hardware to produce results.",
    );
    setText(
      "hardware-evidence",
      `${m.anatomy.binding} Roles come from its policy, not per-warp telemetry. Physical SM identity, occupancy, warp timing, active-lane mask, register values and role-to-thread binding are unknown. The finger strokes explain grouping; they are not measured lane execution. Source revision ${m.anatomy.revision.slice(0, 12)}.`,
    );
    $("hardware-lanes").hidden = !expanded;
    $("hardware-lane-grid").dataset.unknown = String(!s.illustrated);
    $("hardware-sm").hidden = m.depth === "sm";
    setText(
      "hardware-floor",
      this.returnView === "engine"
        ? "Worker floor"
        : this.returnView === "internals"
          ? "Back to equipment"
          : this.returnView === "layer"
            ? "Back to layer"
            : this.returnView === "material"
              ? "Back to supplies"
              : "Layer journey",
    );
    $("hardware-expand").hidden = expanded;
    $<HTMLButtonElement>("hardware-watch").disabled =
      s.illustrated && s.block % 8 >= 6;
    setText(
      "hardware-watch",
      s.illustrated && s.block % 8 >= 6
        ? "Idle in this illustration"
        : "Watch this block",
    );
    setText(
      "hardware-lane-reading",
      `Lane ${s.lane} selected. Register values and lane activity are unknown. Ink marks your selection in the machine and grid.`,
    );
    for (const button of $(
      "hardware-lane-grid",
    ).querySelectorAll<HTMLButtonElement>("button"))
      button.setAttribute(
        "aria-pressed",
        String(Number(button.dataset.lane) === s.lane),
      );
    this.tilePanel?.update();
  }
}
