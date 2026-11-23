import "./replay.css";
import "./trace-panel";
import { MaterialPanel } from "./material-panel";
import { HardwarePanel } from "./hardware-panel";
import {
  ReplayClock,
  mechanisms,
  layerSteps,
  stepRole,
  stepLabel,
  activeBlockSteps,
  fmtNs,
  recordingUrls,
  type ReplayData,
  type RecordedStep,
  type MechanismId,
} from "./replay-model";
import { MechanicalWorld, type View } from "./replay-world";
import {
  equipmentState,
  mechanismForRole,
  stepForEquipment,
} from "./workshop-model";
import { ConnectedJourney, journeyPartState } from "./connected-journey";
import { PROMPT_TOKENS, QUESTION } from "./timeline";
import { ReplayPlayback } from "./replay-playback";
import { $, option, facts, paragraph, setText } from "./replay-dom";
import { SourceInspector } from "./source-inspector";
import { DiagnosticPanel } from "./diagnostic-panel";

let data: ReplayData,
  clock: ReplayClock,
  world: MechanicalWorld,
  selectedLayer = 0,
  selectedBlock = 0,
  part: MechanismId = "attention",
  selectedStep: RecordedStep | undefined;
let source: SourceInspector,
  lastBlockState = "";
let playback: ReplayPlayback, journey: ConnectedJourney;
let material: MaterialPanel;
let hardware: HardwarePanel;
let diagnostics: DiagnosticPanel | undefined;
function setFollowing(on: boolean) {
  world.followTray = on;
  if (on) {
    world.workshop.crank = null;
    world.focusedBlock = null;
  }
  const target =
    world.view === "engine"
      ? "active worker"
      : world.view === "stack"
        ? "tray"
        : "active equipment";
  $("journey-camera").setAttribute("aria-pressed", String(on));
  setText("journey-camera", `${on ? "Following" : "Follow"} ${target}`);
  $("journey-camera").title =
    world.view === "engine"
      ? "Track one of the workers operating in parallel. Drag to stop following."
      : "Keep the moving information in view. Drag to stop following.";
}
function followPlayback() {
  if (world.view === "material" || world.view === "hardware") return;
  if (!world.followTray) return;
  const focus = playback.illustrated
    ? {
        layer: journey.timeline.layerIndex,
        part: journey.operation.id,
        step: undefined,
      }
    : world.recorded.focus(clock.time);
  const changedLayer = selectedLayer !== focus.layer;
  selectedLayer = focus.layer;
  if (changedLayer) {
    world.selectedLayer = selectedLayer;
    if (world.view === "layer") world.buildLayer();
    if (world.view !== "internals") updateLayer();
  }
  if (world.view === "layer" && part !== focus.part)
    selectPart(focus.part, false);
  if (world.view === "engine") {
    const blocks =
      focus.step?.observations
        .filter((o) => o[1] <= clock.time && clock.time < o[2])
        .map((o) => o[0]) ??
      world.workers.map((_, i) => i).filter((i) => i % 8 < 6);
    if (blocks.length && !blocks.includes(selectedBlock)) {
      selectedBlock = blocks[0];
      world.selectedBlock = selectedBlock;
      $<HTMLSelectElement>("inspect-block").value = String(selectedBlock);
      lastBlockState = "";
      updateBlock();
    }
  }
  if (world.view === "internals") {
    const rebuilt = changedLayer || part !== focus.part;
    if (rebuilt) {
      part = focus.part;
      world.selectedPart = part;
      world.workshop.open();
    }
    const bench = world.workshop;
    const stages = Math.max(...bench.parts.map((p) => p.stage)) + 1,
      phase =
        journeyPartState(journey.timeline, selectedLayer, part).progress *
        stages;
    const current = playback.illustrated
      ? bench.parts.find((p) => phase >= p.stage && phase < p.stage + 1)
      : bench.parts
          .map((p) => ({
            p,
            state: equipmentState(clock.capture, selectedLayer, p, clock.time),
          }))
          .filter((r) => r.state.active)
          .sort(
            (a, b) =>
              (b.state.step?.lastStart ?? 0) - (a.state.step?.lastStart ?? 0),
          )[0]?.p;
    if (current && current.id !== bench.selected) {
      bench.selected = current.id;
      if (!rebuilt) renderEquipment();
    }
    if (rebuilt) renderWorkshop();
  }
}

function storyCopy(text: string) {
  return playback.illustrated
    ? text
        .split(/(?<=[.!?])\s+/)
        .filter(
          (sentence) =>
            !/(record|measur|entry span|marked Q tail)/i.test(sentence),
        )
        .join(" ")
    : text;
}
function stepButton(step: RecordedStep) {
  const b = document.createElement("button");
  b.textContent = stepLabel(step);
  const small = document.createElement("small");
  small.textContent = `${step.kind === "sentinel" ? "Retired marker" : step.kind} · ${fmtNs(step.end - step.start)} envelope`;
  b.append(small);
  b.addEventListener("click", () => {
    setFollowing(false);
    if (step.layer !== selectedLayer) selectLayer(step.layer);
    selectStep(step);
  });
  return b;
}
function setView(view: View) {
  const inspection = diagnostics?.active && !diagnostics.applying &&
    diagnostics.model.capture === clock.capture && diagnostics.model.step.layer === selectedLayer
      ? { record: diagnostics.model.stepId, block: diagnostics.model.block } : null;
  if (inspection) diagnostics!.applying = true;
  playback.show(view, selectedLayer);
  if (view === "engine" && world.view !== "engine") setFollowing(false);
  if (view === "material" || view === "hardware") setFollowing(false);
  $("replay-app").classList.toggle("material-view", view === "material");
  $("replay-app").classList.toggle("hardware-view", view === "hardware");
  $("journey-camera").hidden = view === "material" || view === "hardware";
  $("viewport").classList.toggle("workshop-view", view === "internals");
  $("viewport").classList.toggle("stack-view", view === "stack");
  world.setView(view);
  document
    .querySelectorAll<HTMLButtonElement>("[data-view]")
    .forEach((b) =>
      b.setAttribute(
        "aria-pressed",
        String(
          b.dataset.view ===
            (view === "internals"
              ? "layer"
              : view === "hardware"
                ? "engine"
                : view),
        ),
      ),
    );
  $("layer-controls").hidden =
    view === "engine" ||
    view === "internals" ||
    view === "material" ||
    view === "hardware";
  $("block-controls").hidden = view !== "engine";
  $("mechanism-controls").hidden = view !== "layer";
  $("record-details").hidden = view !== "layer";
  $("workshop-controls").hidden = view !== "internals";
  $("workshop-navigation").hidden = view !== "internals";
  $("back-stack").hidden = view !== "layer";
  setText(
    "inspector-title",
    view === "stack"
      ? "The layer journey"
      : view === "engine"
        ? "One persistent engine"
        : `Inside layer ${String(selectedLayer + 1).padStart(2, "0")}`,
  );
  setText(
    "inspector-copy",
    view === "stack"
      ? "Each row holds four sliding readers and one full reader. Open any layer to follow its presses, shared memory and preserved residual."
      : view === "engine"
        ? "63 reusable worker blocks. Teal presses work; violet latches wait. Open a block to inspect it."
        : `${clock.capture.layers[selectedLayer].kind === "F" ? "Full-context" : "Sliding-window"} attention, ${selectedLayer >= 15 ? "shared K/V and a wider feed-forward chamber" : "locally owned K/V"}.`,
  );
  setText(
    "view-note",
    view === "stack"
      ? "Four sliding readers, then one full reader · repeated seven times"
      : view === "engine"
        ? "63 recorded blocks · reused across the layer sequence"
        : "Select a mechanism to open its recorded substeps",
  );
  setText(
    "world-note",
    view === "engine"
      ? "Recorded worker intervals · amber copies illustrate weight use, not measured transfers."
      : view === "layer"
        ? "Trays and read copies wait during gaps in recorded work. Their routes are illustrative; values and transfers were not captured."
        : "Amber copies feed the layers and wait during gaps in recorded work. Stored weights remain; transfer timing is illustrative.",
  );
  if (view === "internals") {
    setText("inspector-title", "Inside the equipment");
    setText(
      "inspector-copy",
      "Select a part to follow what it receives, uses and produces.",
    );
    setText(
      "view-note",
      `Layer ${selectedLayer + 1} · ${mechanisms.find((m) => m.id === part)!.name}`,
    );
    setText(
      "world-note",
      "Teal paths carry information; amber drawers supply reusable weights. Select equipment to open its parts.",
    );
    renderWorkshop();
  } else if (view === "engine") {
    lastBlockState = "";
    updateBlock();
  } else updateLayer();
  if (playback.illustrated) {
    setText(
      "inspector-copy",
      view === "stack"
        ? "The question enters at the top. Amber balls stay in storage while read copies feed the moving presses. Teal trays carry the changing information to the answer printer."
        : view === "layer"
          ? "This is the same question tray, inside one layer. Watch it split through the readers, rejoin and pass through the weighted presses."
          : view === "internals"
            ? "Follow the same tray through the parts of this machine. Its weight supply and connections stay with it."
            : "The same reusable workers carry the illustrated layer work. Switch to a recorded token for measured worker intervals.",
    );
    setText(
      "world-note",
      "Teal trays carry information. Amber read copies feed the presses; stored weight balls remain.",
    );
    setText("seek-layer", "Watch this layer");
    $("record-details").hidden = true;
  } else setText("seek-layer", "Play from this layer");
  $<HTMLInputElement>("duration").value = String(playback.seconds);
  setText(
    "duration-scope",
    playback.scoped ? "This inspection" : "Whole animation",
  );
  $("layer-ticks").hidden = playback.scoped;
  $("worker-overview").hidden = view !== "engine";
  setFollowing(world.followTray);
  setText(
    "mode-note",
    playback.illustrated
      ? "Illustrated story · Gemma architecture"
      : "Measured time · symbolic machinery",
  );
  $("scrubber").setAttribute(
    "aria-label",
    playback.illustrated
      ? "Position in this journey"
      : "Position in the recorded decode",
  );
  $("previous").setAttribute(
    "aria-label",
    playback.illustrated
      ? "Previous journey step"
      : "Previous recorded operation",
  );
  $("next").setAttribute(
    "aria-label",
    playback.illustrated ? "Next journey step" : "Next recorded operation",
  );
  $("restart").setAttribute(
    "aria-label",
    playback.illustrated ? "Replay this journey" : "Replay recording",
  );
  if (playback.illustrated && view === "layer")
    setText(
      "view-note",
      "One tray, through the layer · open any machine to follow its parts",
    );
  if (playback.illustrated && view === "engine")
    setText("view-note", "63 reusable workers · illustrative movement");
  if (inspection) {
    diagnostics!.applying = false;
    diagnostics!.select(inspection.record, inspection.block);
  }
}
function updateLayer() {
  const l = clock.capture.layers[selectedLayer];
  $<HTMLSelectElement>("inspect-layer").value = String(selectedLayer);
  if (world.view === "layer") {
    setText(
      "inspector-title",
      `Inside layer ${String(selectedLayer + 1).padStart(2, "0")}`,
    );
    if (!playback.illustrated)
      setText(
        "inspector-copy",
        `${l.kind === "F" ? "Full-context" : "Sliding-window"} attention, ${l.owner === l.id ? "locally owned K/V" : "shared K/V and a wider feed-forward chamber"}.`,
      );
  }
  facts("layer-facts", [
    [
      l.kind === "F" ? "Full-context capacity" : "Sliding window",
      `${l.window.toLocaleString()} positions`,
    ],
    [
      "Query / K/V heads",
      `${clock.capture.queryHeads} / ${clock.capture.kvHeads}`,
    ],
    ["K/V owner", l.owner === l.id ? "This layer" : `Layer ${l.owner + 1}`],
    ["FFN width", l.ffnWidth.toLocaleString()],
    ["Head width", `${l.headDim} channels`],
    ["Recorded envelope", fmtNs(l.end - l.start)],
  ]);
  $("show-owner").hidden = l.owner === l.id;
  setText("show-owner", `Visit layer ${l.owner + 1}, this layer’s K/V owner`);
  const steps = layerSteps(clock.capture, selectedLayer);
  $("record").replaceChildren(
    ...steps.map((s) =>
      option(
        s.id,
        `${s.parent ? "↳ " : ""}${s.kind === "sentinel" ? "Retired · " : ""}${stepLabel(s)}`,
      ),
    ),
  );
  setText("record-count", `${steps.length} records`);
  const matching =
    selectedStep && steps.find((s) => s.role === selectedStep!.role);
  selectStep(matching ?? steps[0]);
  selectPart(part, false);
}
function selectLayer(index: number) {
  setFollowing(false);
  selectedLayer = index;
  playback.inspect(index);
  world.selectedLayer = index;
  world.buildLayer();
  setView("layer");
}
function selectPart(id: MechanismId, open = true) {
  if (open) setFollowing(false);
  part = id;
  world.selectedPart = id;
  if (open && world.view === "internals") {
    openWorkshop(id);
    return;
  }
  if (open && world.view !== "layer") setView("layer");
  const item = mechanisms.find((m) => m.id === id)!;
  $<HTMLSelectElement>("mechanism").value = id;
  setText("mechanism-title", item.name);
  setText("mechanism-copy", storyCopy(item.description));
  const s = item.role
    ? stepRole(clock.capture, selectedLayer, item.role)
    : undefined;
  setText(
    "mechanism-evidence",
    playback.illustrated
      ? "Illustrated operation · movement shows the role, not measured tensor values"
      : !s
        ? "Anatomy only · no independent recorded interval"
        : id === "ple"
          ? "Shares the down-consumer tail · no separate PLE duration"
          : "Recorded interval · mechanical motion illustrates its progress",
  );
  if (s) selectStep(s);
  else {
    $("record-metrics").replaceChildren();
    setText(
      "record-description",
      "This mechanism has no independent timestamp. Choose a recorded operation below to inspect a measured interval.",
    );
    selectedStep = undefined;
    $<HTMLSelectElement>("record").selectedIndex = -1;
    $<HTMLButtonElement>("seek-step").disabled = true;
    $("parent-step").hidden = true;
    $("child-steps").replaceChildren();
  }
  world.labels.forEach((l) => {
    if (l.part) l.element.classList.toggle("selected", l.part === part);
  });
  source?.search(item.source);
}
function selectStep(step: RecordedStep) {
  selectedStep = step;
  diagnostics?.fromMachine(step, selectedBlock);
  $<HTMLSelectElement>("record").value = step.id;
  facts("record-metrics", [
    ["Evidence", step.kind === "sentinel" ? "Retired marker" : step.kind],
    ["Participating blocks", step.observations.length],
    ["First entry → last exit", fmtNs(step.end - step.start)],
    ["Last entry → last exit", fmtNs(step.end - step.lastStart)],
  ]);
  setText("record-description", step.description);
  $("parent-step").hidden = !step.parent;
  $<HTMLButtonElement>("seek-step").disabled =
    step.kind === "sentinel" ||
    !step.observations.length ||
    step.end <= step.start;
  $("child-steps").replaceChildren(
    ...clock.capture.steps.filter((s) => s.parent === step.id).map(stepButton),
  );
  const open = $<HTMLButtonElement>("open-record-equipment");
  if (open) open.disabled = step.kind === "sentinel";
}
function openWorkshop(id: MechanismId = part) {
  setFollowing(false);
  part = id;
  world.selectedPart = id;
  playback.inspect(selectedLayer, id);
  world.workshop.crank = null;
  world.workshop.open(diagnostics?.active ? diagnostics.model.step.role : undefined);
  setView("internals");
}
function renderWorkshop() {
  const bench = world.workshop;
  $<HTMLSelectElement>("workshop-housing").value = part;
  setText(
    "workshop-location",
    [
      `Layer ${selectedLayer + 1}`,
      mechanisms.find((m) => m.id === part)!.name,
      ...bench.trail.map((p) => p.name),
    ].join(" / "),
  );
  setText(
    "workshop-back",
    bench.trail.length ? "Back to enclosing assembly" : "Back to layer",
  );
  $("workshop-equipment").replaceChildren(
    ...bench.parts.map((p, i) => option(p.id, `${i + 1}. ${p.name}`)),
  );
  renderEquipment();
}
function renderEquipment() {
  const bench = world.workshop,
    p = bench.current,
    index = bench.parts.indexOf(p);
  if (!p) return;
  $<HTMLSelectElement>("workshop-equipment").value = p.id;
  setText("equipment-count", `${index + 1} / ${bench.parts.length}`);
  $<HTMLButtonElement>("equipment-prev").disabled = index === 0;
  $<HTMLButtonElement>("equipment-next").disabled =
    index === bench.parts.length - 1;
  setText("equipment-name", p.name);
  setText("equipment-action", storyCopy(p.does));
  setText("equipment-input", p.input);
  setText("equipment-uses", p.uses);
  setText("equipment-output", p.output);
  $("equipment-open").hidden = !p.children.length;
  setText("equipment-open", `Open ${p.children.length} internal parts`);
  $<HTMLButtonElement>("equipment-seek").disabled =
    !playback.illustrated && !stepForEquipment(clock.capture, selectedLayer, p);
  setText(
    "equipment-seek",
    playback.illustrated ? "Watch this machine again" : "Seek recorded work",
  );
  renderEquipmentState();
  if (diagnostics?.active && !diagnostics.applying) {
    const recorded = stepForEquipment(clock.capture, selectedLayer, p);
    if (recorded) selectStep(recorded);
  }
}
function renderEquipmentState() {
  if (world.view !== "internals") return;
  const bench = world.workshop,
    p = bench.current;
  if (!p) return;
  if (playback.illustrated) {
    const state = journeyPartState(journey.timeline, selectedLayer, part);
    setText(
      "equipment-timing",
      state.done
        ? "This machine’s pass is complete. Continue the journey above."
        : "The same information tray · illustrated movement",
    );
    return;
  }
  const state = equipmentState(
    clock.capture,
    selectedLayer,
    p,
    clock.time,
    bench.crank,
  );
  setText(
    "equipment-timing",
    `${state.evidence}${state.step && bench.crank === null ? ` · ${state.active ? "active" : state.done ? "finished" : "not yet active"}` : ""}`,
  );
  if (part === "weights" && bench.crank === null) {
    const read = world.recorded.reads(selectedLayer, clock.time).at(-1);
    setText(
      "equipment-timing",
      read
        ? `Illustrating weight use during ${read.step!.name} · transfer timings not captured`
        : "Stored weights retained · no active recorded weight consumer",
    );
  }
  $<HTMLInputElement>("crank-mode").checked = bench.crank !== null;
  $("crank-controls").hidden = bench.crank === null;
}
function updateBlock() {
  if (playback.illustrated) {
    facts("block-state", [
      ["Selected block", String(selectedBlock).padStart(2, "0")],
      ["Movement", "Illustrated"],
    ]);
    $("block-intervals").replaceChildren(
      paragraph(
        "The same workers are reused as the tray advances. Select a recorded token above to inspect measured worker intervals.",
      ),
    );
    return;
  }
  if (world.view !== "engine") return;
  const steps = activeBlockSteps(clock.capture, selectedBlock, clock.time),
    key = selectedBlock + steps.map((s) => s.id).join();
  if (key === lastBlockState) return;
  lastBlockState = key;
  facts("block-state", [
    ["Selected block", String(selectedBlock).padStart(2, "0")],
    ["Active recorded intervals", steps.length],
  ]);
  $("block-intervals").replaceChildren(
    ...(steps.length
      ? steps.map(stepButton)
      : [
          paragraph(
            "No active interval at this point. Move the timeline to inspect another moment.",
          ),
        ]),
  );
}
function rebuildCapture() {
  $("inspect-layer").replaceChildren(
    ...clock.capture.layers.map((l) =>
      option(
        String(l.id),
        `${String(l.id + 1).padStart(2, "0")} · ${l.kind === "S" ? "Sliding" : "Full"}${l.owner !== l.id ? " · shared" : ""}`,
      ),
    ),
  );
  $("inspect-block").replaceChildren(
    ...Array.from({ length: clock.capture.grid }, (_, i) =>
      option(String(i), `Block ${String(i).padStart(2, "0")}`),
    ),
  );
  selectedBlock = Math.max(0, Math.min(selectedBlock, clock.capture.grid - 1));
  world.selectedBlock = selectedBlock;
  $<HTMLSelectElement>("inspect-block").value = String(selectedBlock);
  $("layer-ticks").replaceChildren(
    ...clock.capture.layers.map((l) => {
      const b = document.createElement("button");
      b.title = `Layer ${l.id + 1} · ${l.kind === "F" ? "Full" : "Sliding"}`;
      b.setAttribute("aria-label", `Seek to layer ${l.id + 1}`);
      b.style.left = `${playback.layerPosition(l.id) * 100}%`;
      b.className = l.kind === "F" ? "full" : "";
      b.addEventListener("click", () => {
        playback.seek(playback.layerPosition(l.id));
        selectLayer(l.id);
      });
      return b;
    }),
  );
  const c = clock.capture;
  $("evidence-content").replaceChildren(
    paragraph(
      `Position ${c.position} · ${c.steps.length} records · ${c.steps.filter((s) => s.kind === "sentinel").length} retired markers`,
      "evidence-source",
    ),
    paragraph(c.source, "evidence-source"),
    ...(c.promptText ? [paragraph(`Recorded prompt: ${c.promptText}`, "evidence-source")] : []),
    ...(c.tokenRecord ? [paragraph(`Input token ${c.tokenRecord.input} → output token ${c.tokenRecord.output}. Source: ${c.tokenRecord.source}`, "evidence-source")] : []),
    paragraph(`Tape SHA256 ${c.tapeSha}`, "source-path"),
    paragraph(`Launch tables SHA256 ${c.tablesSha}`, "source-path"),
    paragraph(
      `Source anatomy snapshot ${data.sourceRevision}. Source-only roles: 1 controller warp, 1 loader warp, 6 consumer warps; the tape does not resolve individual warps.`,
      "evidence-source",
    ),
  );
  updateLayer();
}

function updateReadout() {
  const { progress, seconds, playing } = playback;
  if (document.activeElement !== $("duration"))
    $<HTMLInputElement>("duration").value = String(seconds);
  setText(
    "duration-scope",
    playback.scoped ? "This inspection" : "Whole animation",
  );
  $("layer-ticks").hidden = playback.scoped;
  setText("viewing-time", `${(progress * seconds).toFixed(1)} / ${seconds} s`);
  $<HTMLInputElement>("scrubber").value = String(progress);
  $("scrubber").style.setProperty("--progress", `${progress * 100}%`);
  const playLabel = playing ? "Pause" : progress >= 1 ? "Replay" : "Play";
  if ($("play").getAttribute("aria-label") !== playLabel) {
    $("play").setAttribute("aria-label", playLabel);
    $("play-icon").innerHTML = playing
      ? '<path d="M8 5v14M16 5v14"/>'
      : '<path d="m8 4 11 8-11 8V4Z"/>';
  }
  const tickKey = playback.illustrated
    ? String(journey.timeline.passIndex)
    : clock.capture.id;
  if ($("layer-ticks").dataset.pass !== tickKey) {
    $("layer-ticks").dataset.pass = tickKey;
    Array.from($("layer-ticks").children).forEach(
      (node, i) =>
        ((node as HTMLElement).style.left =
          `${playback.layerPosition(i) * 100}%`),
    );
  }
  if (world.view === "engine") {
    const states = world.workerStates;
    const working = states.filter((s) => s.active && !s.wait).length,
      waiting = states.filter((s) => s.active && s.wait).length;
    setText(
      "worker-overview",
      `${working} working · ${waiting} waiting · ${states.length - working - waiting} idle${playback.illustrated ? " · illustrated" : ""}`,
    );
  }
  if (playback.illustrated) {
    const t = journey.timeline,
      scope = journey.scope,
      p = journey.progress;
    if (world.view === "stack" && t.layerActive) {
      selectedLayer = t.layerIndex;
      world.selectedLayer = selectedLayer;
      $<HTMLSelectElement>("inspect-layer").value = String(selectedLayer);
      const owner = clock.capture.layers[selectedLayer].owner;
      $("show-owner").hidden = owner === selectedLayer;
      setText("show-owner", `Visit layer ${owner + 1}, this layer’s K/V owner`);
    }
    setText(
      "phase-name",
      scope
        ? `Layer ${scope.layer + 1}${scope.part ? " · " + mechanisms.find((m) => m.id === scope.part)!.name : ""}`
        : t.stage.name,
    );
    setText(
      "phase-detail",
      t.layerActive
        ? journey.operation.verb
        : t.stage.id === "done"
          ? "The answer remains; replay starts with the question."
          : t.stage.id.startsWith("return")
            ? "The selected token goes back through the same machine."
            : t.stage.id.startsWith("choose")
              ? "The final layer feeds the selector and printer."
              : "Paper becomes tokens; tokens become information trays.",
    );
    setText(
      "recorded-time",
      scope
        ? "A closer look · same question and same journey"
        : "Question → 35 layers → answer → feedback",
    );
    setText(
      "story-location",
      scope
        ? `Inside layer ${scope.layer + 1} of 35${scope.part ? " · " + mechanisms.find((m) => m.id === scope.part)!.name : ""}`
        : t.layerActive
          ? `${t.currentToken === "question" ? "Question trays" : `“${t.currentToken}” tray`} · layer ${t.layerIndex + 1} of 35`
          : t.stage.name,
    );
    setText(
      "story-answer",
      t.answer ||
        (t.progress < 0.11
          ? "Waiting for the question"
          : "Travelling through the layers…"),
    );
    setText(
      "follow-story",
      scope ? "Return to the whole journey" : "Open the moving layer",
    );
    setText(
      "replay-status",
      `${t.playing ? "Playing" : p >= 1 ? "Pass complete" : "Paused"} · ${scope ? "inside the same journey" : "illustrated question"}`,
    );
    document
      .querySelectorAll("#story-tokens span")
      .forEach((el, i) =>
        el.classList.toggle(
          "ready",
          t.progress >= 0.065 || t.progress / 0.065 >= i / 7,
        ),
      );
    renderEquipmentState();
    return;
  }
  const active = clock.activeLayers,
    c = clock.capture,
    time = clock.time;
  const focus = world.recorded.focus(time);
  if (world.view === "stack" && !diagnostics?.active && selectedLayer !== focus.layer) {
    selectedLayer = focus.layer;
    world.selectedLayer = selectedLayer;
    updateLayer();
  }
  setText(
    "story-location",
    `Layer ${focus.layer + 1} of ${c.layers.length} · ${focus.step.name}`,
  );
  setText("follow-story", "Open the moving layer");
  setText(
    "phase-name",
    clock.progress === 1
      ? "Recording complete"
      : active.length
        ? `Layer ${String(focus.layer + 1).padStart(2, "0")}${active.length > 1 ? ` + ${active.length - 1} overlapping` : ""}`
        : "Between recorded intervals",
  );
  setText(
    "phase-detail",
    clock.progress === 1
      ? "Final token selection was not recorded"
      : active.length
        ? focus.step.name
        : "No active recorded interval",
  );
  setText(
    "recorded-time",
    `${fmtNs(time - c.start)} / ${fmtNs(c.duration)} recorded`,
  );
  setText(
    "replay-status",
    `${clock.progress === 1 ? "Complete" : clock.playing ? "Playing" : "Paused"} · position ${c.position}`,
  );
  updateBlock();
  renderEquipmentState();
}
async function start() {
  const response = await fetch(recordingUrls(location.href).recording);
  if (!response.ok)
    throw new Error("The recorded Gemma 4 data could not be loaded.");
  data = await response.json();
  if (data.schema !== "qgre.mechanical-replay.v1" || !data.captures.length)
    throw new Error("The recording has an unsupported format.");
  playback = new ReplayPlayback(data.captures[0]);
  clock = playback.clock;
  journey = playback.journey;
  world = new MechanicalWorld(
    $<HTMLCanvasElement>("replay-canvas"),
    $("world-labels"),
    clock.capture,
  );
  world.story = journey.timeline;
  $("replay-app").classList.add("story-mode");
  $("story-tokens").replaceChildren(
    ...PROMPT_TOKENS.map((token) => {
      const s = document.createElement("span");
      s.textContent = token;
      return s;
    }),
  );
  $("capture").replaceChildren(
    option("story", "France question · illustrated"),
    ...data.captures.map((c) =>
      option(c.id, `Recorded token ${c.position} · ${fmtNs(c.duration)}`),
    ),
  );
  $<HTMLSelectElement>("capture").disabled = false;
  $("mechanism").replaceChildren(
    ...mechanisms.map((m) => option(m.id, m.name)),
  );
  rebuildCapture();
  material = new MaterialPanel(playback, world, setView, setFollowing);
  setView("stack");
  $("workshop-housing").replaceChildren(
    ...mechanisms.map((m) => option(m.id, m.name)),
  );
  world.onInternal = (id) => {
    setFollowing(false);
    world.workshop.select(id);
    renderEquipment();
  };
  $("open-internals").addEventListener("click", () => openWorkshop());
  $("workshop-housing").addEventListener("change", () =>
    openWorkshop($<HTMLSelectElement>("workshop-housing").value as MechanismId),
  );
  $("workshop-equipment").addEventListener("change", () =>
    world.onInternal($<HTMLSelectElement>("workshop-equipment").value),
  );
  $("workshop-back").addEventListener("click", () => {
    if (world.workshop.trail.length) {
      world.workshop.back();
      renderWorkshop();
    } else setView("layer");
  });
  for (const [id, delta] of [
    ["equipment-prev", -1],
    ["equipment-next", 1],
  ] as const)
    $(id).addEventListener("click", () => {
      const b = world.workshop,
        next = b.parts[b.parts.indexOf(b.current) + delta];
      if (next) world.onInternal(next.id);
    });
  $("equipment-open").addEventListener("click", () => {
    setFollowing(false);
    world.workshop.enter();
    renderWorkshop();
  });
  $("equipment-focus").addEventListener("click", () => {
    setFollowing(false);
    world.workshop.focus();
  });
  $("equipment-seek").addEventListener("click", () => {
    if (playback.illustrated) {
      playback.restart();
      return;
    }
    const step = stepForEquipment(
      clock.capture,
      selectedLayer,
      world.workshop.current,
    );
    if (step) {
      world.workshop.crank = null;
      playback.playing = false;
      clock.seekTime(step.start);
    }
  });
  $("crank-mode").addEventListener("change", () => {
    setFollowing(false);
    const on = $<HTMLInputElement>("crank-mode").checked;
    world.workshop.crank = on
      ? Number($<HTMLInputElement>("crank").value)
      : null;
    if (on) clock.playing = false;
    renderEquipmentState();
  });
  $("crank").addEventListener("input", () => {
    world.workshop.crank = Number($<HTMLInputElement>("crank").value);
    clock.playing = false;
  });
  $("equipment-source").addEventListener("click", async () => {
    await source.open(world.workshop.current.source);
    $("source-details").scrollIntoView({ block: "nearest" });
  });
  const openRecord = document.createElement("button");
  openRecord.id = "open-record-equipment";
  openRecord.className = "outline-button";
  openRecord.textContent = "Open this operation’s equipment";
  $("record-details").append(openRecord);
  openRecord.addEventListener("click", () => {
    const step = selectedStep;
    if (!step || step.kind === "sentinel") return;
    openWorkshop(mechanismForRole(step.role));
    const matching = world.workshop.parts.find((p) => p.role === step.role);
    if (matching) world.onInternal(matching.id);
  });
  world.onLayer = selectLayer;
  world.onPart = (id) => {
    if (world.view === "stack" || world.view === "engine")
      selectLayer(
        playback.illustrated
          ? journey.timeline.layerIndex
          : world.recorded.layerAt(clock.time),
      );
    selectPart(id);
  };
  world.onBlock = (n) => {
    setFollowing(false);
    selectedBlock = n;
    world.selectedBlock = n;
    $<HTMLSelectElement>("inspect-block").value = String(n);
    lastBlockState = "";
    updateBlock();
    diagnostics?.inspectWorker(n);
  };
  document.querySelectorAll<HTMLButtonElement>("[data-view]").forEach((b) =>
    b.addEventListener("click", () => {
      if (b.dataset.view === "layer")
        selectLayer(
          diagnostics?.active ? diagnostics.model.step.layer : playback.illustrated
            ? journey.timeline.layerIndex
            : world.recorded.layerAt(clock.time),
        );
      else setView(b.dataset.view as View);
    }),
  );
  $("inspect-layer").addEventListener("change", () =>
    selectLayer(Number($<HTMLSelectElement>("inspect-layer").value)),
  );
  $("inspect-block").addEventListener("change", () =>
    world.onBlock(Number($<HTMLSelectElement>("inspect-block").value)),
  );
  $("mechanism").addEventListener("change", () =>
    selectPart($<HTMLSelectElement>("mechanism").value as MechanismId),
  );
  $("record").addEventListener("change", () => {
    setFollowing(false);
    selectStep(
      clock.capture.steps.find(
        (s) => s.id === $<HTMLSelectElement>("record").value,
      )!,
    );
  });
  $("parent-step").addEventListener("click", () => {
    setFollowing(false);
    const p = clock.capture.steps.find((s) => s.id === selectedStep?.parent);
    if (p) selectStep(p);
  });
  $("seek-step").addEventListener("click", () => {
    if (selectedStep) {
      clock.playing = false;
      clock.seekTime(selectedStep.start);
    }
  });
  $("seek-layer").addEventListener("click", () => {
    selectLayer(selectedLayer);
    playback.watchLayer(selectedLayer);
  });
  $("show-owner").addEventListener("click", () =>
    selectLayer(clock.capture.layers[selectedLayer].owner),
  );
  $("back-stack").addEventListener("click", () => setView("stack"));
  $("fit").addEventListener("click", () => {
    setFollowing(false);
    world.fit();
  });
  $("focus-part").addEventListener("click", () => {
    setFollowing(false);
    world.focusPart();
  });
  $("focus-block").addEventListener("click", () => {
    setFollowing(false);
    world.focusWorker();
  });
  $("capture").addEventListener("change", () => {
    const value = $<HTMLSelectElement>("capture").value;
    playback.use(
      value === "story"
        ? data.captures[0]
        : data.captures.find((c) => c.id === value)!,
      value === "story",
    );
    world.story = playback.illustrated ? journey.timeline : null;
    world.use(clock.capture);
    world.workshop.crank = null;
    selectedStep = undefined;
    selectedLayer = playback.illustrated ? journey.timeline.layerIndex : 0;
    world.selectedLayer = selectedLayer;
    setFollowing(false);
    setView("stack");
    rebuildCapture();
    lastBlockState = "";
    $("replay-app").classList.toggle("story-mode", playback.illustrated);
    setText(
      "story-question",
      playback.illustrated
        ? QUESTION
        : clock.capture.tokenRecord ? `Token ${clock.capture.tokenRecord.input} · position ${clock.capture.position}` : `Token at position ${clock.capture.position}`,
    );
    $("story-tokens").hidden = !playback.illustrated;
    setText(
      "story-kind",
      playback.illustrated
        ? "Illustrated example · Gemma architecture"
        : clock.capture.promptText ? `Recorded prompt: ${clock.capture.promptText.slice(0, 115)}${clock.capture.promptText.length > 115 ? "…" : ""}` : "Measured recording · prompt text unavailable",
    );
    setText(
      "story-answer",
      playback.illustrated
        ? journey.timeline.answer
        : clock.capture.tokenRecord ? `Next token: ${clock.capture.tokenRecord.output} · recorded` : "Next token · selection not captured",
    );
    $("restart-story").hidden = !playback.illustrated;
    setText(
      "world-title",
      playback.illustrated
        ? "A question, through Gemma."
        : "Inside the recorded Gemma decode.",
    );
    setText(
      "world-subtitle",
      playback.illustrated
        ? "Follow the information. See the weights at work."
        : "Captured work and waits on the same architecture.",
    );
  });
  const play = () => {
    world.workshop.crank = null;
    playback.toggle();
  };
  $("play").addEventListener("click", play);
  const stepPlayback = (direction: number) => {
    world.workshop.crank = null;
    playback.step(direction);
  };
  $("previous").addEventListener("click", () => stepPlayback(-1));
  $("next").addEventListener("click", () => stepPlayback(1));
  $("restart").addEventListener("click", () => {
    world.workshop.crank = null;
    playback.restart();
  });
  $("scrubber").addEventListener("input", () => {
    world.workshop.crank = null;
    playback.seek(Number($<HTMLInputElement>("scrubber").value));
  });
  $("duration").addEventListener("change", () => {
    playback.setDuration($<HTMLInputElement>("duration").valueAsNumber);
    $<HTMLInputElement>("duration").value = String(playback.seconds);
  });
  $("duration").addEventListener("input", () => {
    const value = $<HTMLInputElement>("duration").valueAsNumber;
    if (value >= 1 && value <= 3600) playback.setDuration(value);
  });
  $("follow-story").addEventListener("click", () => {
    if (playback.illustrated && journey.scope) {
      world.followTray = true;
      setView("stack");
    } else {
      const targetLayer = playback.illustrated
        ? journey.timeline.layerIndex
        : world.recorded.layerAt(clock.time);
      selectedLayer = targetLayer;
      playback.inspect(targetLayer);
      world.selectedLayer = targetLayer;
      world.buildLayer();
      world.followTray = true;
      setView("layer");
    }
    setFollowing(true);
  });
  $("journey-camera").addEventListener("click", () =>
    setFollowing(!world.followTray),
  );
  for (const event of ["pointerdown", "wheel"])
    $("replay-canvas").addEventListener(event, () => setFollowing(false), {
      passive: true,
    });
  document
    .querySelectorAll<HTMLButtonElement>("[data-journey-anchor]")
    .forEach((b) =>
      b.addEventListener("click", () => {
        setFollowing(false);
        const p =
          b.dataset.journeyAnchor === "input"
            ? [-16, 18]
            : b.dataset.journeyAnchor === "weights"
              ? [-26, 1]
              : [15, -18.5];
        const frameAnchor = () => {
          world.camera.setTarget(
            world.camera.target.clone().set(-p[0], 1, -p[1]),
          );
          world.camera.radius = 13;
          world.camera.beta = 0.6;
          world.camera.alpha = Math.PI / 2 + 0.07;
        };
        world.userMovedCamera = false;
        world.resizeFraming = frameAnchor;
        frameAnchor();
      }),
    );
  const startQuestion = () => {
    playback.restart(true);
    setView("stack");
  };
  $("restart-story").addEventListener("click", startQuestion);
  world.onEndpoint = (end) => {
    if (!playback.illustrated) return;
    if (world.view === "layer") {
      setView("stack");
      return;
    }
    if (end === "input") startQuestion();
    else {
      journey.scope = null;
      journey.timeline.seek(0.43);
      setView("stack");
    }
  };
  source = new SourceInspector(() =>
    world.view === "internals"
      ? world.workshop.current.source
      : mechanisms.find((m) => m.id === part)!.source,
  );
  hardware = new HardwarePanel(playback, world, source, setView, (n) =>
    world.onBlock(n),
  );
  diagnostics = new DiagnosticPanel(data, playback, world, (step, block) => {
    setFollowing(false);
    selectedBlock = block;
    world.selectedBlock = block;
    $<HTMLSelectElement>("inspect-block").value = String(block);
    const changed = selectedLayer !== step.layer;
    selectedLayer = step.layer;
    world.selectedLayer = step.layer;
    part = mechanismForRole(step.role);
    world.selectedPart = part;
    if (changed && world.view === "layer") world.buildLayer();
    if (world.view === "internals") { world.workshop.open(step.role); renderWorkshop(); }
    selectStep(step);
    lastBlockState = "";
    updateBlock();
  }, (view) => {
    if (view === "internals") openWorkshop(part);
    else setView(view);
  }, (id) => {
    $<HTMLSelectElement>("capture").value = id;
    $("capture").dispatchEvent(new Event("change"));
  });
  document.addEventListener("keydown", (e) => {
    if (
      e.ctrlKey ||
      e.metaKey ||
      e.altKey ||
      (e.target as HTMLElement).closest(
        "input,select,textarea,button,a,summary",
      )
    )
      return;
    if (e.code === "Space") {
      e.preventDefault();
      play();
    }
    if (e.code === "ArrowRight" || e.code === "ArrowLeft") {
      e.preventDefault();
      stepPlayback(e.code === "ArrowRight" ? 1 : -1);
    }
    if (e.code === "Home") {
      e.preventDefault();
      setView("stack");
    }
  });
  const reduced = matchMedia("(prefers-reduced-motion: reduce)");
  clock.playing = !reduced.matches;
  journey.timeline.playing = !reduced.matches;
  diagnostics.restore();
  reduced.addEventListener("change", (e) => {
    if (e.matches) {
      clock.playing = false;
      journey.timeline.playing = false;
    }
  });
  $<HTMLButtonElement>("play").disabled = false;
  $<HTMLInputElement>("scrubber").disabled = false;
  $("load-state").hidden = true;
  setFollowing(world.followTray);
  let last = performance.now();
  world.engine.runRenderLoop(() => {
    const now = performance.now(),
      dt = document.hidden ? 0 : (now - last) / 1000;
    last = now;
    playback.tick(dt);
    followPlayback();
    world.update(playback.time, dt);
    updateReadout();
    material.update();
    hardware.update();
    diagnostics?.update();
  });
  Object.assign(window, {
    gemmaMachine: {
      data,
      clock,
      world,
      journey,
      playback,
      material,
      hardware,
      diagnostics,
      selectLayer,
      selectPart,
      setView,
      selectStep,
      updateReadout,
      get illustrated() {
        return playback.illustrated;
      },
      get selectedStep() {
        return selectedStep;
      },
    },
  });
}
start().catch((error) => {
  console.error(error);
  setText(
    "load-state",
    `The recorded machine could not load. ${error instanceof Error ? error.message : String(error)} Reload this page to try again.`,
  );
});
