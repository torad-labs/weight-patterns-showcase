import type { MechanicalWorld } from "./replay-world";
import type { ReplayPlayback, View } from "./replay-playback";
import {
  MaterialModel,
  materialTopics,
  contextLabels,
  storyPhase,
  type MaterialTopic,
} from "./material-model";
import { MaterialWorld } from "./material-world";
import { $, setText } from "./replay-dom";

const explanation: Record<MaterialTopic, string> = {
  cart: "A token keeps its identity while its channels change. The open cartridge compares incoming pins with their current values. Twelve channels illustrate a much larger hidden state.",
  context:
    "K and V stay on their shelves. Attention reads copies from eligible positions. Future positions remain out of reach; each request owns separate context.",
  split:
    "One output can be divided into smaller dot products. Six two-channel trays take parallel lanes. The collecting beam waits for every partial result before releasing their sum.",
  supply:
    "Stored weights remain in the reservoir. A loader fills staging slots; consumers read the tiles, then release those slots for reuse. The three slots and delivery schedule are illustrative.",
  worker:
    "The controller announces work, the loader supplies operands, and six consumer presses compute. This source cutaway explains roles within a reusable worker; it is not a map of physical GPU cores.",
  wait: "The stop arm protects a dependency: this example cannot release an output before every partial arrives. A recorded wait is observable, but its underlying cause may remain unknown.",
  precision:
    "A value lands on a representable notch. The signed code and a shared scale describe the stored value. The difference shown here is rounding error; it is not automatically a stored correction.",
  choice:
    "The final hidden state passes sizing rollers and a vocabulary press. Four representative candidates receive authored scores. This example chooses the highest score and feeds the selected token back.",
  requests:
    "Two independent copies of the France question can overlap on shared machinery and weights. A has one identity notch; B has two. Their context shelves and answers remain separate. This is an authored overlap, not a measured batching schedule.",
};

export class MaterialPanel {
  model: MaterialModel;
  world: MaterialWorld;
  private key = "";
  constructor(
    public playback: ReplayPlayback,
    public host: MechanicalWorld,
    private show: (view: View) => void,
    private follow: (on: boolean) => void,
  ) {
    this.model = new MaterialModel(playback);
    this.world = new MaterialWorld(host, this.model);
    host.material = this.world;
    const section = document.createElement("section");
    section.id = "material-controls";
    section.hidden = true;
    section.innerHTML = `
      <label for="material-topic">Explore<select id="material-topic"></select></label>
      <h3 id="material-title"></h3>
      <p id="material-evidence" class="evidence-tag"></p>
      <div class="equipment-actions"><button id="material-watch" class="solid-button">Watch this process</button><button id="material-focus" class="outline-button">Look closer</button></div>
      <details class="material-explanation"><summary>What happens here</summary><p id="material-copy" class="small-copy"></p></details>
      <div id="material-live" class="facts"></div>
      <div class="material-options">
        <label id="material-token-field" for="material-token">Prompt position<select id="material-token"></select></label>
        <label id="material-window-field"><input id="material-window" type="checkbox"> Compare a 3-position teaching window</label>
        <label id="material-batch-field"><input id="material-batch" type="checkbox"> Share the machinery with a second request</label>
        <label id="material-request-field" for="material-request">Inspect request<select id="material-request"><option value="0">A · one notch</option><option value="1">B · two notches</option></select></label>
      </div>
      <div id="material-content"></div>
      <button id="material-tile" class="outline-button">Follow a tile on the hardware</button>
      <button id="material-return" class="text-button">Return to the layer journey</button>
    `;
    $("replay-inspector").insertBefore(section, $("source-details"));
    for (const [id, name] of materialTopics) {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = name;
      $("material-topic").append(option);
    }
    for (const [i, token] of contextLabels.slice(0, 7).entries()) {
      const option = document.createElement("option");
      option.value = String(i);
      option.textContent = `${i + 1} · ${token}`;
      $("material-token").append(option);
    }
    $<HTMLSelectElement>("material-token").value = "6";
    $("material-topic").addEventListener("change", () =>
      this.open($<HTMLSelectElement>("material-topic").value as MaterialTopic),
    );
    $("material-token").addEventListener("change", () => {
      this.model.promptToken = Number(
        $<HTMLSelectElement>("material-token").value,
      );
      this.key = "";
      this.update();
    });
    $("material-window").addEventListener("change", () => {
      this.model.shortWindow = $<HTMLInputElement>("material-window").checked;
      this.key = "";
      this.update();
    });
    $("material-batch").addEventListener("change", () => {
      this.model.multiple = $<HTMLInputElement>("material-batch").checked;
      if (!this.model.multiple) this.model.request = 0;
      this.key = "";
      this.update();
    });
    $("material-request").addEventListener("change", () => {
      this.model.request = Number(
        $<HTMLSelectElement>("material-request").value,
      );
      this.key = "";
      this.update();
    });
    $("material-focus").addEventListener("click", () => {
      this.follow(false);
      this.world.focus();
    });
    $("material-return").addEventListener("click", () => this.show("stack"));
    $("material-watch").addEventListener("click", () => this.watch());
    this.world.onSelect = (topic) => this.open(topic);
    host.onMaterial = (topic) => this.open(topic as MaterialTopic);
  }
  open(topic: MaterialTopic = this.model.topic) {
    this.model.topic = topic;
    this.follow(false);
    if (this.host.view !== "material") this.show("material");
    $<HTMLSelectElement>("material-topic").value = topic;
    this.key = "";
    this.update();
    this.world.focus();
  }
  watch() {
    if (!this.playback.illustrated) return;
    const topic = this.model.topic;
    if (topic === "requests" || topic === "choice") {
      this.playback.journey.continue();
      this.playback.seek(topic === "choice" ? 0.43 : 0.11);
    } else {
      // Explicit Watch operates on the primary journey; selection alone never seeks.
      this.model.request = 0;
      $<HTMLSelectElement>("material-request").value = "0";
      this.playback.inspect(
        this.world.state.layer,
        topic === "context"
          ? "attention"
          : topic === "split" || topic === "wait"
            ? "down"
            : "qkv",
      );
      this.playback.seek(0);
    }
    this.playback.playing = true;
  }
  update() {
    const visible = this.host.view === "material";
    $("material-controls").hidden = !visible;
    const s = (this.world.state = this.model.snapshot()),
      topic = this.model.topic;
    setText(
      "phase-boundary",
      this.playback.illustrated
        ? `${s.requests.length === 2 ? "Request A · " : ""}${storyPhase(this.playback.journey.timeline)}${storyPhase(this.playback.journey.timeline) === "Prefill" ? " · read the prompt" : storyPhase(this.playback.journey.timeline) === "Decode" ? " · predict the next token" : ""}`
        : "Decode · one recorded token position",
    );
    $("request-companion").hidden = s.requests.length !== 2;
    if (s.requests.length === 2) {
      const second = s.requests[1];
      setText(
        "request-companion",
        `B · ${second.phase} · layer ${second.layer + 1}${innerWidth > 650 ? " · " + (second.answer || "answer pending") : ""}`,
      );
    }
    $("story-answer").dataset.empty = String(
      this.playback.illustrated && !this.playback.journey.timeline.answer,
    );
    if (!visible) return;
    setText(
      "inspector-title",
      `Layer ${s.layer + 1}${s.requests.length === 2 ? " · Request " + (s.request ? "B" : "A") : ""}`,
    );
    setText("inspector-copy", `${s.phase} · ${s.operation}`);
    setText(
      "view-note",
      "Same cart and same clock · open a connected mechanism",
    );
    setText(
      "material-return",
      topic === "requests"
        ? "See the requests in the layer journey"
        : "Return to the layer journey",
    );
    setText("material-title", materialTopics.find((x) => x[0] === topic)![1]);
    setText("material-copy", explanation[topic]);
    setText(
      "material-evidence",
      s.illustrated
        ? "Illustrative values, scheduling and movement"
        : s.timing + " · missing values stay unknown",
    );
    $("material-token-field").hidden =
      !s.illustrated ||
      !["cart", "context"].includes(topic) ||
      s.phase !== "Prefill";
    $("material-window-field").hidden =
      topic !== "context" || !s.illustrated || s.kind !== "S";
    $("material-batch-field").hidden = topic !== "requests" || !s.illustrated;
    $("material-request-field").hidden = s.requests.length !== 2;
    $("material-watch").hidden = !s.illustrated;
    setText(
      "material-watch",
      topic === "requests"
        ? "Watch both requests"
        : topic === "choice"
          ? "Watch token selection"
          : s.requests.length === 2
            ? "Watch request A"
            : "Watch this process",
    );
    const live = s.illustrated
      ? `${s.phase === "Prefill" ? "Question token" : "Token"} ${(s.token ?? 0) + 1}: “${contextLabels[s.token ?? 0]}”`
      : s.identity;
    setText("material-live", live);
    const key = [
      topic,
      s.illustrated,
      s.layer,
      s.request,
      s.token,
      s.requests.length,
      s.teachingWindow,
    ].join(":");
    if (key !== this.key) {
      this.key = key;
      const content = $("material-content");
      if (topic === "cart")
        content.innerHTML = s.channels
          ? `<table class="channel-table"><caption>12 illustrative channels · bars share ±<span id="channel-range"></span></caption><thead><tr><th>Channel</th><th>In</th><th>Now</th></tr></thead><tbody>${s.channels.input.map((_, i) => `<tr><th>${i + 1}</th>${["in", "now"].map((side) => `<td><span class="channel-reading"><i class="channel-bar" aria-hidden="true"><b id="bar-${side}-${i}"></b></i><span id="pin-${side}-${i}"></span></span></td>`).join("")}</tr>`).join("")}</tbody></table>`
          : '<p class="unknown-reading">Channel values were not captured. Equal pins mark unknown values.</p>';
      if (topic === "context")
        content.innerHTML = s.entries
          ? `<p class="small-copy">K/V owner: layer ${s.owner + 1}. ${s.kind === "F" ? "Full attention" : s.teachingWindow ? "Three-position teaching comparison; the model window is " + s.window : s.window + "-position model window"}.</p><div class="context-slots">${s.entries.map((e) => `<span id="context-slot-${e.index}"><small>${e.index + 1}</small>${e.label}</span>`).join("")}</div><p id="context-readout" class="small-copy"></p><p class="footnote">Teal: stored and eligible. Green: stored but outside this query’s reach. Pale: not stored yet. Short prompts may fit entirely inside the model’s sliding window.</p>`
          : '<p class="unknown-reading">K/V contents and exact read positions were not captured. The unfilled shelf is an unknown, not an empty cache.</p>';
      if (topic === "split" || topic === "wait")
        content.innerHTML = `<ol class="partial-list">${Array.from({ length: 6 }, (_, i) => `<li><span>Partial ${i + 1}</span><output id="partial-${i}"></output></li>`).join("")}</ol><p id="partial-result" class="small-copy"></p><p id="dependency-state" class="evidence-tag"></p><p class="footnote">Six lanes explain splitting one illustrative output. They are not six tokens or measured child-instruction timings.</p>`;
      if (topic === "supply")
        content.innerHTML = `<ol class="partial-list">${[0, 1, 2].map((i) => `<li><span>Staging slot ${i + 1}</span><output id="staging-${i}"></output></li>`).join("")}</ol><p class="small-copy">The reservoir is retained. Slots are reusable; releasing one never deletes the model weight.</p>${s.illustrated ? "" : '<p class="unknown-reading">Transfer timing, traffic, cache hits and ring occupancy were not recorded. The loader stays parked.</p>'}`;
      if (topic === "worker")
        content.innerHTML = `<dl class="material-flow"><div><dt>Controller</dt><dd>Announces the work and required readiness</dd></div><div><dt>Loader</dt><dd>Moves operand tiles into staging slots</dd></div><div><dt>Six consumers</dt><dd>Read operands and produce partial results</dd></div></dl><p class="footnote">Roles follow the separately labeled source snapshot. Their individual execution is not measured in these recordings.</p>`;
      if (topic === "precision")
        content.innerHTML = s.rounding
          ? '<dl class="material-flow"><div><dt>Incoming value</dt><dd id="round-input"></dd></div><div><dt>Sign / level / scale</dt><dd id="round-code"></dd></div><div><dt>Stored value</dt><dd id="round-stored"></dd></div><div><dt>Rounding difference</dt><dd id="round-error"></dd></div></dl><p class="footnote">An E2M1 notch example with an authored scale. This is not an engine tensor or a simulation of all its arithmetic. The residual’s separate hi/lo holders remain in the residual equipment.</p>'
          : '<p class="unknown-reading">Codes, scales and numerical rounding effects were not captured. The gauge has no measured reading.</p>';
      if (topic === "choice")
        content.innerHTML = s.candidates
          ? `<ol class="candidate-list">${s.candidates.map((_, i) => `<li><span id="candidate-name-${i}"></span><meter id="candidate-score-${i}" min="0" max="4"></meter><output id="candidate-value-${i}"></output></li>`).join("")}</ol><p id="choice-state" class="evidence-tag"></p><p class="footnote">Four illustrative candidates, not the whole vocabulary. Highest score is selected for this authored story; no measured logits or sampling are claimed.</p>`
          : '<p class="unknown-reading">Final normalization, vocabulary scores and token selection were not captured. The readout remains unmeasured.</p>';
      if (topic === "requests")
        content.innerHTML = s.illustrated
          ? `<div class="request-list">${s.requests.map((r, i) => `<section><h4>Request ${r.id} · ${i ? "two notches" : "one notch"}</h4><p>What’s the capital of France?</p><p id="request-state-${i}"></p><output id="request-answer-${i}"></output></section>`).join("")}</div><p class="footnote">Shared weights and machinery; independent context and outputs. Arrival offset and overlap are authored, not a throughput comparison.</p>`
          : '<p class="unknown-reading">This replay contains one token position. Switch Replay to the France illustration to compare independent requests.</p>';
    }
    const number = (v: number) => v.toFixed(3);
    if (topic === "cart" && s.channels) {
      const range = Math.max(
        0.125,
        ...s.channels.input.map(Math.abs),
        ...s.channels.current.map(Math.abs),
      );
      setText("channel-range", number(range));
      s.channels.input.forEach((v, i) => {
        setText(`pin-in-${i}`, number(v));
        setText(`pin-now-${i}`, number(s.channels!.current[i]));
        for (const [side, value] of [
          ["in", v],
          ["now", s.channels!.current[i]],
        ] as const) {
          const bar = $(`bar-${side}-${i}`),
            width = (Math.abs(value) / range) * 50;
          bar.style.width = `${width}%`;
          bar.style.left = `${value < 0 ? 50 - width : 50}%`;
        }
      });
    }
    if (topic === "context" && s.entries) {
      s.entries.forEach((e) => {
        const el = $(`context-slot-${e.index}`);
        el.classList.toggle("stored", e.stored);
        el.classList.toggle("eligible", e.eligible);
      });
      setText(
        "context-readout",
        `${s.cacheCount} stored positions · ${s.entries.filter((e) => e.eligible).length} eligible for ${s.identity}. Reading sends copies; storage remains.`,
      );
    }
    if (topic === "split" || topic === "wait") {
      s.branches.forEach((p, i) =>
        setText(
          `partial-${i}`,
          s.illustrated
            ? p === 1
              ? `Ready · ${number(s.channels!.partials[i])}`
              : "In flight"
            : "Internal completion unmeasured",
        ),
      );
      setText(
        "partial-result",
        s.illustrated
          ? `Collected sum: ${s.released ? number(s.channels!.result) : s.branches.every((p) => p === 1) ? "collecting the six contributions" : "waiting for all contributions"}`
          : "No measured partial values",
      );
      setText("dependency-state", s.waitReason);
    }
    if (topic === "supply")
      for (let i = 0; i < 3; i++)
        setText(`staging-${i}`, s.slots?.[i] ?? "Unknown");
    if (topic === "precision" && s.rounding) {
      setText("round-input", number(s.rounding.value));
      setText(
        "round-code",
        `${s.rounding.value < 0 ? "−" : "+"} / ${s.rounding.levels[s.rounding.code]} / × ${s.rounding.scale}`,
      );
      setText("round-stored", number(s.rounding.stored));
      setText("round-error", number(s.rounding.difference));
    }
    if (topic === "choice" && s.candidates) {
      s.candidates.forEach((candidate, i) => {
        setText(`candidate-name-${i}`, candidate.label);
        setText(
          `candidate-value-${i}`,
          s.choiceStage || s.returning ? candidate.score.toFixed(1) : "Pending",
        );
        $<HTMLMeterElement>(`candidate-score-${i}`).value =
          s.choiceStage || s.returning ? candidate.score + 0.3 : 0;
      });
      setText(
        "choice-state",
        s.selectedCandidate
          ? `${s.selectedCandidate === "End" ? "Stop signal" : "Selected: " + s.selectedCandidate} · answer ${s.answer || "not printed yet"}`
          : "Waiting for the final layer’s output",
      );
    }
    if (topic === "requests" && s.illustrated)
      s.requests.forEach((r, i) => {
        setText(
          `request-state-${i}`,
          `${r.phase} · layer ${r.layer + 1} · ${r.cache} retained context positions`,
        );
        setText(`request-answer-${i}`, r.answer || "Answer pending");
      });
  }
}
