import type { Anatomy, SourceNode } from "./replay-model";
import { recordingUrls } from "./replay-model";
import { $, paragraph } from "./replay-dom";

/** The source snapshot is a separate, lazily loaded inspector, never replay state. */
export class SourceInspector {
  private anatomy?: Anatomy;
  private pending?: Promise<void>;
  private limit = 30;
  private focus: string | null = null;

  constructor(source: () => string) {
    $("source-details").addEventListener("toggle", () => {
      if ($<HTMLDetailsElement>("source-details").open)
        void this.open(source());
    });
    $("source-search").addEventListener("input", () => {
      this.focus = null;
      this.limit = 30;
      this.render();
    });
    $("source-more").addEventListener("click", () => {
      this.limit += 30;
      this.render();
    });
  }

  private async load() {
    if (this.anatomy) return;
    if (!this.pending)
      this.pending = (async () => {
        const response = await fetch(recordingUrls(location.href).anatomy);
        if (!response.ok)
          throw new Error(
            "The source anatomy file could not be loaded. Close and reopen to retry.",
          );
        this.anatomy = await response.json();
      })().finally(() => {
        this.pending = undefined;
      });
    return this.pending;
  }

  async open(source: string) {
    try {
      await this.load();
      $<HTMLDetailsElement>("source-details").open = true;
      this.search(source);
    } catch (error) {
      $("source-status").textContent = String(error);
    }
  }

  async read(): Promise<Anatomy> {
    await this.load();
    return this.anatomy!;
  }

  search(source: string) {
    if (!$<HTMLDetailsElement>("source-details").open) return;
    $<HTMLInputElement>("source-search").value = source;
    this.focus = null;
    this.limit = 30;
    this.render();
  }

  private button(node: SourceNode) {
    const button = document.createElement("button");
    button.textContent = `L${node.level} · ${node.label}`;
    const small = document.createElement("small");
    small.textContent = node.module ?? node.kind;
    button.append(small);
    button.addEventListener("click", () => {
      this.focus = node.id;
      this.render();
    });
    return button;
  }

  private render() {
    if (!this.anatomy) return;
    const needle = $<HTMLInputElement>("source-search")
      .value.trim()
      .toLowerCase();
    const focus = this.anatomy.nodes.find((node) => node.id === this.focus);
    $("source-node").replaceChildren();
    let nodes: SourceNode[];
    if (focus) {
      const heading = document.createElement("h4");
      heading.textContent = `L${focus.level} · ${focus.label}`;
      $("source-node").append(
        heading,
        paragraph(focus.module ?? focus.kind, "source-path"),
      );
      for (const binding of focus.bindings ?? [])
        $("source-node").append(
          paragraph(
            `${binding.direction} ${binding.port} : ${binding.type}`,
            "source-port",
          ),
        );
      const back = document.createElement("button");
      back.className = "text-button";
      back.textContent = focus.parent
        ? "Up to parent part"
        : "Back to search results";
      back.addEventListener("click", () => {
        this.focus = focus.parent ?? null;
        this.render();
      });
      $("source-node").append(back);
      nodes = this.anatomy.nodes.filter((node) => node.parent === focus.id);
      if (!nodes.length)
        $("source-node").append(
          paragraph(
            "No deeper child is present in this source projection. This is not a separately timed operation.",
            "footnote",
          ),
        );
    } else
      nodes = this.anatomy.nodes.filter((node) =>
        `${node.label} ${node.module ?? ""}`.toLowerCase().includes(needle),
      );
    $("source-status").textContent =
      `${nodes.length} ${focus ? "child parts" : "matching parts"} · ${this.anatomy.nodes.length} parts in snapshot`;
    $("source-results").replaceChildren(
      ...nodes.slice(0, this.limit).map((node) => this.button(node)),
    );
    $("source-more").hidden = nodes.length <= this.limit;
  }
}
