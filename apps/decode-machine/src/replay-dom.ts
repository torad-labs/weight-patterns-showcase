export const $ = <T extends HTMLElement = HTMLElement>(id: string) =>
  document.getElementById(id) as T;

export function setText(id: string, text: string) {
  const element = $(id);
  if (element.textContent !== text) element.textContent = text;
}

export function option(value: string, text: string) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = text;
  return o;
}

export function facts(id: string, entries: [string, string | number][]) {
  $(id).replaceChildren(
    ...entries.map(([name, value]) => {
      const d = document.createElement("div");
      d.className = "fact";
      const label = document.createElement("span"),
        v = document.createElement("strong");
      label.textContent = name;
      v.textContent = String(value);
      d.append(label, v);
      return d;
    }),
  );
}

export function paragraph(text: string, className = "small-copy") {
  const p = document.createElement("p");
  p.className = className;
  p.textContent = text;
  return p;
}
