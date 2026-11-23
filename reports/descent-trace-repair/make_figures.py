"""Render the manuscript figures. Use --refresh-history to rebuild the archived continuation."""
import argparse
import hashlib
import json
import math
import re
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
RETRO = ROOT / "records/retrospectives"
DEV = ROOT / "records/development"
STUDY = ROOT / "studies/04-round-2-overlap"
TEAL, RED, INK, GRAY, RULE = "#08665f", "#a03f29", "#192f37", "#4a6065", "#d3dedb"


def read(path):
    return json.loads(path.read_text())


class Canvas:
    def __init__(self, title, height, description=""):
        desc = f"<desc>{escape(description)}</desc>" if description else ""
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="800" height="{height}" viewBox="0 0 800 {height}"><title>{escape(title)}</title>{desc}<g font-family="DejaVu Sans" fill="{INK}">']

    def line(self, x1, y1, x2, y2, color=RULE, width=1, dash=False):
        stroke = ' stroke-dasharray="5 5"' if dash else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{stroke}/>')

    def txt(self, x, y, value, size=15, anchor="start", color=INK, bold=False):
        weight = ' font-weight="bold"' if bold else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}"{weight}>{escape(str(value))}</text>')

    def mark(self, x, y, color, diamond=False, radius=4.4):
        if diamond:
            self.parts.append(f'<path d="M{x} {y-radius} l{radius} {radius} l{-radius} {radius} l{-radius} {-radius} Z" fill="{color}" stroke="white" stroke-width="0.6"/>')
        else:
            self.parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}" stroke="white" stroke-width="0.6"/>')

    def save(self, name):
        (FIG / name).write_text("".join(self.parts) + "</g></svg>\n")


def refresh_history():
    path = HERE / "figure-1-continuation.json"
    snapshot = read(path)
    loop = ROOT / snapshot["source"]
    offset = read(loop.parent / "declaration.json")["counter_semantics"]["historical_predecessor"]
    final = read(loop.parent / "final-state.json")
    member = "discovery-05-channel-h1"
    cases = read(STUDY / "inputs/acquisition/discovery-cases.json")
    expected = next(case["expected"] for case in cases if case["id"] == member)
    expected_fields = (expected["availableJ"], expected["requiredJ"], expected["answer"])
    ending = re.compile(r"Available energy = (\d+) J\nRequired energy = (\d+) J\nVerdict = (YES|NO)$")
    rows = []
    for local in range(snapshot["first_local_generation"], final["iteration"] + 1):
        directory = loop / str(local)
        measured = (directory / "measurement.json").read_bytes()
        generated = (directory / "capture/raw" / member / "result.json").read_bytes()
        m, g = json.loads(measured), json.loads(generated)
        if not (m["generation"] == local and m["member"] == g["id"] == member and m["complete"] and g["ended"]):
            raise ValueError(f"Unmatched or incomplete fitted observation at {directory}")
        match = ending.search(g["content"].strip())
        if match is None:
            raise ValueError(f"Unrecognized answer format at {directory}")
        available, required, verdict = match.groups()
        outputs = (int(available), int(required), verdict)
        rows.append(dict(step=offset + local, loss=m["objective"], available_j=outputs[0],
                         required_j=outputs[1], verdict=verdict,
                         all_requested_outputs_correct=outputs == expected_fields, measured_at=m["measured_at"],
                         measurement_sha256=hashlib.sha256(measured).hexdigest(),
                         generated_result_sha256=hashlib.sha256(generated).hexdigest()))
    if rows[-1]["measurement_sha256"] != final["last_measurement"]["sha256"]:
        raise ValueError("The endpoint differs from the run's final recorded measurement.")
    snapshot["rows"] = rows
    path.write_text(json.dumps(snapshot, indent=2) + "\n")


def history():
    rows = read(RETRO / "2026-11-17-one-pair-correction/fitted-outcomes.json")
    rows += read(HERE / "figure-1-continuation.json")["rows"]
    c = Canvas("Fresh fitted trace loss and audited free arithmetic", 292)
    c.txt(5, 20, "Fresh trace loss", bold=True)
    x = lambda s: 84 + (s - rows[0]["step"]) * 687 / (rows[-1]["step"] - rows[0]["step"])
    y = lambda v: 220 - (math.log10(v) - math.log10(0.00005)) * 178 / math.log10(20)
    for v, label in [(0.00005,"0.00005"),(0.0001,"0.0001"),(0.0002,"0.0002"),(0.0005,"0.0005"),(0.001,"0.001")]:
        c.line(84,y(v),771,y(v))
        c.txt(77,y(v)+5,label,13,"end",GRAY)
    for a,b in zip(rows,rows[1:]):
        c.line(x(a["step"]),y(a["loss"]),x(b["step"]),y(b["loss"]),"#79908c",1.8)
    for row in rows:
        right = row["all_requested_outputs_correct"]
        c.mark(x(row["step"]),y(row["loss"]),TEAL if right else RED,not right,4.6)
    for s in (317,325,335,345,355,365,375,383,390):
        c.txt(x(s),241,s,14,"middle")
    c.txt(430,262,"Cumulative installed update",14,"middle")
    c.mark(134,284,TEAL); c.txt(145,289,"91 J available · YES",14)
    c.mark(478,284,RED,True); c.txt(489,289,"11 J available · NO",14)
    c.save("fitted-history.svg")


def contexts():
    d = read(DEV / "2026-11-17-descent-context/native-result.json")
    c = Canvas("Addition-minus-subtraction logit margin at frozen checkpoint 175", 252)
    left,right=312,732
    x=lambda v: left + (v+8)*(right-left)/16
    c.txt(5,19,"Receiving question / pre-sign context",14,bold=True)
    c.txt((left+right)/2,19,"Addition minus subtraction (logits)",14,"middle",bold=True)
    for tick in (-8,-4,0,4,8):
        c.line(x(tick),34,x(tick),199,GRAY if tick==0 else RULE)
        c.txt(x(tick),221,f"{tick:+d}" if tick else "0",13,"middle")
    for i,(key,label) in enumerate([("fit__fit","Fitted / fitted context"),("fit__dev","Fitted / second context"),("dev__fit","Second / fitted context"),("dev__dev","Second / second context")]):
        val=d["native"]["observations"][key]["sign"]["margin"]; y=58+42*i
        color=TEAL if val>0 else RED
        c.txt(5,y+5,label,16)
        c.line(x(0),y,x(val),y,color,8); c.mark(x(val),y,color,val<0,5.6)
        c.txt(x(val)+(9 if val>0 else -9),y+5,f"{val:+.2f}",14,"start" if val>0 else "end",color,True)
    c.txt(left,248,"Favors subtraction",13,color=RED); c.txt(right,248,"Favors addition",13,"end",TEAL)
    c.save("context-margins.svg")


def geometry():
    base=RETRO / "2026-11-17-native-tolerance-recurrence"
    radial=read(base/"radial-separation.json")
    axis=read(base/"axis-row312-merge61.json")
    analysis=read(base/"analysis.json")
    cell=next(c for c in analysis["cells"] if c["reference_position"]==312 and c["merge"]==61)
    direc=cell["directional"]
    c=Canvas("Radial distance and signed projection of the same 28 development traces", 311)
    c.mark(8,14,TEAL); c.txt(20,20,"Net-correct descent (15)",14)
    c.mark(310,14,RED,True); c.txt(322,20,"Wrong descent (13)",14)
    for i,(title,lim,limit,rticks) in enumerate([("Distance from the reference turn",0.75,radial["original_radius"],[0,.2,.4,.6]),("Projection on the fitted difference",.45,axis["upper_relative_to_reference"],[0,.1,.2,.3,.4])]):
        off=37+i*137; x=lambda v:170+v*593/lim
        c.txt(5,off+15,title,15,bold=True)
        c.line(x(limit),off+24,x(limit),off+88,GRAY,1.7,True)
        c.txt(x(limit),off+20,f"limit {limit:.3f}",13,"middle",GRAY)
        for j,(key,short,col) in enumerate([("positive_distances","Net correct",TEAL),("wrong_distances","Wrong",RED)]):
            y=off+49+j*26
            c.txt(150,y+4,short,14,"end")
            c.line(170,y,763,y)
            coords=[]
            for row in radial[key]:
                member=row["member"]
                if i==0: v=row["distance"]
                elif j==0: v=max(direc["positive_projections"].get(member,[0.0]))
                else: v=direc["wrong"][member]["mean_projection"]
                coords.append(v)
            # Stack overlapping symbols vertically; offsets do not encode a quantity.
            taken=[]
            for v in sorted(coords):
                offsets=(0,-7,7,-14,14,-21,21)
                yy=next((y+o for o in offsets if all(abs(y+o-py)>6 or abs(x(v)-px)>6 for px,py in taken)),y)
                taken.append((x(v),yy)); c.mark(x(v),yy,col,j==1,4.1)
        for tick in rticks:
            c.txt(x(tick),off+105,f"{tick:g}",13,"middle",GRAY)
    c.save("precursor-geometry.svg")



def held():
    d=read(RETRO / "2026-11-17-natural-descent-decisions/transfer-correspondence.json")["original/transfer"]
    c=Canvas("Frozen projection at three measured positions of the reserved question",148)
    x=lambda v: 335+v*1000
    c.txt(5,16,"Reserved question · original weights",14,bold=True)
    c.txt(x(d["limit"]),16,f'Limit {d["limit"]:.3f}',14,"middle",GRAY,True)
    c.line(x(d["limit"]),23,x(d["limit"]),125,GRAY,1.5,True)
    samples=[("Largest before sign", d["largest_prior_decoded_projection"]),("Immediately before sign",d["one_before_operator"]),("Paired to reference",d["actual_training_cell"][0])]
    for i,(label,sample) in enumerate(samples):
        y=42+i*34
        c.txt(5,y+5,f'{label} (row {sample["current_row"]})',15)
        v=sample["projection"]
        c.line(x(0),y,x(.42),y)
        c.mark(x(v),y,INK,False,5)
        c.txt(x(v)-9,y-5,f"{v:.3f}",13,"end")
    for v in (0,.1,.2,.3,.4):
        c.txt(x(v),143,f"{v:g}",13,"middle",GRAY)
    c.save("reserved-projection.svg")


def operation_controls():
    run = STUDY / "runs/descent-operand-identity"
    controls = read(run / "annotations/events.json")["controls"]
    original = read(run / "result.json")["summary"]["local"]
    c = Canvas("Descent and friction in generated answers", 249,
               "On the fitted prompt, the model subtracts 40 J of descent energy at update 387 "
               "and adds 40 J at 388. Both generations subtract 13 J for friction. In the original "
               "cohort the local trace rule flags 15 of 15 wrong descent choices, 3 of 15 valid "
               "descent choices, and 0 of 30 friction choices.")
    c.txt(5, 17, "Energy in the fitted answer (J)", 17, bold=True)
    c.txt(423, 17, "Pre-choice alerts in the original cohort", 17, bold=True)
    c.line(403, 0, 403, 245)

    def y(energy):
        return 186 - 1.24 * energy

    xs = (47, 177, 303)
    for step, color, diamond in (("387", RED, True), ("388", TEAL, False)):
        descent, friction = (controls[step][role] for role in ("descent", "friction"))
        if descent["result_J"] != friction["lhs_J"]:
            raise ValueError(f"Disconnected energy accounting at update {step}")
        start, bottom, end = descent["lhs_J"], descent["result_J"], friction["result_J"]
        c.line(xs[0], y(start), xs[1], y(bottom), color, 2.5, diamond)
        c.line(xs[1], y(bottom), xs[2], y(end), color, 2.5, diamond)
        c.mark(xs[1], y(bottom), color, diamond, 5.6)
        c.mark(xs[2], y(end), color, diamond, 5.6)
        c.txt(xs[1], y(bottom) + (22 if diamond else -10), bottom, 15, "middle", color, True)
        c.txt(xs[2] + 12, y(end) + 5, end, 15, color=color, bold=True)
    start_wrong = controls["387"]["descent"]["lhs_J"]
    if start_wrong != controls["388"]["descent"]["lhs_J"]:
        raise ValueError("Fitted generations have different starting energies.")
    c.mark(xs[0], y(start_wrong), INK, radius=5.6)
    c.txt(xs[0], y(start_wrong) - 12, start_wrong, 15, "middle", bold=True)
    for x, label in zip(xs, ("Start", "After descent", "After friction")):
        c.txt(x, 209, label, 14, "middle")
    for x, step, color, diamond in ((17, 387, RED, True), (188, 388, TEAL, False)):
        c.line(x, 235, x + 28, 235, color, 2.5, diamond)
        c.mark(x + 14, 235, color, diamond, 4.8)
        c.txt(x + 35, 240, f"Update {step}", 14)

    wrong, valid = original["wrong"], original["valid"]
    data = (("Wrong descent", wrong["target_alert"], wrong["count"]),
            ("Valid descent", valid["target_alert"], valid["count"]),
            ("Friction subtraction", wrong["friction_alert"] + valid["friction_alert"],
             wrong["count"] + valid["count"]))
    for idx, (label, count, total) in enumerate(data):
        baseline = 62 + 65 * idx
        c.txt(424, baseline - 9, label, 15)
        c.line(425, baseline + 5, 682, baseline + 5, RULE, 6)
        if count:
            c.line(425, baseline + 5, 425 + 257 * count / total, baseline + 5, INK, 6)
        c.txt(696, baseline + 10, f"{count} / {total}", 16, bold=True)
    for x, label in ((425, "0%"), (682, "100%")):
        c.txt(x, 225, label, 13, "middle", GRAY)
    c.save("descent-friction.svg")


def prefix_transfer():
    run = STUDY / "runs/prefix-direction-generalization"
    panel = read(run / "panel.json")
    paired = read(run / "full-field/paired-prefix-objectives.json")["cases"]
    parameters = {case["family"]: case["physics"] for case in panel}
    changes = {}
    for case in paired:
        before, after = case["objective_on_common_source_cells"]
        changes[case["family"], case["ordering"]] = 100 * (after - before) / before

    # Both columns use the same scale. The printed change resolves near-zero marks.
    largest = max(map(abs, changes.values()))
    canvas = Canvas("Change in the held-out prefix loss after eleven updates", 174,
                    "Paired free generations scored on computational locations represented at both checkpoints.")
    canvas.txt(4, 20, "Energy at foot / climb cost", size=14, bold=True)
    columns = [(350, "chronological", "In route order"), (655, "endpoint-first", "Endpoint first")]
    for center, _, label in columns:
        canvas.txt(center, 20, label, size=15, anchor="middle", bold=True)
        canvas.line(center, 29, center, 155, color=GRAY)
    for row, (family, p) in enumerate(parameters.items()):
        y = 49 + row * 33
        available = p["m"] * p["v"] ** 2 / 2 + p["m"] * p["g"] * p["drop"] - p["friction"]
        required = p["m"] * p["g"] * p["rise"]
        canvas.txt(4, y + 5, f"{available:g} J / {required:g} J", size=16)
        for center, ordering, _ in columns:
            change = changes[family, ordering]
            x = center + (78 * change / largest if largest else 0)
            color = TEAL if change < 0 else RED
            canvas.line(center, y, x, y, color=color, width=3)
            canvas.mark(x, y, color)
            align = "start" if change >= 0 else "end"
            canvas.txt(x + (8 if change >= 0 else -8), y + 4,
                       f"{change:+.2f}%", size=13, anchor=align, color=INK)
    canvas.save("prefix-transfer.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-history", action="store_true")
    if parser.parse_args().refresh_history:
        refresh_history()
    FIG.mkdir(exist_ok=True)
    history()
    contexts()
    geometry()
    held()
    operation_controls()
    prefix_transfer()
