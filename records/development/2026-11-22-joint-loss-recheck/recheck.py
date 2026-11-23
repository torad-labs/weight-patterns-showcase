"""The joint stock loop's group loss, read again with a lost step counted. Reads records; trains nothing.

  python3 -I recheck.py <out.json> <steps folder> [<steps folder> ...]

Each steps folder holds <n>/group-measurement.json for the revisions of one run. Several folders are read
as one lineage, in the order given: a later run's revision 0 is the earlier run's last revision measured
again, and is kept only if its cells equal that last revision's.

The loop's loss sums, over the operations it registers on each freshly generated answer, each measured
cell's weighted squared error. A registered cell the matcher does not find in the new answer adds nothing
(`group_contrast_objective.py`: "Missing cells contribute no derivative and retain their coverage status").
So the loss can fall because a computation was repaired, or because it is no longer found.

This program separates the two. A cell is a frozen donor location of an operation of a member: its key is
(member, operation, source event, merge). Its mass is fixed by the declared donor support, so its weighted
loss is comparable across revisions. For every revision it reports:

  as logged   the run's own `measured_contribution`: the cells found in this revision only
  carried     the cells found in this revision at their measured value, and every cell not found charged
              the value it last had when it was found
  common      the cells found in every revision, so the same cells are compared throughout

`carried` is the rule Marcos chose on November 22, 2026 for the next loop: an operation the matcher loses
keeps the cost it last had. It assumes a lost cell is no better than when it was last seen. The arithmetic
of the regenerated answers (the run's `fresh-fit-readback.json`) is the independent check of that.
"""
import collections
import json
import math
import os
import sys


def cells_of(measurement):
    """{(member, operation, source event, merge): weighted loss} and each operation's role."""
    found, roles = {}, {}
    for member in measurement["members"]:
        for operation in member["operations"]:
            roles[(member["id"], operation["operation"])] = operation["role"]
            for cell in operation.get("cells") or []:
                key = (member["id"], operation["operation"], cell["source_event"], cell["merge"])
                found[key] = found.get(key, 0.0) + cell["weighted_loss"]      # a cell visited twice has its mass split over its visits
    return found, roles


def main():
    out, folders = sys.argv[1], sys.argv[2:]
    lineage = []                                   # (label, cells, logged)
    roles = {}
    for folder in folders:
        revisions = sorted(int(name) for name in os.listdir(folder)
                           if name.isdigit() and os.path.isfile(os.path.join(folder, name, "group-measurement.json")))
        for revision in revisions:
            data = json.load(open(os.path.join(folder, str(revision), "group-measurement.json"), encoding="utf-8"))
            cells, found_roles = cells_of(data)
            roles.update(found_roles)
            if not math.isclose(math.fsum(cells.values()), data["measured_contribution"], rel_tol=1e-9):
                raise SystemExit("%s revision %d: the cells sum to %.12g and the record says %.12g"
                                 % (folder, revision, math.fsum(cells.values()), data["measured_contribution"]))
            run = data["model_identity"]["run"].split("/runs/")[-1]
            if revision == 0 and lineage:
                same = set(cells) == set(lineage[-1][1]) and all(math.isclose(cells[k], lineage[-1][1][k], rel_tol=1e-9, abs_tol=1e-15) for k in cells)
                print("%s revision 0 measures the previous run's last revision again: %s" % (run, "equal, cell for cell" if same else "DIFFERENT"))
                if same:
                    continue
            lineage.append(("%s %d" % (run, revision), cells, data["measured_contribution"]))

    every = set().union(*(set(cells) for _, cells, _ in lineage))
    common = set.intersection(*(set(cells) for _, cells, _ in lineage))
    last, rows = {}, []
    for label, now, logged in lineage:
        last.update(now)
        lost = [key for key in last if key not in now]
        split = collections.defaultdict(float)
        for key in set(now) | set(lost):
            split[roles[(key[0], key[1])]] += now.get(key, last[key])
        lost_by, lost_loss = collections.Counter(), collections.defaultdict(float)
        for key in lost:
            lost_by[(key[0], key[1])] += 1
            lost_loss[(key[0], key[1])] += last[key]
        rows.append(dict(revision=label, as_logged=logged, cells_found=len(now), cells_lost=len(lost),
                         carried=math.fsum(now.values()) + math.fsum(last[key] for key in lost),
                         carried_by_role=dict(split), common=math.fsum(now[key] for key in common),
                         lost=[dict(member=m, operation=o, role=roles[(m, o)], cells=lost_by[(m, o)], charged=lost_loss[(m, o)])
                               for (m, o) in sorted(lost_by, key=lambda k: -lost_loss[k])]))

    stock = rows[0]["carried"]
    print("cells ever registered: %d | found in every revision: %d" % (len(every), len(common)))
    print("%-48s %-11s %-11s %-9s %-21s %-11s %s" % ("revision", "as logged", "carried", "vs stock", "carried failing/kept", "common", "found/lost"))
    for row in rows:
        print("%-48s %-11.7f %-11.7f %+8.1f%% %-21s %-11.7f %d/%d" % (
            row["revision"][-48:], row["as_logged"], row["carried"], 100 * (row["carried"] / stock - 1),
            "%.5f / %.5f" % (row["carried_by_role"].get("failing", 0.0), row["carried_by_role"].get("preservation", 0.0)),
            row["common"], row["cells_found"], row["cells_lost"]))
    final = rows[-1]
    print("largest charges for lost cells at the last revision (%s):" % final["revision"])
    for item in final["lost"][:8]:
        print("  %-26s %-18s %-12s %4d cells, charged %.7f" % (item["member"], item["operation"], item["role"], item["cells"], item["charged"]))
    with open(out, "w", encoding="utf-8") as sink:
        json.dump(dict(sources=folders, cells_ever_registered=len(every), cells_in_every_revision=len(common), revisions=rows), sink, indent=1)
        sink.write("\n")


main()
