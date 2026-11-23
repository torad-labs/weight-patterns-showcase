"""Measure common signed portions of independently formed group envelopes."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from pipeline.overlap.analyze import SHARED, read, write, finite
from pipeline.overlap.envelopes import group_envelope, common_difference, omission_cosines
from pipeline.native_records.run_declaration import check_native_run


def main(output):
    output = Path(output); dest = output / "common-coordinates"
    check_native_run(dest, output / "declaration.json")
    if dest.exists(): raise ValueError("Never overwrite a common-overlap measurement")
    dest.mkdir()
    source = read(SHARED / "observation/response-correspondence/index.json")
    original = SHARED / "observation/original-model-traces/full-response-turns"
    arrays = {}; gains = {}
    for case, meta in source["cases"].items():
        if meta["role"] not in ("failure", "natural_numerical_success"): continue
        arrays[case] = np.load(original / case / "turns.npy", mmap_mode="r")
        norms = np.load(original / case / "norms.npy", mmap_mode="r")
        gains[case] = np.log(norms[:, :, 1].astype("f8") / norms[:, :, 0].astype("f8"))
    cards = []
    for gpu in sorted({source["cases"][case]["gpu"] for case in arrays}):
        previous = output / f"contrasts/gpu{gpu}"
        meta = read(previous / "index.json"); raw = np.load(previous / "difference.npy", mmap_mode="r")
        folder = dest / f"gpu{gpu}"; folder.mkdir()
        shape = raw.shape
        gap = np.lib.format.open_memmap(folder / "gap.npy", mode="w+", dtype="f8", shape=shape)
        bounds = {key: np.lib.format.open_memmap(folder / (key + ".npy"), mode="w+", dtype="f8", shape=shape)
            for key in ("failure_min", "failure_max", "success_min", "success_max")}
        metrics = np.zeros((*shape[:2], 7)); gain_gap = np.zeros(shape[:2]); loo = []; rows = []
        for i, row in enumerate(meta["rows"]):
            fm, sm = row["failure_members"], row["success_members"]
            f = np.stack([arrays[c][p] for c, p in fm]).astype("f8")
            s = np.stack([arrays[c][p] for c, p in sm]).astype("f8")
            fe, se = group_envelope(f), group_envelope(s)
            d = common_difference(fe, se); g = d["gap"]; gap[i] = g
            for name, envelope in (("failure", fe), ("success", se)):
                bounds[name + "_min"][i] = envelope["minimum"]
                bounds[name + "_max"][i] = envelope["maximum"]
            if not np.allclose(d["raw_difference"], raw[i], rtol=0, atol=0):
                raise ValueError("Independent means differ from the saved group overlay")
            cosines = omission_cosines(f, s, g)
            nonempty = np.any(np.isfinite(cosines), axis=0)
            worst = np.full(g.shape[0], np.nan); average = worst.copy()
            worst[nonempty] = np.nanmin(cosines[:,nonempty], axis=0)
            average[nonempty] = np.nanmean(cosines[:,nonempty], axis=0)
            raw_sq = np.sum(raw[i]**2, axis=-1); gap_sq = np.sum(g*g, axis=-1)
            projection = np.sum(raw[i]*g, axis=-1)
            # Agreement fraction has no cutoff. The common gap itself requires unanimity.
            direction = np.sign(raw[i]); pair_agreement = np.zeros_like(g)
            for fv in f:
                pair_agreement += np.mean((s-fv)*direction > 0, axis=0) / len(f)
            mean_agreement = np.divide(np.sum(pair_agreement*raw[i]**2, axis=-1), raw_sq,
                out=np.full_like(raw_sq, np.nan), where=raw_sq>0)
            metrics[i] = np.stack([raw_sq, gap_sq, np.count_nonzero(g, axis=-1),
                projection, worst, average, mean_agreement], axis=-1)
            gf = group_envelope(np.stack([gains[c][p] for c,p in fm])[:,:,None])
            gs = group_envelope(np.stack([gains[c][p] for c,p in sm])[:,:,None])
            gain_gap[i] = common_difference(gf,gs)["gap"][:,0]
            row = dict(row); row["gap_squared"] = gap_sq; row["common_coordinate_count"] = np.count_nonzero(g,axis=-1)
            row["minimum_omission_cosine"] = worst; rows.append(row)
            loo.append(dict(position=i, member_order=fm+sm, cosines=cosines))
        gap.flush()
        for value in bounds.values(): value.flush()
        np.save(folder / "metrics.npy", metrics); np.save(folder / "gain-gap.npy", gain_gap)
        summaries = []
        for role in dict.fromkeys(row["role"] for row in rows):
            for coverage in ("all_aligned", "full_population"):
                ids = [i for i,row in enumerate(rows) if row["role"]==role and (coverage=="all_aligned" or row["full_population"])]
                if not ids: continue
                m = metrics[ids]; full = int(sum(rows[i]["full_population"] for i in ids))
                summaries.append(dict(role=role,coverage=coverage,positions=len(ids),full_population_positions=full,
                    raw_squared_by_merge=np.mean(m[:,:,0],axis=0),common_squared_by_merge=np.mean(m[:,:,1],axis=0),
                    common_coordinates_by_merge=np.mean(m[:,:,2],axis=0),
                    cells_with_common_coordinates=int(np.sum(m[:,:,2]>0)),
                    common_gain_by_merge=np.mean(gain_gap[ids],axis=0)))
        earliest = {}
        for coverage in ("all_aligned", "full_population"):
            for row in rows:
                if coverage=="full_population" and not row["full_population"]: continue
                found = np.flatnonzero(np.asarray(row["common_coordinate_count"])>0)
                if len(found):
                    earliest[coverage] = dict(position=row["position"],failure_prediction=row["failure_prediction"],
                        success_prediction=row["success_prediction"],role=row["role"],merge=int(found[0]),
                        layer=int(found[0]//3),common_coordinate_count=int(row["common_coordinate_count"][found[0]]),
                        gap_squared=float(row["gap_squared"][found[0]])); break
        record = dict(gpu=gpu,population=meta["population"],rows=rows,summaries=summaries,earliest=earliest,
            metrics=["raw_difference_squared","common_gap_squared","common_coordinate_count","raw_dot_gap",
                     "minimum_omission_cosine","mean_omission_cosine","raw_energy_weighted_sign_agreement"],
            interpretation="Every retained signed coordinate occurs between every included cross-group member. Native-coordinate, conservative descriptive common portion; no semantic or causal identification and no statistical significance claim.")
        write(folder / "index.json", finite(record)); write(folder / "omissions.json", finite(loo))
        cards.append(dict(gpu=gpu,path=str(folder),positions=len(rows),common_cells=int(np.sum(metrics[:,:,2]>0)),earliest=earliest))
        print(finite(cards[-1]), flush=True)
    write(dest / "index.json", finite(dict(cards=cards,definition="Coordinatewise gap between independent group ranges; zero for overlapping ranges or singleton groups.",
        raw_vector_support_correction="The original whole-vector margins do not filter member-specific components. They remain descriptive, not a recurring target.",
        numerical_scope="Tests use the stored finite float32 direction changes, converted exactly to float64 before subtraction. No empirical tolerance is used to alter sign; last-bit nonzero differences are not evidence of significance.")))

if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("output",type=Path);args=parser.parse_args();main(args.output)
