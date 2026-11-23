"""Report exact label calibration without changing the statistic or selecting a target."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline.overlap.analyze import read, write
from pipeline.native_records.run_declaration import check_native_run


def tail(row):
    return f"{row['greater_equal']:,}/{row['assignments']:,} ({100*row['greater_equal']/row['assignments']:.2f}%)"


def main(output):
    output=Path(output);folder=output/'calibration'
    figures=folder/'figures';check_native_run(figures,output/'declaration.json')
    figures.mkdir(exist_ok=True)
    data=read(folder/'summary.json')
    joint={(r['gpu'],r['scope'],r['passage'],r['metric']):r for r in data['joint_statistics']}
    report=[
        '# Exact within-speed label calibration', '',
        '**Status: calibration complete; overlap acceptance remains withdrawn pending independent review. No edit, fit, inference or new capture.**', '',
        'Unequal group sizes do bias the held-out statistic, but the exact null does not explain all the observed climb separation. On GPU 0, the equal-response, class-balanced climb margin exceeds most assignments preserving those same sizes. The full-population position subset gives the same qualitative result. This is conditional observational support; it does not isolate a capability direction.', '',
        '## Design and denominator', '',
        'One outcome label belongs to an entire response, unchanged across its positions. Every assignment with the original failure count is enumerated separately within card and speed. Each held-out member is absent from the direction used to score it. A response averages valid merges at its own aligned prediction, then predictions; group summaries average responses equally within outcome, then average the two outcomes equally. No cell is counted as an independent trial.', '',
        '| Card | Speed | Failures | Successes | Assignments | Conditional assignments: failure / success |',
        '|---|---:|---:|---:|---:|---:|']
    for s in data['strata']:
        report.append(f"| GPU {s['gpu']} | {s['speed']} | {s['failures']} | {s['successes']} | {s['assignments']} | {s['conditional_assignments_per_failure']} / {s['conditional_assignments_per_success']} |")
    report += ['', 'GPU 0 combines 28 × 36 × 495 = 498,960 scalar assignment combinations. These reuse 29 responses; they are not additional observations. GPU 1 has only 35 assignments on seven responses. Its speed-4 (1 failure / 5 successes) and speed-6 (1 / 2) strata remain undefined: leaving out the sole failure leaves no failure centroid.', '',
        'All-aligned positions are the main view. Each assignment recomputes which positions contain enough members of both labels. The full-population view contains only positions with all eligible responses on that card and therefore holds membership fixed. Missing comparisons are not zero scores. Every joint statistic reported below is finite for all assignments; per-member coverage is saved in summary.json.', '',
        '## Primary mean-margin result', '',
        '| Card / passage | Coverage | Observed | Null mean | Assignments at least as large |',
        '|---|---|---:|---:|---:|']
    for gpu in (0,1):
        for role in ('destination','descent'):
            for scope in ('all_aligned','full_population'):
                r=joint[(gpu,scope,role,'mean_margin')]
                report.append(f"| GPU {gpu} / {'climb' if role=='destination' else role} | {scope} | {r['observed']:.6f} | {r['mean']:.6f} | {tail(r)} |")
    report += ['', 'The last column is an exact conditional rank including the observed assignment, not an unconditional causal p-value or a chosen acceptance threshold. Smaller means fewer design-matched assignments score as highly. No significance cutoff was selected.', '',
        '![Exact assignment distributions](figures/exact-null-distributions.png)', '',
        '## Other declared statistics', '',
        'The fraction of responses with a positive mean is weaker on GPU 0 than the average signed margin. This distinction matters: magnitude carries information that the zero/nonzero count discards.', '',
        '| Card / passage | Statistic (all aligned) | Observed | Null mean | Assignments at least as large |',
        '|---|---|---:|---:|---:|']
    for gpu in (0,1):
        for role in ('destination','descent'):
            for metric in ('positive_mean_fraction','own_side_fraction'):
                r=joint[(gpu,'all_aligned',role,metric)]
                report.append(f"| GPU {gpu} / {role} | {metric} | {r['observed']:.6f} | {r['mean']:.6f} | {tail(r)} |")
    report += ['', 'Own-side fraction averages positive held-out merge projections within predictions and responses. Neither fraction measures correct answers or a repair.', '',
        '## The group-size effect is real', '',
        '| Card / speed, climb | Failure observed / null mean | Success observed / null mean | Balanced margin rank |',
        '|---|---:|---:|---:|']
    for s in data['strata']:
        g=s['group_distributions']['all_aligned_destination'];f=g['failure_mean_margin'];v=g['success_mean_margin'];b=g['balanced_mean_margin']
        report.append(f"| GPU {s['gpu']} / {s['speed']} | {f['observed']:.5f} / {f['mean']:.5f} | {v['observed']:.5f} / {v['mean']:.5f} | {tail(b)} |")
    report += ['', 'For GPU 0 at speed 4, random labels already give the minority failure group a negative mean and the majority success group a positive mean. Speed 6 shows the same effect; speed 8 reverses the majority. Therefore a zero margin is not a calibrated chance reference. However, both observed outcome means exceed their own size-matched null means in each GPU-0 speed stratum. That weakens a pure group-size explanation without proving a procedure-specific cause.', '',
        '## Individual responses', '',
        'Each member is compared with assignments where it retains its actual outcome label. This fixes its own group size. All-assignment distributions are also retained. Above the null mean is descriptive, not a significance rule.', '',
        '| Card / passage | Failures above their own null mean | Successes above their own null mean |',
        '|---|---:|---:|']
    for gpu in (0,1):
        for role in ('destination','descent'):
            counts=[]
            for outcome in ('failure','natural_numerical_success'):
                rr=[r['same_outcome_null'] for r in data['per_response'] if r['gpu']==gpu and r['passage']==role and r['scope']=='all_aligned' and r['outcome']==outcome]
                counts.append(f"{sum(r['observed']>r['mean'] for r in rr)}/{len(rr)}")
            report.append(f"| GPU {gpu} / {role} | {' | '.join(counts)} |")
    report += ['', 'GPU 1 is still unresolved: its four failures all improve against their own null mean, but one of three successes is well below it. GPU 0 also has weak and negative individual comparisons. Descent is less consistent than climb and its response-positive fraction does not stand out from the null.', '',
        '![Each climb response against its own conditional null](figures/member-null-margins.png)', '',
        'The full per-response table follows in [PER-RESPONSE-CALIBRATION.md](PER-RESPONSE-CALIBRATION.md), with machine-readable rows in per-response-calibration.csv. Names, speeds, coverage, observed margins, null means, and exact conditional ranks are all retained.', '',
        '## What this does and does not establish', '',
        '- It calibrates the previously uncalibrated statistic on the exact observed design. The approximately 69% cell average cannot be read as 69% independent predictive accuracy.',
        '- The GPU-0 climb difference has evidence beyond group-size bias under the fixed alignment. This survives the strict full-population subset; it is not solely caused by changing coverage.',
        '- The correspondence was originally built with outcome groups. Permuting labels conditional on that correspondence does not test the possible selection effect of building the correspondence itself.',
        '- Speed is held fixed. Mass, drop, loss, rise, wording, generated history and dependencies among template variants remain possible explanations. Labels were not randomly assigned. Exact enumeration does not cure these limitations.',
        '- Many later differences can be consequences of the wrong calculation. These observational results do not establish causal repair locations or a clean editing direction.',
        '- The descriptive L13/feed-forward and layers-10–14 location results, and component reconstruction, remain in OVERLAP.md. This calibration neither reselects components nor validates a causal interpretation of them.', '',
        '## Acquisition requirement', '',
        'This fixed-data permutation distribution cannot identify how many new responses will be sufficient. It holds the existing trace values fixed and has no measurement of future signal, noise, template dependence or failure/success yield. Choosing a power level, projecting independent-sample behavior, or treating duplicated existing responses as new data would add assumptions not measured here.', '',
        'A separate acquisition proposal therefore distinguishes the measured imbalance and resolution deficits from a sufficient sample size, which remains unidentified. No arbitrary target count or stopping threshold is supplied. The reserved-40b questions and outcomes remain sealed.', '',
        '## Verification', '']
    compared=sum(s['direct_check']['compared'] for s in data['strata']);error=max(s['direct_check']['max_absolute_error'] for s in data['strata'])
    report += [f'The Gram calculation reproduced {compared:,} saved finite held-out margins, with maximum absolute difference {error:.3g}, zero sign differences and zero validity differences. All {len(data["reproduction"])} eligible response/passage means reproduced to maximum absolute error {max(r["absolute_difference"] for r in data["reproduction"]):.3g}, with identical position counts.', '',
        'The focused overlap suite passed 24 tests. Three calibration tests compare every small partition against the direct held-out calculation, verify enumeration, and check rotation/translation invariance and undefined zero directions. The round-two verifier passed. The first launcher attempt stopped on a duplicate keyword before reading data; the corrected committed launcher completed with exit 0. No scientific result came from the failed launch.', '']
    (folder/'CALIBRATION.md').write_text('\n'.join(report))
    member_text=['# Per-response exact calibration', '', 'Each row conditions the null on that response retaining its actual label. The rank counts assignments at least as large as the observed mean; no acceptance threshold is imposed. Undefined strata appear at the end.', '']
    csvrows=[]
    for gpu in (0,1):
        for role in ('destination','descent'):
            member_text += [f'## GPU {gpu}: {role}', '', '| Response | Outcome | Speed | Positions | Observed | Null mean | At least observed |', '|---|---|---:|---:|---:|---:|---:|']
            for r in data['per_response']:
                s=r['same_outcome_null']
                if r['scope']=='all_aligned' and r['gpu']==gpu and r['passage']==role:
                    member_text.append(f"| {r['case']} | {'F' if r['outcome']=='failure' else 'S'} | {r['speed']} | {r['observed_positions']} | {s['observed']:.6f} | {s['mean']:.6f} | {tail(s)} |")
            member_text.append('')
    for r in data['per_response']:
        s=r['same_outcome_null'];csvrows.append({k:r[k] for k in ('gpu','speed','case','outcome','scope','passage','observed_positions')}|s)
    member_text += ['## Unsupported strata', '']+[f"- GPU {r['gpu']}, speed {r['speed']}: {r['failures']} failures / {r['successes']} successes; {', '.join(r['cases'])}." for r in data['unavailable']]
    (folder/'PER-RESPONSE-CALIBRATION.md').write_text('\n'.join(member_text)+'\n')
    with (folder/'per-response-calibration.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=csvrows[0]);w.writeheader();w.writerows(csvrows)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(16,10),layout='constrained')
    with np.load(folder/'joint-distributions.npz') as arrays:
        for gpu in (0,1):
            for col,role in enumerate(('destination','descent')):
                ax=axes[gpu,col];notes=[]
                for scope,color,label in [('all_aligned','#197b76','All aligned'),('full_population','#bd7e22','Full population')]:
                    a=np.sort(arrays[f'gpu{gpu}_{scope}_{role}_mean_margin']);r=joint[(gpu,scope,role,'mean_margin')]
                    # Duplicate x values are collapsed without changing the exact ECDF.
                    x,count=np.unique(a,return_counts=True);ax.step(x,np.cumsum(count)/len(a),where='post',color=color,label=label)
                    ax.axvline(r['observed'],color=color,ls='--',lw=1.5)
                    notes.append(f"{label}: {tail(r)} at least observed")
                ax.set_title(f"GPU {gpu} · {'final climb' if role=='destination' else 'descent'}",loc='left')
                ax.set_xlabel('Class-balanced, equal-response mean signed margin');ax.set_ylabel('Fraction of label assignments ≤ margin')
                ax.grid(alpha=.15);ax.legend(loc='upper left',frameon=False);ax.text(.98,.04,'\n'.join(notes),transform=ax.transAxes,ha='right',va='bottom',fontsize=9)
    fig.suptitle('Group size biases the score; the climb still stands out under this conditional null',fontsize=19,x=.02,ha='left')
    fig.supxlabel('Dashed lines = actual labels. Labels are fixed across each response and permuted only within speed and card.\n'
                  '498,960 combinations on GPU 0 reuse 29 responses; GPU 1 has 35 combinations on seven. Conditional on the saved alignment; no causal claim.',fontsize=10)
    for suffix in ('png','svg'):fig.savefig(figures/f'exact-null-distributions.{suffix}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(17,12),layout='constrained',sharex=True)
    for gpu,ax in enumerate(axes):
        rr=[r for r in data['per_response'] if r['gpu']==gpu and r['scope']=='all_aligned' and r['passage']=='destination']
        labels=[]
        for y,r in enumerate(rr):
            s=r['same_outcome_null'];center=s['mean'];color='#197b76' if r['outcome']=='failure' else '#bd7e22'
            ax.plot([s['minimum']-center,s['maximum']-center],[y,y],color='#bcc6c5',lw=2)
            ax.scatter(s['observed']-center,y,c=color,s=35,zorder=3)
            labels.append(f"{'F' if r['outcome']=='failure' else 'S'} v{r['speed']} · {r['case']}")
        ax.set_yticks(range(len(rr)),labels,fontsize=9);ax.set_ylim(len(rr)-.5,-.5);ax.axvline(0,color='#999',lw=1)
        ax.set_title(f'GPU {gpu}: {len(rr)} scored responses',loc='left');ax.set_xlabel('Mean margin minus this member’s conditional null mean');ax.grid(axis='y',alpha=.1)
    fig.suptitle('Climb: each response compared with labels preserving its own group size',fontsize=19,x=.02,ha='left')
    fig.supxlabel('Dot = observed; gray line = exact null minimum to maximum, centered on that member’s null mean. Teal F = failure; ochre S = success.\n'
                  'Ranges are not confidence intervals. GPU 1’s other nine responses have undefined same-speed folds and are listed in the report.',fontsize=10)
    for suffix in ('png','svg'):fig.savefig(figures/f'member-null-margins.{suffix}',dpi=160)
    plt.close(fig)
    write(folder/'report-metadata.json',{'status':'pending independent review; accepted remains false','no_new_inference':True,
        'report':'CALIBRATION.md','per_response':'PER-RESPONSE-CALIBRATION.md','direct_margin_count':compared,'direct_max_error':error})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
