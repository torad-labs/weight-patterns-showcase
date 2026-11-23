"""One mean margin per response: make stratum and individual disagreement visible."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline.overlap.analyze import read
from pipeline.native_records.run_declaration import check_native_run


def main(output):
    output = Path(output)
    destination = output/'cross-validated/per-response-figures'
    check_native_run(destination, output/'declaration.json')
    destination.mkdir(exist_ok=True)
    records = read(output/'cross-validated/per-response.json')['records']
    fig, axes = plt.subplots(1, 2, figsize=(18, 12), layout='constrained', sharex=True)
    fig.patch.set_facecolor('#f7f6f1')
    for gpu, ax in enumerate(axes):
        rows = [r for r in records if r['gpu']==gpu and r['mode']=='within_speed']
        climb = sorted([r for r in rows if r['passage']=='destination'], key=lambda r:(r['outcome'],r['speed'],r['case']))
        lookup = {(r['case'],r['passage']):r for r in rows}
        labels = []
        for y, row in enumerate(climb):
            group = 'F' if row['outcome']=='failure' else 'S'
            labels.append(f"{group} · v={row['speed']} · {row['case']}")
            for passage, marker, color in [('destination','o','#247f78'), ('descent','s','#c38b39')]:
                r = lookup[(row['case'],passage)]
                if r['mean_signed_margin'] is not None:
                    ax.scatter(r['mean_signed_margin'], y, marker=marker, color=color, s=42, zorder=3)
            if row['mean_signed_margin'] is None:
                ax.text(0, y, '  no supported speed stratum', va='center', fontsize=9, color='#777')
        ax.axvline(0,color='#aaa',lw=1)
        ax.set_yticks(range(len(labels)),labels,fontsize=9)
        # Include rows with no plotted points; text alone does not extend limits.
        ax.set_ylim(len(labels)-.5,-.5);ax.grid(axis='y',alpha=.16)
        ax.set_xlabel('Mean signed held-out margin · positive = own group side')
        ax.set_title(f'GPU {gpu} · {len(climb)} original responses',loc='left',fontsize=15)
        ax.set_facecolor('#f7f6f1');ax.spines[['top','right','left']].set_visible(False)
        ax.scatter([],[],marker='o',color='#247f78',label='Final climb')
        ax.scatter([],[],marker='s',color='#c38b39',label='Descent')
        ax.legend(frameon=False,loc='lower right')
    fig.suptitle('Equal-speed separation differs between responses and speed strata',x=.03,ha='left',fontsize=21)
    fig.supxlabel('F = originally failing response; S = natural numerical success. Each dot averages one response’s own aligned positions and the recorded merges.\n'
                  'Missing comparisons are listed, not set to zero. Positions, directions and questions remain correlated; these are descriptive member scores.\n'
                  'GPU 1 remains unresolved. No new direction, edit, model run or acceptance threshold was introduced.',fontsize=10)
    fig.savefig(destination/'per-response-held-out.png',dpi=160)
    fig.savefig(destination/'per-response-held-out.svg')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    main(parser.parse_args().output)
