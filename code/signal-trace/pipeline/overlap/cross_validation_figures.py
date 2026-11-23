"""Display held-out group evidence, aligned populations and speed-control limits."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline.overlap.analyze import read
from pipeline.research_guard.verify_round2 import verify, ROOT

ROLES = ['opening', 'initial', 'descent', 'after_loss', 'destination', 'comparison', 'final_fields']
LABELS = ['Opening', 'Initial energy', 'Descent', 'After loss', 'Final climb', 'Comparison', 'Final fields']


def main(output):
    if verify(ROOT):
        raise ValueError('Round-two verifier refused')
    output = Path(output)
    data = read(output / 'cross-validated/summary.json')
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), layout='constrained')
    fig.patch.set_facecolor('#f7f6f1')
    for gpu in (0, 1):
        rows = {(r['coverage'], r['passage']): r for r in data['passages'] if r['gpu'] == gpu}
        ax = axes[0, gpu]
        for scope, family, color, marker, name in [
            ('all_aligned', 'all_members', '#64746c', 'o', 'All aligned members'),
            ('all_aligned', 'within_speed', '#247f78', 'o', 'Within equal speed'),
            ('full_population', 'within_speed', '#247f78', 's', 'Equal speed, fully covered positions')]:
            x, y = [], []
            for i, role in enumerate(ROLES):
                row = rows.get((scope, role))
                if row:
                    x.append(i); y.append(row[family]['mean_balanced_fraction'])
            ax.plot(x, y, marker=marker, color=color, ls=':' if scope == 'full_population' else '-',
                    mfc='none' if scope == 'full_population' else color, label=name)
        ax.axhline(.5, color='#aaa', ls='--', lw=1)
        ax.set_ylim(0, 1); ax.set_ylabel('Mean class-balanced own-side fraction')
        ax.set_xticks(range(len(ROLES)), LABELS, rotation=25, ha='right')
        ax.set_title(f'GPU {gpu} · predict each omitted response from the others', loc='left')
        ax.legend(frameon=False, fontsize=9)
        counts = []
        for role in ROLES:
            row = rows.get(('all_aligned', role))
            counts.append(f"{row['matched_positions']}/{row['positions']}" if row else '—')
        ax.text(.01, -.28, 'Matched / aligned positions: ' + ' · '.join(counts), transform=ax.transAxes, fontsize=9)
        values = np.full((len(ROLES), 35), np.nan)
        layers = {(r['passage'], r['layer']): r for r in data['layers']
                  if r['gpu'] == gpu and r['coverage'] == 'all_aligned'}
        for i, role in enumerate(ROLES):
            for layer in range(35):
                row = layers.get((role, layer))
                if row:
                    values[i, layer] = row['within_speed']['mean_balanced_fraction']
        ax = axes[1, gpu]
        cmap = plt.get_cmap('BrBG').copy(); cmap.set_bad('#d9dcd7')
        im = ax.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect='auto')
        ax.set_yticks(range(len(ROLES)), LABELS); ax.set_xlabel('Native layer')
        ax.set_title('Within-speed evidence across the response', loc='left')
        fig.colorbar(im, ax=ax, shrink=.7, label='Own-side fraction · midpoint 0.5')
        ax = axes[2, gpu]
        for key, color, name in [('mean_failure_margin', '#d16b4b', 'Omitted failures'),
                                 ('mean_success_margin', '#247f78', 'Omitted successes')]:
            vals = [layers[('destination', layer)]['within_speed'][key] for layer in range(35)]
            ax.plot(range(35), vals, color=color, label=name)
        ax.axhline(0, color='#aaa', ls='--', lw=1)
        ax.set_title('Final climb · distance onto the omitted member’s own side', loc='left')
        ax.set_xlabel('Native layer'); ax.set_ylabel('Signed distance along training direction')
        ax.legend(frameon=False)
        if gpu == 1:
            ax.text(.02, .98, 'Only speed 8 has enough members: at most 4 failures + 3 successes.',
                    transform=ax.transAxes, va='top', fontsize=9)
    for ax in axes.flat:
        ax.set_facecolor('#f7f6f1'); ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Outcome separation persists at the climb when speed is held equal', x=.04, ha='left', fontsize=22)
    fig.supxlabel('Each omitted member is absent from its training centroids, direction and midpoint. All aligned subsets are the main view.\n'
                  'Cells are repeated observations, not independent trials. Card populations and matched populations differ. Other numerical inputs and history remain uncontrolled.\n'
                  'Positive separation is predictive evidence; it does not make the full-data vector a purified capability or an accepted editing target.', fontsize=10)
    fig.savefig(output / 'figures/cross-validated-overlap.png', dpi=160)
    fig.savefig(output / 'figures/cross-validated-overlap.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
