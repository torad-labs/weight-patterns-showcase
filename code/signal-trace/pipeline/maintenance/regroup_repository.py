"""Regroup the repository into study groups, leaving a link at every old path.

Moves are same-filesystem renames, so bytes and inodes are unchanged. Every old
path becomes a relative symlink to its new home, which keeps the absolute paths
recorded inside historical JSON records valid without rewriting them (rewriting
would invalidate recorded hashes). Relative links inside moved Markdown/HTML are
recomputed so documents still open their neighbours. Idempotent: a source that
is already a link is skipped.
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
E = 'runs/legs/gemma4-e2b'
S1 = 'studies/01-early-exploration'
S2 = 'studies/02-physics-trace-observation'
S3 = 'studies/03-round-1-single-pair-edits'
S4 = 'studies/04-round-2-overlap'
M = 'models/gemma4/gemma4-e2b'

MOVES = [
    # Hypothesis and journal
    ('research/theory/RESIDUAL-TRACE-THEORY.md', 'hypothesis/RESIDUAL-TRACE-THEORY.md'),
    ('research/theory/RESIDUAL-TRACE-HANDOFF.md', 'hypothesis/RESIDUAL-TRACE-HANDOFF.md'),
    ('research/strategies/README.md', 'hypothesis/TWO-RESEARCH-FRONTS.md'),
    ('research/diary', 'journal'),
    ('research/README.md', 'journal/RESEARCH-NOTEBOOK.md'),
    ('research/PEER-HANDOFF.md', 'journal/PEER-HANDOFF.md'),
    # 01 early exploration (Nov 5-6)
    (f'{E}/ghost-retrieval-2026-11-05', f'{S1}/ghost-retrieval/run'),
    (f'{E}/ghost-inspection-2026-11-05', f'{S1}/ghost-inspection/run'),
    ('research/studies/GHOST-RETRIEVAL-EXPERIMENT.md', f'{S1}/ghost-retrieval/STUDY.md'),
    ('research/studies/GHOST-INSTRUMENTATION.md', f'{S1}/ghost-inspection/STUDY.md'),
    (f'{E}/decode-observatory-30k-2026-11-05', f'{S1}/decode-observatory/run'),
    (f'{E}/behavior-pattern-20-2026-11-06', f'{S1}/behavior-pattern/run'),
    (f'{E}/distributed-pathways-reference-2026-11-06', f'{S1}/distributed-pathways/run'),
    (f'{E}/individual-edit-2026-11-06', f'{S1}/individual-weight-edit/run'),
    ('research/studies/INDIVIDUAL-WEIGHT-EXPERIMENT.md', f'{S1}/individual-weight-edit/STUDY.md'),
    (f'{E}/targeted-lookup-2026-11-06', f'{S1}/targeted-lookup/run'),
    ('research/studies/TARGETED-LOOKUP-EXPERIMENT.md', f'{S1}/targeted-lookup/STUDY.md'),
    (f'{E}/lookup-failure-family-2026-11-06', f'{S1}/lookup-failure-family/run'),
    (f'{E}/lookup-pattern-observation-2026-11-06', f'{S1}/lookup-pattern-observation/run'),
    ('code/lookup-pattern-observation-2026-11-06', f'{S1}/lookup-pattern-observation/code'),
    (f'{E}/llamacpp-lookup-comparison-2026-11-06', f'{S1}/llamacpp-lookup-comparison/run'),
    (f'{E}/llamacpp-preference-edit-2026-11-06', f'{S1}/llamacpp-preference-edit/run'),
    ('code/lookup-preference-2026-11-06', f'{S1}/llamacpp-preference-edit/code'),
    (f'{E}/long-context-capture-7263027c5-gpu1-2', f'{S1}/long-context-capture/run'),
    (f'{E}/mechanism-study-2026-11-06', f'{S1}/mechanism-study/run'),
    (f'{E}/path-inspection-2026-11-06', f'{S1}/path-inspection/run'),
    ('research/studies/WEIGHT-INVESTIGATIONS.md', f'{S1}/WEIGHT-INVESTIGATIONS.md'),
    ('research/studies/WEIGHT-SPACE-ORIENTATION.md', f'{S1}/WEIGHT-SPACE-ORIENTATION.md'),
    # 02 physics trace observation (before any edit)
    (f'{E}/physics-trace-2026-11-06', f'{S2}/physics-trace-pilot/run'),
    ('code/physics-trace-2026-11-06', f'{S2}/physics-trace-pilot/code'),
    ('research/studies/PHYSICS-TRACE-PILOT.md', f'{S2}/physics-trace-pilot/STUDY.md'),
    (f'{E}/physics-trace-bf16-2026-11-06', f'{S2}/physics-trace-bf16/run'),
    ('code/physics-trace-bf16-2026-11-06', f'{S2}/physics-trace-bf16/code'),
    ('research/studies/PHYSICS-TRACE-BF16.md', f'{S2}/physics-trace-bf16/STUDY.md'),
    (f'{E}/physics-trace-followup-2026-11-06', f'{S2}/physics-trace-followup/run'),
    ('code/physics-trace-followup-2026-11-06', f'{S2}/physics-trace-followup/code'),
    ('research/studies/PHYSICS-TRACE-FOLLOWUP.md', f'{S2}/physics-trace-followup/STUDY.md'),
    (f'{E}/physics-intervention-2026-11-07', f'{S2}/physics-intervention/run'),
    ('code/physics-intervention-2026-11-07', f'{S2}/physics-intervention/code'),
    ('research/studies/PHYSICS-TRACE-INTERVENTION.md', f'{S2}/physics-intervention/STUDY.md'),
    # 03 round one: single-pair edits A-D
    (f'{E}/physics-weight-edit-2026-11-07', f'{S3}/A-weight-edit/run'),
    ('code/physics-weight-edit-2026-11-07', f'{S3}/A-weight-edit/code'),
    ('research/studies/PHYSICS-TRACE-WEIGHT-EDIT.md', f'{S3}/A-weight-edit/STUDY.md'),
    (f'{M}/gguf-bf16-physics-weight-edit-2026-11-07', f'{M}/round-1/A-weight-edit'),
    (f'{E}/physics-edit-footprint-2026-11-07', f'{S3}/B-selective-footprint/run'),
    ('code/physics-edit-footprint-2026-11-07', f'{S3}/B-selective-footprint/code'),
    (f'{M}/gguf-bf16-physics-selective-edit-2026-11-07', f'{M}/round-1/B-selective-footprint'),
    (f'{E}/physics-connected-edit-2026-11-07', f'{S3}/C-connected-preservation/run'),
    ('code/physics-connected-edit-2026-11-07', f'{S3}/C-connected-preservation/code'),
    (f'{M}/gguf-bf16-physics-connected-edit-2026-11-07', f'{M}/round-1/C-connected-preservation'),
    (f'{E}/physics-background-sensitivity-2026-11-07', f'{S3}/D-background-sensitivity/run'),
    ('code/physics-background-sensitivity-2026-11-07', f'{S3}/D-background-sensitivity/code'),
    (f'{M}/gguf-bf16-physics-background-edit-2026-11-07', f'{M}/round-1/D-background-sensitivity'),
    (f'{E}/physics-multitrace-2026-11-07', f'{S3}/multitrace-observations/run'),
    ('code/physics-multitrace-2026-11-07', f'{S3}/multitrace-observations/code'),
    ('reports/paper-a-d', f'{S3}/paper'),
    ('reports/understanding-the-method', f'{S3}/method-guide'),
    # 04 round two: overlap
    ('research/experiments/signal-trace-repair/ROUND-2.md', f'{S4}/ROUND-2.md'),
    ('research/experiments/signal-trace-repair/round-2-method.json', f'{S4}/round-2-method.json'),
    ('research/AUTOMATION-LESSONS.md', f'{S4}/LESSONS-FROM-FIRST-ATTEMPT.md'),
    (f'{E}/signal-trace-A-2026-11-07/shared', f'{S4}/inputs'),
    (f'{E}/signal-trace-repair-2026-11-07', f'{S4}/inputs/recorder-qualification'),
    ('research/experiments/signal-trace-repair/shared', f'{S4}/inputs/docs'),
    (f'{E}/round-2', f'{S4}/runs'),
    ('research/experiments/signal-trace-repair/STEP-5-MEASUREMENT-DESIGN.md', f'{S4}/design-notes/STEP-5-MEASUREMENT-DESIGN.md'),
    ('research/experiments/signal-trace-repair/acquisition-proposal', f'{S4}/design-notes/acquisition-proposal'),
    ('research/experiments/signal-trace-repair/qualification.json', f'{S4}/inputs/docs/qualification.json'),
    # Early-study checkpoints
    (f'{M}/ghost-retrieval-2026-11-05', f'{M}/early-studies/ghost-retrieval'),
    (f'{M}/individual-edit-2026-11-06', f'{M}/early-studies/individual-weight-edit'),
    (f'{M}/targeted-lookup-2026-11-06', f'{M}/early-studies/targeted-lookup'),
    (f'{M}/gguf-first-preference-2026-11-06', f'{M}/early-studies/llamacpp-preference-edit'),
    # Publications and records
    ('reports/FINDINGS.md', 'publications/findings/FINDINGS.md'),
    ('reports/EXPERIMENTS.md', 'publications/findings/EXPERIMENTS.md'),
    ('reports/TABLES.md', 'publications/findings/TABLES.md'),
    ('blog', 'publications/blog'),
    ('figures', 'publications/figures'),
    ('provenance', 'records/provenance'),
    ('runs/maintenance', 'records/maintenance-runs'),
    ('runs/organization-2026-11-07', 'records/organization-2026-11-07'),
]

LINK = re.compile(r'(\]\(|href=")([^)"#\s]+)')


def mapped(path):
    path = os.path.normpath(path)
    best = None
    for src, dst in MOVES:
        s = os.path.normpath(str(ROOT / src))
        if path == s or path.startswith(s + os.sep):
            if best is None or len(s) > len(best[0]):
                best = (s, os.path.normpath(str(ROOT / dst)))
    return path if best is None else best[1] + path[len(best[0]):]


def fix_links(new_file, old_file):
    text = new_file.read_text(errors='ignore')
    def repl(m):
        link = m.group(2)
        if '://' in link or link.startswith(('mailto:', '/', 'data:')):
            return m.group(0)
        target = mapped(os.path.join(os.path.dirname(old_file), link))
        if not os.path.lexists(target):
            return m.group(0)
        rel = os.path.relpath(target, os.path.dirname(str(new_file)))
        return m.group(1) + rel
    fixed = LINK.sub(repl, text)
    if fixed != text:
        new_file.write_text(fixed)
        return 1
    return 0


def main(dry):
    moved, docs = [], []
    for src, dst in MOVES:
        s, d = ROOT / src, ROOT / dst
        if s.is_symlink() or not s.exists():
            continue
        if d.exists():
            raise FileExistsError(d)
        if dry:
            print('move', src, '->', dst)
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        os.rename(s, d)
        os.symlink(os.path.relpath(d, s.parent), s)
        moved.append((src, dst))
        for f in ([d] if d.is_file() else d.rglob('*')):
            if f.is_file() and not f.is_symlink() and f.suffix in {'.md', '.html'} and f.stat().st_size < 2_000_000:
                docs.append((f, str(s / f.relative_to(d)) if d.is_dir() else str(s)))
    fixed = sum(fix_links(f, old) for f, old in docs) if not dry else 0
    record = dict(moves=moved, documents_with_relinked_paths=fixed,
                  rule='same-filesystem rename; relative symlink left at every old path; recorded absolute paths unchanged')
    if not dry:
        out = ROOT / 'records/regroup-2026-11-09.json'
        out.write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps(dict(moved=len(moved), relinked_documents=fixed)))


if __name__ == '__main__':
    main(dry='--dry-run' in sys.argv)
