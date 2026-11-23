"""A frozen proposal's native reading against a field its sweep was NOT solved against: a separately labeled diagnostic.

  python -m pipeline.weight_edit.fixed_reference_compare <run folder>

Astra's entry point (whole_context_compare.main) scores a swept model against the field its sweep was solved against and refuses any
other. The recipient-prefix specification (records/retrospectives/2026-11-14-recipient-prefix-target-astra.md) also asks for the same
final reading against the unchanged matched field, with no second capture: what changing the target changed, read on one fixed
reference. This program is that second comparison and nothing else.

The arithmetic and every check of the reading, the protocol, the correspondence and the cached control are Astra's, called unchanged:
whole_context_compare.compare, comparison_protocol and verify_correspondence_files, and whole_context_score.NativeReading. What differs
from the entry point is what the declaration must say (`fixed_reference_comparison`, beside `native_comparison`):

  the reference field is ANOTHER field than the sweep's own, named with its manifest's sha256, and the sweep's frozen record must name
      the primary target, not this one;
  the reading, the capture, the reader protocol, the frozen sweep record, the supplied history and the terminal normalization are the
      primary comparison's own, so no second capture or second protocol can enter;
  the correspondence record, the cached control score and the output folder are this comparison's own, since each is bound to the
      field it is of.

It reports and accepts nothing. It is no choice between two fields' totals, which are not comparable, and no regrading of the run the
reference field was solved in."""
import hashlib
import json
import sys
from pathlib import Path

from pipeline.native_records.native_trace import checkpoint_values
from pipeline.native_records.run_declaration import check_native_run
from pipeline.paths import ROOT
from pipeline.weight_edit import context_targets as c, gap_targets, whole_context_compare as w
from pipeline.weight_edit.whole_context_score import NativeReading

SHARED = ('protocol', 'sweep_record', 'candidate_reading', 'capture', 'history_before', 'normalization')
OWN = ('target', 'target_sha256', 'control_score', 'correspondence', 'output')
LABEL = ("Fixed-reference diagnostic: this model's sweep was solved against another field. The sums are this model's distance from the reference "
         "field named here, beside the original's under that same field; they are not the run's primary comparison.")


def held(asked, primary, frozen, model_sha256):
    """The declaration's two comparison blocks and the frozen sweep record say what this diagnostic needs; a ValueError names what does not."""
    for name in SHARED:
        if name not in asked or asked[name] != primary.get(name):
            raise ValueError(f"The diagnostic's {name} is not the primary comparison's: no second capture, protocol or normalization may enter")
    for name in OWN:
        if not asked.get(name) or asked[name] == primary.get(name):
            raise ValueError(f"The diagnostic's {name} must be its own, bound to the reference field, not the primary comparison's: "
                             "the sweep's own field is the entry point's to compare")
    if frozen['sweep']['original_sha256'] != model_sha256 or frozen['sweep']['target_sha256'] != primary['target_sha256']:
        raise ValueError("The frozen sweep is not the declared sweep from the original against the run's primary target")
    names = frozen['sweep']['matrices']
    if not names or [m['matrix'] for m in frozen['matrices']] != names:
        raise ValueError('A partial sweep is not the declared completed proposal')
    last = frozen['last_state']
    if last['every_other_byte_is_the_originals'] is not True or frozen['matrices'][-1]['state_after'] != last['identity']:
        raise ValueError('Frozen final state is not the sweep records final state')
    return last


def main(folder):
    run = Path(folder).resolve()
    if not run.is_relative_to(ROOT/'studies/04-round-2-overlap/runs'):
        raise ValueError('The diagnostic must belong to a declared run')
    declaration = check_native_run(run, run/'declaration.json')
    if declaration['method_step'] != 5:
        raise ValueError('The diagnostic belongs to a step-5 construction declaration')
    asked, primary = declaration['fixed_reference_comparison'], declaration['native_comparison']
    for binding in asked['implementation']:
        if c.binding(ROOT/binding['record'], ROOT) != binding:
            raise ValueError('Diagnostic implementation changed since declaration')
    output = w._output(run, asked['output'])
    if output.exists():
        raise FileExistsError('Preserve an existing or partial diagnostic')
    freeze_path = w._output(run, asked['sweep_record'])
    frozen = json.loads(freeze_path.read_text())
    last = held(asked, primary, frozen, declaration['model_sha256'])
    targets = gap_targets.read(w._source(asked['target'], declaration), ROOT)
    if targets.sha256 != asked['target_sha256']:
        raise ValueError('Reference field differs from the declaration')
    if frozen['sweep']['fitting'] != [m['id'] for m in targets.answers]:
        raise ValueError("The reference field's answers are not the sweep's fitting answers")
    source_protocol = w._bound(asked['protocol'], declaration)
    if source_protocol['original_control']['model_sha256'] != declaration['model_sha256']:
        raise ValueError('Reader protocol is not bound to the declared original')
    for name, digest in source_protocol['instruments'].items():
        if c.sha256(ROOT/name) != digest:
            raise ValueError(f'Acquisition implementation changed since protocol: {name}')
    control = NativeReading(w._source(source_protocol['original_control']['reading'], declaration))
    cached = w._bound(asked['control_score'], declaration)
    candidate = NativeReading(w._output(run, asked['candidate_reading']))
    protocol = w.comparison_protocol(source_protocol, asked['protocol']['sha256'], control, candidate, targets.sha256, last['model_sha256'], asked['history_before'])
    mapping_path = w._output(run, asked['correspondence'])
    correspondence = json.loads(mapping_path.read_text())
    w.verify_correspondence_files(correspondence, w._output(run, asked['capture']), candidate)
    normal = asked['normalization']
    tensors, reference = w._bound(normal['tensors'], declaration), w._bound(normal['reference'], declaration)
    terminal = reference['terminal']
    model = w._source(normal['model'], declaration)
    weight = checkpoint_values(model, tensors[terminal['output_norm']]).astype('<f8').ravel()
    if hashlib.sha256(weight.tobytes()).hexdigest() != terminal['output_norm_sha256']:
        raise ValueError('Original output normalization changed')
    name, scale = next(iter(terminal['layer_output_scale'].items()))
    if checkpoint_values(model, tensors[name]).ravel().tolist() != [scale]:
        raise ValueError('Original terminal scale changed')
    report, native = w.compare(control, candidate, targets, dict(weight=weight, scale=scale, eps=terminal['eps']), protocol, cached, correspondence)
    c.write_json(output/'candidate-score.json', native)
    report['label'] = LABEL
    report['bindings'] = dict(declaration=c.binding(run/'declaration.json', ROOT), sweep=c.binding(freeze_path, ROOT), protocol=asked['protocol'],
                              control_reading_sha256=control.sha256, control_score=asked['control_score'], candidate_reading_sha256=candidate.sha256,
                              reference_target_manifest_sha256=targets.sha256, the_sweeps_own_target_manifest_sha256=primary['target_sha256'],
                              correspondence=c.binding(mapping_path, ROOT), candidate_score=c.binding(output/'candidate-score.json', ROOT))
    c.write_json(output/'result.json', report)
    print(json.dumps(dict(diagnostic='fixed reference', complete=report['prediction_vs_result']['complete'], groups=report['prediction_vs_result']['groups'],
                          changed_histories=len(report['prediction_vs_result']['changed_histories']), result=str((output/'result.json').relative_to(ROOT)))), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
