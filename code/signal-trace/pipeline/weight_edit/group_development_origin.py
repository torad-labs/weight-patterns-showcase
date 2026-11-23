"""Admit measured native development origins to the full learner."""

from pathlib import Path

from pipeline.weight_edit import common_prefix_objective as prefix, final_experiment as baseline

FORMAT = 'prefix-gradient-calibration-origin/1'
PRISTINE_FORMAT = 'pristine-stock-development-origin/1'


def checked_origin(binding, lineage, *, predecessor=None):
    path = prefix.checked(binding)
    origin = baseline.read(path)
    if origin.get('format') == PRISTINE_FORMAT:
        return checked_pristine(binding, path, origin, lineage, predecessor)
    if origin.get('format') != FORMAT:
        raise ValueError('Expected a materialized prefix calibration origin')
    checkpoint, measurement, legacy = (origin[key] for key in ('checkpoint', 'measurement', 'legacy_checkpoint'))
    source = prefix.checked(legacy)
    if source.name != 'final-state.json' or (predecessor is not None and source.parent != predecessor):
        raise ValueError('The calibration origin must name the declared legacy predecessor')
    observed_path = prefix.checked(measurement)
    state = baseline.read(prefix.checked(checkpoint))
    observed = baseline.read(observed_path)
    old = baseline.read(source)
    former = baseline.read(source.parent/'declaration.json')
    if (old['weight_state'] != 'complete' or old.get('pending_install') or old.get('pending_transition')
            or old['last_measurement'] != measurement or state['last_measurement'] != measurement
            or not observed['complete'] or not observed['ended']
            or observed['model_identity'] != old['model_identity'] or state['model_identity'] != old['model_identity']
            or observed['step'] != old['update'] or state['iteration'] != old['update']
            or observed_path != source.parent/'loop'/str(old['update'])/'measurement.json'
            or state['base_identity'] != old['base_seed'] or state['eta'] != old['eta'] or state['eta'] != lineage['eta']
            or state['loss_identity'] != lineage['loss_identity']
            or observed['reference_sha256'] != lineage['loss_identity']['reference_sha256']
            or observed['reference'] != lineage['reference'] or former['reference'] != lineage['reference']
            or former['original_model'] != lineage['original_model']):
        raise ValueError('Materialized checkpoint differs from the measured legacy origin or frozen loss identity')
    override_path = source.parent / old['overrides']['file']
    override = baseline.binding(override_path)
    if override['sha256'] != old['overrides']['sha256']:
        raise ValueError('The legacy native overlay differs from its exact source manifest')
    inputs = [binding, checkpoint, measurement, legacy, override, baseline.binding(source.parent/'declaration.json'),
              *origin.get('inputs', [])]
    for record in origin.get('inputs', []):
        prefix.checked(record)
    return dict(origin=binding, checkpoint=checkpoint, measurement=measurement, legacy_checkpoint=legacy,
                implementation=baseline.binding(Path(__file__)), inputs=inputs)


def checked_pristine(binding, path, origin, lineage, predecessor):
    folder = baseline.run_path(path.parent)
    if path.name != 'origin.json' or (predecessor is not None and folder != predecessor):
        raise ValueError('Pristine origin must be inside the declared predecessor run')
    declaration, checkpoint, measurement, capture, exit_record = (origin[key] for key in (
        'declaration', 'checkpoint', 'measurement', 'capture', 'exit_record'))
    paths = ('declaration.json', 'final-state.json', 'loop/0/measurement.json',
             'loop/0/capture/capture.json', 'launcher-exit.json')
    records = {}
    for source, filename in zip((declaration, checkpoint, measurement, capture, exit_record), paths, strict=True):
        current = prefix.checked(source)
        if current != folder / filename:
            raise ValueError(f'Pristine origin record points outside its own stage: {filename}')
        records[filename] = baseline.read(current)
    source, state, observed, captured, exited = (records[name] for name in paths)
    stock = lineage['original_model']
    stock_identity = dict(file=stock['record'], model_sha256=stock['sha256'])
    revision = dict(kind='native-working-revision/1', run=str(folder.relative_to(baseline.ROOT)),
                    generation=0, original_sha256=stock['sha256'])
    native_inputs = baseline.read(prefix.checked(lineage['native_inputs']))
    if (source['operation'] != 'joint-pristine-stock-origin'
            or source['original_model'] != stock or origin['stock_verified'] != stock
            or source['reference'] != lineage['reference'] or source['loss_identity'] != lineage['loss_identity']
            or source['native_reading'] != lineage['native_reading'] or source['eta'] != lineage['eta']
            or state['format'] != 'full-native-state/1' or state['status'] != 'complete'
            or state['base_identity'] != stock_identity or state['model_identity'] != revision
            or state['iteration'] != 0 or state.get('transition') is not None or state['eta'] != lineage['eta']
            or state['last_measurement'] != measurement or state['loss_identity'] != lineage['loss_identity']
            or observed['model_identity'] != revision or observed['loss_identity'] != lineage['loss_identity']
            or observed['step'] != 0 or observed['generation'] != 0 or observed['previous_measurement'] is not None
            or observed['complete'] is not True or observed['ended'] is not True or observed['merges'] != 105
            or observed['member'] != native_inputs['member'] or observed['capture'] != capture
            or captured['model_identity'] != revision or captured['cases'] != source['cases']
            or exited != dict(status='completed_pristine_measurement', installed_updates=0,
                              measurement=measurement, checkpoint=checkpoint)):
        raise ValueError('Pristine origin is not the measured unedited stock baseline of this learner')
    manifest_path = prefix.checked(state['checkpoint'])
    if manifest_path != folder / 'final-checkpoint/state.json':
        raise ValueError('Pristine origin refers to another checkpoint')
    manifest = baseline.read(manifest_path)
    identity_fields = ('format', 'status', 'iteration', 'model_identity', 'base_identity', 'inventory_identity', 'transition')
    if (manifest['complete'] is not True or any(manifest[key] != state[key] for key in identity_fields)
            or manifest['remainder']['file'] != 'remainder.f32' or manifest['remainder']['bytes'] != 0
            or manifest['remainder']['sha256'] != baseline.digest(manifest_path.with_name('remainder.f32'))
            or manifest_path.with_name('remainder.f32').stat().st_size != 0
            or not manifest['layout'] or any(row['remainder'] != {'kind': 'zero'} for row in manifest['layout'])
            or manifest['native']['file'] != 'native.bin'
            or manifest['native']['bytes'] != manifest_path.with_name('native.bin').stat().st_size):
        raise ValueError('Pristine origin has an incomplete native payload or nonzero remainder')
    inputs = [binding, declaration, checkpoint, measurement, capture, exit_record, source['cases']]
    for source_binding in [source['cases'], origin['program']]:
        prefix.checked(source_binding)
    if source['program'] != origin['program']:
        raise ValueError('Pristine origin program differs from its acquisition declaration')
    return dict(origin=binding, checkpoint=checkpoint, measurement=measurement, pristine_exit=exit_record,
                implementation=baseline.binding(Path(__file__)), inputs=inputs)
