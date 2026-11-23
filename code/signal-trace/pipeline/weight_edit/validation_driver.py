"""Synchronous development evaluation at declared boundaries of the installed-weight learner."""
from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import signal
import sys

from pipeline.weight_edit import final_experiment as baseline
from pipeline.weight_edit.validation_stopping import Policy, numeric

METRIC = 'aligned-reference-turn-rms/1'


def checked(row):
    path = (baseline.ROOT / row['record']).resolve()
    if 'reserved-40b' in path.parts:
        raise ValueError('the sealed evaluation population cannot calibrate stopping')
    if baseline.binding(path)['sha256'] != row['sha256']:
        raise ValueError(f'changed stopping input {path}')
    return path


def read(row):
    return baseline.read(checked(row))


def prompt_digest(tokens):
    return hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()


def load_policy(binding, dec):
    inventory = {}
    def bound(row):
        path = checked(row)
        inventory[row['record']] = row
        return path
    def load(row):
        return baseline.read(bound(row))

    spec = load(binding)
    from pipeline.weight_edit import trace_stopping
    if spec['format'] == trace_stopping.FORMAT:
        return trace_stopping.register(spec, dec, bound, load, inventory)
    if spec['format'] != 'development-operation-stopping/1':
        raise ValueError('unknown stopping policy format')
    calibration = load(spec['calibration'])
    records = [load(row) for row in calibration['fit_history']]
    if (not records or len({r['model_identity']['run'] for r in records}) != 1
            or len({r['capture']['sha256'] for r in records}) != len(records)):
        raise ValueError('the fit history needs distinct acquisitions of one consecutive trajectory')
    first = records[0]['generation']
    for i, r in enumerate(records):
        capture = load(r['capture'])
        if (r['generation'] != first+i or r['model_identity']['generation'] != first+i
                or not (r['complete'] and r['ended']) or r['loss_identity'] != dec['loss_identity']
                or capture['model_identity'] != r['model_identity']
                or r['member'] not in dec['functional_loop']['training_members']):
            raise ValueError('historical fit calibration changed identity, coverage or sequence')
    def observations(rows, mode, label, prompt, operation=None):
        samples = [load(row) for row in rows]
        if not samples or len({s['capture']['sha256'] for s in samples}) != len(samples):
            raise ValueError(f'{mode} requires distinct captures')
        if mode == 'repeat' and len({s['weights']['sha256'] for s in samples}) != 1:
            raise ValueError('repeatability must come from unchanged weights')
        for s in samples:
            capture, weights = load(s['capture']), load(s['weights'])
            if capture['model_identity'] != weights['model_identity']:
                raise ValueError('the calibration capture did not use its bound checkpoint')
            cases = load(capture['cases'])
            if len(cases) != 1 or prompt_digest(cases[0]['tokens']) != s['prompt_sha256']:
                raise ValueError('calibration prompt differs from its acquisition')
            if (s['complete'] is not True or s['operation'] != operation or s['instrument'] != label
                    or s['family'] in held):
                raise ValueError('calibration observation differs in coverage, operation or instrument')
            if mode == 'qualified_correct' and s['status'] != 'right':
                raise ValueError('correct range requires locally correct calibration observations')
            if mode == 'qualified_wrong' and s['status'] != 'wrong':
                raise ValueError('failure calibration requires locally incorrect observations')
        if mode == 'repeat' and {s['prompt_sha256'] for s in samples} != {prompt}:
            raise ValueError('repeatability must repeat the same prompt')
        return [numeric(s['value']) for s in samples]

    fitted = dec['functional_loop']['training_members']
    if len(fitted) != 1 or spec['fit']['id'] != fitted[0]:
        raise ValueError('stopping panel must identify the admitted fit')
    held = set(spec['held_families'])
    if not held or spec['fit']['family'] in held:
        raise ValueError('declare distinct final-test family identities')
    scorer = spec['scorer']
    bound(scorer)
    for source in spec['scorer_dependencies']:
        bound(source)
    operations, labels, prompts, inputs = {}, {}, {}, {tuple(spec['fit']['tokens'])}
    if not inputs or not spec['fit']['tokens']:
        raise ValueError('missing fitted prompt identity')
    ids = set(fitted)
    for member in spec['panel']:
        cases = load(member['cases'])
        if len(cases) != 1 or cases[0]['id'] != member['id'] or set(cases[0]) != {'id','tokens','max_tokens'}:
            raise ValueError('each selector case file must contain its one named native prompt')
        tokens = tuple(cases[0]['tokens'])
        limit = cases[0]['max_tokens']
        if (type(limit) is not int or limit <= 0 or len(tokens)+limit > dec['native_context']
                or any(type(t) is not int or t < 0 for t in tokens)):
            raise ValueError('selection case exceeds the native context or has invalid inputs')
        if (not tokens or tokens in inputs or member['id'] in ids or Path(member['id']).name != member['id']
                or member['id'] in ('.','..') or member['family'] in held
                or member['family'] == spec['fit']['family']):
            raise ValueError('duplicated example or overlapping development and held/fit families')
        inputs.add(tokens)
        ids.add(member['id'])
        for item in member['operations']:
            name = item['name']
            key = member['id'] + '/' + name
            if key in operations or not name or not member['id']:
                raise ValueError('operation identities must be unique')
            region = load(item['reference'])
            bound(region['turns'])
            cells = region['cells']
            if (not cells or any(not isinstance(c,list) or len(c) != 2 or
                                 any(type(i) is not int or i < 0 for i in c) for c in cells)
                    or len({tuple(c) for c in cells}) != len(cells)):
                raise ValueError('each operation needs a nonempty, unique set of frozen reference cells')
            operations[key] = item
            labels[key] = {'metric':METRIC,'scorer': scorer, 'reference': item['reference']}
            prompts[key] = prompt_digest(cases[0]['tokens'])
    if not {'earlier_repair', 'descent', 'friction'} <= {o['capability'] for o in operations.values()}:
        raise ValueError('declare earlier repairs, descent and friction in the development panel')
    if not any(o['capability'] == 'friction' and o['role'] == 'retain' for o in operations.values()):
        raise ValueError('include a valid friction subtraction as a preservation control')
    if set(calibration['operation']) != set(operations):
        raise ValueError('calibration operation inventory differs from panel')
    values = {k: {mode: observations(calibration['operation'][k][mode],mode,labels[k],prompts[k],k)
                  for mode in ('repeat','qualified_correct')} for k in operations}
    for k, item in operations.items():
        if item['role'] == 'repair':
            wrong = observations(calibration['operation'][k]['qualified_wrong'],'qualified_wrong',labels[k],prompts[k],k)
            if max(wrong) <= max(values[k]['qualified_correct']) + max(values[k]['repeat'])-min(values[k]['repeat']):
                raise ValueError(f'{k} has no calibrated failing example beyond its correct range')
    policy = Policy(dict(fit_history=[r['objective'] for r in records],
                         fit_repeat=observations(calibration['fit_repeat'],'repeat',dec['loss_identity'],
                                                 prompt_digest(spec['fit']['tokens'])),
                         operation=values), operations)
    policy.bindings = [inventory[k] for k in sorted(inventory)]
    return spec, policy


def local_distances(app, raw, events, merges, width, operations):
    import numpy as np
    import torch
    from pipeline.overlap.complete_overlap import distance_path, unit_steps

    with torch.no_grad():
        native = app.held.raw_forward(raw,merges,events,width)
        current = np.stack([t.detach().cpu().numpy() for t in
                            app.turn_pair.turns(native,expected_merges=len(merges))],axis=1)
    normalized = unit_steps(current)
    by_trace = {}
    for op in operations:
        region = read(op['reference'])
        key = tuple(region['turns'][k] for k in ('record','sha256'))
        by_trace.setdefault(key,[]).append((op['name'],region['cells']))
    result = {}
    for (path,sha), rows in by_trace.items():
        reference = np.load(checked(dict(record=path,sha256=sha)),mmap_mode='r',allow_pickle=False)
        if reference.ndim != 3 or reference.shape[1:] != current.shape[1:]:
            raise ValueError('selection trace and reference shape differ')
        pairs = distance_path(unit_steps(reference) @ normalized.T)
        for name,cells in rows:
            losses = []
            for e,m in cells:
                if type(e) is not int or type(m) is not int or not (0 <= e < len(reference) and 0 <= m < len(merges)):
                    raise ValueError('operation lies outside the declared reference trace')
                matched = pairs[pairs[:,0] == e,1]
                if not len(matched):
                    raise ValueError('alignment omitted a declared reference event')
                deltas = current[matched,m] - reference[e,m]
                losses.append(float(np.mean(np.square(deltas).sum(axis=-1))))
            result[name] = float(np.sqrt(np.mean(losses)))
    return result


class Boundary:
    def __init__(self, app, run, spec, policy, assessor):
        self.app, self.run, self.spec, self.policy, self.assessor = app, run, spec, policy, assessor
        self.terminal = False
        self.state = None
        self.selected = None

    def panel(self, n, args):
        out = {}
        app = self.app
        if args['case']['tokens'] != self.spec['fit']['tokens']:
            raise ValueError('declared fit and actual acquisition prompt differ')
        for member in self.spec['panel']:
            folder = self.run / 'validation' / str(n) / member['id']
            folder.mkdir(parents=True)
            prompt = read(member['cases'])[0]
            # Allocate each acquisition only while the full exit reserve remains.
            app.reserve(self.run, args['budget'])
            with args['recorder'].context(measurement_role='development_selection', validation_member=member['id'],
                                          model_generation=n), args['recorder'].span('native.capture', detached=True):
                evidence = app.factory.capture(args['fd'], folder/'capture', args['model_identity'],
                    checked(member['cases']), args['engine'], dict(args['inputs'], member=member['id']))
            raw = folder / 'capture/raw' / member['id']
            answer = app.legacy.native_events.answer_of(prompt, baseline.read(raw/'result.json'), prompt['max_tokens'])
            if not answer['ended_by_itself']:
                observations = {op['name']: dict(distance=None, status='unscorable') for op in member['operations']}
            else:
                # A reviewed, domain-specific scorer reads ONLY fresh selection
                # data and the fixed operation references. It never receives the
                # model fd, gradient, predictions or final-test answers.
                labels = self.assessor.score(raw, answer, member['operations'])
                distances = local_distances(app,raw,len(prompt['tokens'])+answer['steps']-1,
                                            args['reading']['merges'],args['reference'].width,member['operations'])
                observations = {name: dict(status=label,distance=distances[name]) for name,label in labels.items()}
            if set(observations) != {o['name'] for o in member['operations']}:
                raise ValueError('validation scorer omitted or added operations')
            app.save(folder/'assessment.json', dict(model_identity=evidence['model_identity'],
                capture=app.binding(folder/'capture/capture.json'), scorer=self.spec['scorer'],
                case=member['id'], observations=observations))
            out.update({member['id']+'/'+name: row for name,row in observations.items()})
        return out

    def after_acquire(self, args, result):
        if not result[-2]:
            if hasattr(self.policy, 'trace_ids'):
                self.app.save(self.run/'validation-decision.json',
                    dict(generation=len(self.policy.fit), terminal=True, decision='incomplete_fitted_acquisition',
                         capability_success='unmeasured',durability='unmeasured', measurement=result[5],
                         selected=self.selected))
                self.terminal = True
                signal.raise_signal(signal.SIGUSR1)
            return
        local = args['number']
        if local == 0 and self.policy.fit:
            # This is a fresh replication on the restored bytes. The historical
            # controller uses it for its next proposal. It consumes no update.
            return
        n = len(self.policy.fit)
        panel = None
        if self.policy.due(n):
            if hasattr(self.policy, 'trace_ids'):
                from pipeline.weight_edit import trace_stopping
                panel = trace_stopping.observe_panel(self, args, result)
            else:
                panel = self.panel(local,args)
        decision = self.policy.observe(n, result[3]['objective'], panel)
        if decision and decision['promote']:
            if n == 0:
                self.selected = {'generation': 0, 'state': self.declaration['resume_checkpoint']}
            elif decision['terminal']:
                # The historical controller writes this exact current state on orderly exit.
                self.selected = {'generation': n, 'exit_state': 'final-state.json',
                                 'exit_checkpoint': 'final-checkpoint/state.json'}
            else:
                # A selected revision uses one checkpoint across the whole model.
                app = self.app
                native_size = args['budget']['native_checkpoint_bytes']
                remainder = args['budget']['remainder_checkpoint_bytes']
                app.reserve(self.run, {'free_required_bytes': args['budget']['free_required_bytes']+native_size+remainder})
                checkpoint = self.run/'validation'/str(local)/'selected-checkpoint'
                self.state.checkpoint(checkpoint)
                manifest = app.binding(checkpoint/'state.json')
                receipt = checkpoint.parent/'selected-state.json'
                app.save(receipt,dict(self.state.receipt(), checkpoint=manifest,last_measurement=result[5],
                                      loss_identity=self.declaration['loss_identity'],eta=self.declaration['eta']))
                self.selected = {'generation': n, 'checkpoint': manifest,'state':app.binding(receipt)}
        source=self.declaration['validation_stopping']['policy']
        self.app.save(self.run/'loop'/str(local)/'selection-observation.json',
                      dict(generation=n, measurement=result[5],panel=panel,policy=source,selected=self.selected))
        if decision is None:
            return
        record = dict(decision, measurement=result[5], selected=self.selected,
                      policy=self.declaration['validation_stopping']['policy'])
        self.app.save(self.run/'validation'/str(local)/'decision.json', record)
        if decision['terminal']:
            self.app.save(self.run/'validation-decision.json', record)
            self.terminal = True
            signal.raise_signal(signal.SIGUSR1)

    @contextmanager
    def connected(self, declaration):
        self.declaration = declaration
        acquire, restore, archive = self.app.acquire, self.app.continuation.restore, self.app.archive_source.archive_previous
        self.fit_acquire = acquire
        @wraps(acquire)
        def measured(*args, **kwargs):
            bound = inspect.signature(acquire).bind(*args,**kwargs).arguments
            result = acquire(*args,**kwargs)
            try:
                self.after_acquire(bound,result)
            except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
                # Preserve the just-written observation as the controller's last
                # measured state even when the independent panel cannot be read.
                self.app.save(self.run/'validation-decision.json',
                    dict(generation=bound['number'],terminal=True,decision='incomplete_validation',
                         error=str(error),measurement=result[5],selected=self.selected))
                self.terminal = True
                signal.raise_signal(signal.SIGUSR1)
            return result
        @wraps(restore)
        def restored(*args, **kwargs):
            self.state = restore(*args,**kwargs)
            return self.state
        @wraps(archive)
        def archived(*args, **kwargs):
            if self.terminal:
                return dict(archives=[], phase='stopping', retained=True)
            return archive(*args,**kwargs)
        self.app.acquire, self.app.continuation.restore, self.app.archive_source.archive_previous = measured, restored, archived
        try:
            yield
        finally:
            self.app.acquire, self.app.continuation.restore, self.app.archive_source.archive_previous = acquire, restore, archive


def replay(policy, declaration):
    selected = None
    for i,bound in enumerate(declaration.get('history',[])):
        event = read(bound)
        measured = read(event['measurement'])
        if (event['policy'] != declaration['policy'] or event['generation'] != i
                or not (measured['ended'] and measured['complete'])):
            raise ValueError('stopping history is not contiguous or uses a different policy')
        if hasattr(policy, 'trace_ids') and event['panel'] is not None:
            for name,row in event['panel'].items():
                observation = read(row['measurement'])
                if (observation['objective'] != row['objective'] or observation['member'] != name
                        or observation['model_identity'] != measured['model_identity']):
                    raise ValueError('development history differs from its measured trajectory')
                checked(row['initial_inside'])
                for key in ('local_maps','repeat_measurement','repeat_maps'):
                    if key in row:
                        checked(row[key])
        decision = policy.observe(i,measured['objective'],event['panel'])
        if decision and decision['terminal']:
            raise ValueError('a terminal policy cannot restart its allowance')
        if decision and decision['promote']:
            selected = event['selected']
            if selected['generation'] != i:
                raise ValueError('selected checkpoint differs from the stopping history')
            checked(selected['state'])
        if event['selected'] != selected:
            raise ValueError('selected checkpoint changed without a promotion')
    return selected


def main():
    app = baseline.controller(baseline.MEASURED)
    run = baseline.run_path(Path(sys.argv[1]))
    dec = app.held.check_native_run(run, run/'declaration.json')
    if dec.get('retired_input_admission'):
        from pipeline.weight_edit import retired_admission
        retired_admission.bind(app,dec)
    stopping = dec.get('validation_stopping')
    boundary = None
    if stopping:
        spec, policy = load_policy(stopping['policy'], dec)
        assessor = (app.load_module('validation_stage_assessor', checked(spec['scorer']))
                    if spec['format'] == 'development-operation-stopping/1' else None)
        boundary = Boundary(app,run,spec,policy,assessor)
        boundary.selected = replay(policy,stopping)
    diagnostic = None
    if dec.get('trace_diagnostic'):
        from pipeline.weight_edit import trace_diagnostic
        diagnostic = trace_diagnostic.from_declaration(app,run,dec)
    capture_source = app.load_module('final_experiment_capture', app.SELF/'capture.py')
    app.archive_source = app.load_module('final_experiment_archive', app.SELF/'archive.py')
    recorder = app.Recorder(run/'observability/events.jsonl', dict(project='weight-patterns',
        folder=run.name, run=str(run.relative_to(app.ROOT)), operation=dec['operation'],
        entrypoint=(stopping or dec['trace_diagnostic'])['driver']['record'],
        declaration_sha256=app.binding(run/'declaration.json')['sha256'],
        original_model_sha256=dec['original_model']['sha256'],reference_sha256=dec['reference']['sha256'],
        tolerance_sha256=dec['tolerance']['sha256'],resumed_checkpoint_sha256=dec['resume_checkpoint']['sha256'],
        model_identity_kind=app.held.working_identity.KIND, model_identity_run=str(run.relative_to(app.ROOT)),
        capture_schema=capture_source.FORMAT))
    capture = capture_source.Capture(recorder)
    completion = dict(status='FAILED',reason='before execution')
    try:
        with recorder.span('research.invocation', detached=True) as outcome:
            with (boundary.connected(dec) if boundary else nullcontext()), \
                 (diagnostic.connected() if diagnostic else nullcontext()), capture.instrument(vars(app)):
                app.execute(run,dec,recorder,capture,completion)
            if boundary and boundary.terminal:
                receipt = baseline.read(run/'validation-decision.json')
                completion['validation_decision'] = receipt['decision']
                if receipt.get('selected'):
                    completion['validation_selected_generation'] = receipt['selected']['generation']
            outcome.update(app.legacy.capture_source.scalars(completion,'output.'))
    finally:
        capture.coverage(run/'observability/coverage.json')
        recorder.close(completion.pop('status'), **completion)


if __name__ == '__main__':
    main()
