"""Measure whole regenerated development trajectories with the installed learner's objective."""
from pathlib import Path
from statistics import median

from .validation_stopping import numeric

FORMAT = 'whole-trace-development-stopping/1'
METRICS = ('objective', 'newly_outside_loss')


def register(spec, dec, bound, load, inventory):
    fit = spec['fit']
    held = set(spec['held_families'])
    if (not held or not all(isinstance(f, str) and f for f in held)
            or not fit['family'] or fit['family'] in held
            or dec['functional_loop']['training_members'] != [fit['id']]):
        raise ValueError('declare a separate fit, development panel and held families')
    if set(spec) != {'format', 'fit', 'held_families', 'panel', 'calibration'}:
        raise ValueError('whole-trace stopping takes frozen trajectories and a calibration history')
    if (set(fit) != {'id', 'family', 'tokens'} or not fit['tokens'] or not fit['id']
            or fit['id'] in ('.', '..') or Path(fit['id']).name != fit['id']):
        raise ValueError('the fit must identify its native prompt')
    ids = [fit['id']]
    tokens = {tuple(fit['tokens'])}
    for case in spec['panel']:
        if set(case) not in ({'id', 'family', 'cases'}, {'id', 'family', 'cases', 'target'}):
            raise ValueError('each development member takes a native prompt and optional frozen target')
        name = case['id']
        if (not isinstance(name,str) or not name or Path(name).name != name or name in ('.', '..')
                or name in ids or not case['family'] or case['family'] in held | {fit['family']}):
            raise ValueError('duplicate member or overlap with fit or held families')
        item = load(case['cases'])
        if len(item) != 1 or set(item[0]) != {'id', 'tokens', 'max_tokens'} or item[0]['id'] != name:
            raise ValueError('a development prompt must be the single declared native case')
        prompt, limit = item[0]['tokens'], item[0]['max_tokens']
        if (not prompt or tuple(prompt) in tokens or any(type(t) is not int or t < 0 for t in prompt)
                or type(limit) is not int or limit < 1 or len(prompt)+limit > dec['native_context']):
            raise ValueError('missing, duplicate or invalid native prompt')
        ids.append(name)
        tokens.add(tuple(prompt))
        if 'target' in case:
            target = case['target']
            if set(target) != {'reference', 'tolerance', 'tolerance_policy'}:
                raise ValueError('bind each alternative whole reference and its fixed tolerance')
            reference = load(target['reference'])
            if reference['source_model_sha256'] != dec['model_sha256']:
                raise ValueError('the development reference belongs to a different original model')
            bound(reference['trace'])
            bound(reference['source_definition'])
            if target['tolerance_policy'] == 'retained-original-chance':
                if target['tolerance'] != reference['chance']:
                    raise ValueError('development tolerance differs from the original chance')
                bound(target['tolerance'])
            else:
                tolerance = load(target['tolerance'])
                if tolerance['reference'] != target['reference'] or tolerance['status'] != 'reviewed':
                    raise ValueError('development tolerance differs from its reviewed reference')
                bound(tolerance['chance'])
    if len(ids) < 2:
        raise ValueError('at least one separate development prompt is required')
    calibration = load(spec['calibration'])
    if set(calibration) != {'fit_history'}:
        raise ValueError('retain the historical sequence of fitted native measurements')
    records = [load(r) for r in calibration['fit_history']]
    if (len(records) < 3 or len({r['model_identity']['run'] for r in records}) != 1
            or len({r['capture']['sha256'] for r in records}) != len(records)):
        raise ValueError('calibrate on consecutive distinct fresh native acquisitions')
    first = records[0]['generation']
    for i, row in enumerate(records, first):
        capture = load(row['capture'])
        cases = load(capture['cases'])
        if len(cases) != 1 or cases[0]['id'] != fit['id'] or cases[0]['tokens'] != fit['tokens']:
            raise ValueError('the calibration acquisition used a different fitted prompt')
        if (row['generation'] != i or row['model_identity']['generation'] != i or row['member'] != fit['id']
                or not (row['ended'] and row['complete']) or row['loss_identity'] != dec['loss_identity']
                or capture['model_identity'] != row['model_identity']):
            raise ValueError('fit calibration changed loss, prompt, completion or model identity')
    policy = TracePolicy(ids, [r['objective'] for r in records])
    policy.bindings = [inventory[k] for k in sorted(inventory)]
    return spec, policy


class TracePolicy:
    def __init__(self, ids, history):
        self.trace_ids = tuple(ids)
        if len(self.trace_ids) < 2 or len(set(self.trace_ids)) != len(self.trace_ids):
            raise ValueError('the whole-trace policy needs the fit and distinct development trajectories')
        self.fit_id = ids[0]
        history = [numeric(x) for x in history]
        if len(history) < 3:
            raise ValueError('at least two historical installed updates are required')
        best = history[0]
        recovered = []
        dry = 0
        for value in history[1:]:
            if value < best:
                recovered.append(dry)
                dry, best = 0, value
            else:
                dry += 1
        self.has_recovery = any(recovered)
        # Without a recovered reversal, measure every update and use budget-only stopping.
        self.cadence = 1 + max(recovered, default=0)
        self.budget = len(history)-1
        h = self.cadence
        changes = [median(history[i-h:i]) - median(history[i:i+h]) for i in range(h,len(history)-h+1)]
        self.variation = (median(abs(x-median(changes)) for x in changes) if len(changes) >= 2 else None)
        self.fit = []
        self.panels = []
        self.selected = None
        self.resolution = {}

    def due(self, n):
        return n % self.cadence == 0 or n == self.budget

    def gains(self, before, after, ids):
        return [(name, metric) for name in ids for metric in METRICS
                if before[name][metric] - after[name][metric] > self.resolution[name][metric]]

    def observe(self, n, loss, panel=None):
        if n != len(self.fit) or n > self.budget or (self.panels and self.panels[-1]['terminal']):
            raise ValueError('each installed generation must appear once in order')
        self.fit.append(numeric(loss))
        if not self.due(n):
            if panel is not None:
                raise ValueError('unexpected development observation')
            return None
        if panel is None or set(panel) != set(self.trace_ids):
            raise ValueError('missing a trajectory at the evaluation boundary')
        row = dict(generation=n, traces=panel, decision='continue', promote=False,
                   capability_success='unmeasured', durability='unmeasured')
        if any(not item['complete'] for item in panel.values()):
            row['decision'] = 'incomplete_trace_evaluation'
        else:
            for name, item in panel.items():
                for metric in METRICS:
                    numeric(item[metric])
                if n == 0:
                    self.resolution[name] = {metric: numeric(item['repeat_range'][metric]) for metric in METRICS}
            if panel[self.fit_id]['objective'] != loss:
                raise ValueError('fitted observation differs from the native learner')
            if self.selected is None:
                self.selected, row['promote'] = row, True
            else:
                prior = self.selected['traces']
                row['gained_since_selection'] = self.gains(prior,panel,self.trace_ids)
                row['worsened_since_selection'] = self.gains(panel,prior,self.trace_ids)
                if row['gained_since_selection'] and not row['worsened_since_selection']:
                    self.selected, row['promote'] = row, True
            h = self.cadence
            if self.has_recovery and self.variation is not None and n >= 2*h and n % h == 0:
                prior, previous = self.panels[-2]['traces'], self.panels[-1]['traces']
                development = self.trace_ids[1:]
                # A cumulative gain can resolve even when each separate interval does not.
                gains = (self.gains(prior,previous,development) + self.gains(previous,panel,development)
                         + self.gains(prior,panel,development))
                row['development_gains_over_two_intervals'] = sorted(set(gains))
                gain = median(self.fit[-2*h:-h]) - median(self.fit[-h:])
                row['fit_window_gain'] = gain
                row['fit_gain_resolution'] = max(self.variation,self.resolution[self.fit_id]['objective'])
                reference = self.selected['traces']
                departed = [name for name in self.trace_ids
                            if all(p[name]['newly_outside_loss'] - reference[name]['newly_outside_loss'] >
                                   self.resolution[name]['newly_outside_loss'] for p in (previous,panel))]
                row['persistent_retention_departures'] = departed
                if departed:
                    row['decision'] = 'persistent_loss_on_previously_in_tolerance_cells'
                elif not gains:
                    row['decision'] = ('fit_gain_without_resolved_development_gain' if gain > row['fit_gain_resolution']
                                       else 'no_resolved_fit_or_development_gain')
            if n == self.budget and row['decision'] == 'continue':
                row['decision'] = 'compute_allowance_exhausted'
        row['selected_generation'] = self.selected['generation'] if self.selected else None
        row['terminal'] = row['decision'] != 'continue'
        self.panels.append(row)
        return row


def local_retention(maps, baseline_inside=None):
    """All matches at each reference position and merge contribute, including repeated DTW rows."""
    import numpy as np

    a = maps.arrays
    outside = np.zeros_like(a['reference_position_merge_loss'],dtype=bool)
    np.logical_or.at(outside,a['pairs'][:,0],a['outside'])
    protected = ~outside if baseline_inside is None else baseline_inside
    if protected.dtype != bool or protected.shape != outside.shape:
        raise ValueError('reference or local retention mask changed during evaluation')
    return protected, dict(objective=maps.summary['objective'], complete=maps.summary['observation_complete'],
        baseline_inside_cells=int(protected.sum()), newly_outside_cells=int(np.sum(protected & outside)),
        newly_outside_loss=float(a['reference_position_merge_loss'][protected].sum()),
        inside_now_cells=int(np.sum(~outside)), reference_cells=outside.size,
        max_excess=maps.summary['max_excess'], pairs=maps.summary['pairs'])


class SelectionRecorder:
    """Keep reused native measurement spans, with separate time series for development decodes."""
    def __init__(self, recorder, member, repetition):
        self.context, self.span = recorder.context, recorder.span
        self.recorder, self.member, self.repetition = recorder, member, repetition

    def metric(self, name, value, *, step):
        self.recorder.metric('development.trace.'+self.member+'.'+('repeat.' if self.repetition else '')+
                             name.removeprefix('native.'), value, step=step)


def observe_panel(boundary, args, fit_result):
    import numpy as np
    from . import validation_driver as driver

    app, policy = boundary.app, boundary.policy
    n, local = len(policy.fit), args['number']
    fit = boundary.spec['fit']
    if args['case']['tokens'] != fit['tokens'] or args['case']['id'] != fit['id']:
        raise ValueError('development policy and actual native fitted prompt differ')
    output = {}
    for member in [fit, *boundary.spec['panel']]:
        name = member['id']
        folder = boundary.run/'validation'/str(local)/name
        folder.mkdir(parents=True)
        same_fit = name == fit['id']
        call = dict(args)
        if not same_fit:
            call.update(case=driver.read(member['cases'])[0], cases_binding=member['cases'],
                        inputs=dict(args['inputs'],member=name))
            if 'target' in member:
                call['reference'],call['loss_identity'] = app.load_reference(dict(boundary.declaration,**member['target']))
        def acquire(repeat):
            source = folder/('unchanged-repeat' if repeat else 'current')
            source.mkdir()
            recorder = call['recorder']
            with recorder.context(measurement_role='development_unchanged_repeat' if repeat else 'development_selection',
                                  validation_member=name,model_generation=local):
                result = boundary.fit_acquire(**dict(call,run=source,previous=None,
                    recorder=SelectionRecorder(recorder,name,repeat)))
            return result[5], result[-1]
        measurement, maps = (fit_result[5],fit_result[-1]) if same_fit else acquire(False)
        if not same_fit:
            app.full_loss_maps.write_local(maps,folder/'local')
        if n == 0:
            protected, row = local_retention(maps)
            with (folder/'initial-inside.npy').open('xb') as stream:
                np.save(stream,protected,allow_pickle=False)
            anchor = app.binding(folder/'initial-inside.npy')
            replay_measurement, replay_maps = acquire(True)
            _, repeat = local_retention(replay_maps,protected)
            app.full_loss_maps.write_local(replay_maps,folder/'unchanged-local')
            row['complete'] &= repeat['complete']
            row['repeat_range'] = {m: abs(repeat[m]-row[m]) for m in METRICS}
            row['repeat_measurement'] = replay_measurement
            row['repeat_maps'] = app.binding(folder/'unchanged-local'/'loss-maps.json')
            del replay_maps
        else:
            anchor = policy.panels[0]['traces'][name]['initial_inside']
            with driver.checked(anchor).open('rb') as stream:
                protected = np.load(stream,allow_pickle=False)
            _, row = local_retention(maps,protected)
        if not same_fit:
            row['local_maps'] = app.binding(folder/'local'/'loss-maps.json')
        row.update(initial_inside=anchor, measurement=measurement)
        app.save(folder/'assessment.json',row)
        output[name] = row
    return output
