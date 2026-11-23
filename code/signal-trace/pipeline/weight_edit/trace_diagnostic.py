"""Passively read a frozen computation-based signal from the installed learner's existing forwards."""
from contextlib import contextmanager
from functools import wraps
import inspect
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from . import final_experiment as baseline

FORMAT = 'installed-trace-diagnostic/1'
CONTINUOUS = 'continuous_direction'
ROUNDED = 'actual_native_displacement'
METRICS = ('native.trace_signal.match_count','native.trace_signal.score',
           'proposal.trace_signal.continuous_change','proposal.trace_signal.rounded_change',
           'native.trace_signal.fresh_change')


def registration(binding):
    def checked(row):
        path = (baseline.ROOT / row['record']).resolve()
        if 'reserved-40b' in path.parts or baseline.binding(path)['sha256'] != row['sha256']:
            raise ValueError(f'invalid diagnostic binding {row["record"]}')
        return path
    spec = baseline.read(checked(binding))
    if set(spec) != {'format','artifact','reader','dependencies','inputs'} or spec['format'] != FORMAT:
        raise ValueError('bind the diagnostic reader, frozen signal artifact, inputs and dependencies')
    for row in [spec['reader'],spec['artifact'],*spec['dependencies'],*spec['inputs']]:
        checked(row)
    return spec


def scalar(value):
    if not isinstance(value,torch.Tensor) or value.ndim != 0:
        raise ValueError('a diagnostic signal returns one scalar tensor per registered cell')
    number = float(value.detach())
    if not math.isfinite(number):
        raise ValueError('the registered scalar is nonfinite')
    return number


def signals(reader, native, artifact, prompt_events):
    cells = reader.locate(native,artifact,prompt_events=prompt_events)
    if not isinstance(cells,list):
        raise TypeError('locate must return a list of computational matches')
    values = []
    seen = set()
    for cell in cells:
        event,merge,key = cell['event'],cell['merge'],cell['key']
        if (not isinstance(key,str) or not key or type(event) is not int or type(merge) is not int
                or not 0 <= merge < len(native['incoming']) or not 0 <= event < len(native['incoming'][merge])):
            raise ValueError('signal location lies outside the native trajectory')
        if (key,event,merge) in seen:
            raise ValueError('signal reader returned the same site twice')
        seen.add((key,event,merge))
        values.append(dict(cell=cell,value=scalar(reader.score(native,cell,artifact))))
    return values


def unique_values(rows):
    result = {}
    for row in rows:
        key = row['cell']['key']
        result[key] = None if key in result else row
    return result


def comparison(before, after, predictions):
    a, b = unique_values(before), unique_values(after)
    result = {}
    for key in sorted(a.keys()|b.keys()):
        if key not in a or key not in b:
            result[key] = dict(status='missing_source_or_target')
        elif a[key] is None or b[key] is None:
            result[key] = dict(status='ambiguous_source_or_target')
        else:
            source,target = a[key],b[key]
            item = dict(status='matched',source_cell=source['cell'],target_cell=target['cell'],
                        before=source['value'],after=target['value'],observed_change=target['value']-source['value'])
            for phase in (CONTINUOUS,ROUNDED):
                pred = predictions.get(phase,{}).get(key)
                if pred and pred.get('predicted_change') is not None:
                    item[phase] = dict(predicted_change=pred['predicted_change'],
                        prediction_error=item['observed_change']-pred['predicted_change'])
            result[key] = item
    return result


class Diagnostic:
    def __init__(self, app, run, binding, reader, artifact):
        self.app, self.run, self.binding = app, run, binding
        self.reader, self.artifact = reader, artifact
        self.current = self.pending = self.active = None

    def native(self, args, result):
        number = args['number']
        folder = self.run/'loop'/str(number)/'trace-diagnostic'
        folder.mkdir()
        history_hash = hashlib.sha256(np.asarray(result[2],dtype='<i8').tobytes()).hexdigest()
        scope = 'observed'
        with torch.no_grad():
            try:
                rows = signals(self.reader,result[0],self.artifact,len(args['case']['tokens']))
            except (ValueError,TypeError,IndexError,KeyError,RuntimeError) as exc:
                rows, scope = [], 'reader_error: '+str(exc)
        maps = result[-1].arrays
        for row in rows:
            e,m = row['cell']['event'],row['cell']['merge']
            idx = np.flatnonzero(maps['pairs'][:,1] == e)
            row['local_loss'] = [dict(reference_event=int(maps['pairs'][k,0]),
                distance=float(maps['distance'][k,m]),radius=float(maps['reference_radius'][maps['pairs'][k,0],m]),
                weighted_loss=float(maps['weighted_loss'][k,m])) for k in idx]
        record = dict(consumed_history_sha256=history_hash,generation=number,model_identity=args['model_identity'],measurement=result[5],
                      complete=result[-2],scope='fresh_generated_computation',policy=self.binding,
                      cells=rows,match_status=scope if rows or scope != 'observed' else 'no_match')
        self.app.save(folder/'fresh.json',record)
        recorder = args.get('recorder')
        if recorder is not None:
            recorder.metric(METRICS[0],len(rows),step=number)
            if result[-2] and len(rows) == 1:
                recorder.metric(METRICS[1],rows[0]['value'],step=number,attributes={'diagnostic_key':rows[0]['cell']['key']})
        if self.pending is not None:
            source,prediction = self.pending
            if number != source['generation']+1 or args['previous'] != source['measurement']:
                raise ValueError('diagnostic fresh acquisition is not the direct successor of the proposal')
            transition = dict(source=source['measurement'],target=result[5],source_model=source['model_identity'],
                              target_model=args['model_identity'],prediction=prediction['record'],
                              complete=source['complete'] and result[-2],policy=self.binding)
            same_history = source['consumed_history_sha256'] == history_hash
            transition['matched_consumed_history'] = same_history
            transition['independent_conditioned_replay'] = 'unavailable'
            if transition['complete'] and same_history:
                fixed = []
                with torch.no_grad():
                    for row in source['cells']:
                        cell = row['cell']
                        try:
                            fixed.append(dict(cell=cell,value=scalar(self.reader.score(result[0],cell,self.artifact))))
                        except (ValueError,TypeError,IndexError,KeyError,RuntimeError):
                            continue
                transition['fixed_source_cells_native'] = dict(scope='fresh_decode_on_identical_full_consumed_history',
                    measurement=result[5],matches=comparison(source['cells'],fixed,prediction['phases']))
            else:
                transition['fixed_source_cells_native'] = dict(scope='unavailable')
            if transition['complete']:
                transition['matches'] = comparison(source['cells'],rows,prediction['phases'])
                transition['status'] = ('matched' if any(v['status'] == 'matched' for v in transition['matches'].values())
                                        else 'no_unique_correspondence')
            else:
                transition.update(matches={},status='incomplete_native_trajectory')
            self.app.save(folder/'transition.json',transition)
            linked = [v for v in transition['matches'].values() if v['status'] == 'matched']
            if recorder is not None and len(linked) == 1:
                recorder.metric(METRICS[4],linked[0]['observed_change'],step=number,
                                attributes={'diagnostic_key':linked[0]['source_cell']['key']})
            self.pending = None
        self.current = record

    def dual(self, forward, phase):
        data = self.active['phases']
        if phase not in (CONTINUOUS,ROUNDED) or phase in data:
            raise ValueError('unexpected or repeated directional product while observing the registered signal')
        by_key = {}
        for row in self.current['cells']:
            key,cell = row['cell']['key'],row['cell']
            try:
                value = self.reader.score(forward,cell,self.artifact)
                primal,tangent = torch.autograd.forward_ad.unpack_dual(value)
                origin = scalar(primal)
                item = dict(cell=cell,origin=origin,native_origin=row['value'],origin_difference=origin-row['value'],
                            tangent=scalar(tangent) if tangent is not None else None,
                            status='measured' if tangent is not None else 'unconnected_forward_tangent')
            except (ValueError,TypeError,IndexError,KeyError,RuntimeError) as exc:
                item = dict(cell=cell,status='reader_error',error=str(exc))
            by_key.setdefault(key,[]).append(item)
        out = {key: items[0] if len(items) == 1 else dict(status='ambiguous_registration',candidates=items)
               for key,items in by_key.items()}
        data[phase] = out

    @contextmanager
    def connected(self):
        app = self.app
        acquire, propose, residuals = app.acquire, app.propose, app.residuals
        @wraps(acquire)
        def fresh(*args,**kwargs):
            result = acquire(*args,**kwargs)
            bound = inspect.signature(acquire).bind(*args,**kwargs).arguments
            self.native(bound,result)
            return result
        @wraps(residuals)
        def requested(*args,**kwargs):
            read = residuals(*args,**kwargs)
            if self.active is None:
                return read
            phase = self.active['recorder'].context_attributes['response_kind']
            def observe(forward):
                self.dual(forward,phase)
                yield from read(forward)
            return observe
        @wraps(propose)
        def proposal(*args,**kwargs):
            bound = inspect.signature(propose).bind(*args,**kwargs).arguments
            if self.current is None or not self.current['complete']:
                raise ValueError('the registered signal requires a complete current native acquisition')
            self.active = dict(recorder=bound['recorder'],phases={})
            try:
                value,direction = propose(*args,**kwargs)
                phases = self.active['phases']
                length = value.get('applied_length')
                for phase,rows in phases.items():
                    coefficient = length if phase == CONTINUOUS else 1.
                    for site in rows.values():
                        for item in site.get('candidates',[site]):
                            tangent = item.get('tangent')
                            item['coefficient'] = coefficient
                            item['predicted_change'] = (tangent*coefficient if tangent is not None
                                                        and coefficient is not None else None)
                for phase,metric in ((CONTINUOUS,METRICS[2]),(ROUNDED,METRICS[3])):
                    qualified = [v for v in phases.get(phase,{}).values() if v.get('predicted_change') is not None]
                    if len(qualified) == 1:
                        site = qualified[0]
                        bound['recorder'].metric(metric,site['predicted_change'],step=self.current['generation']+1,
                                                  attributes={'diagnostic_key':site['cell']['key']})
                record = dict(policy=self.binding,status=value['status'],source=self.current['measurement'],
                              source_model=self.current['model_identity'],phases=phases,
                              coefficient_scope='continuous tangent times applied length; rounded displacement tangent times one')
                path = self.run/'loop'/str(self.current['generation'])/'trace-diagnostic'/'proposal.json'
                app.save(path,record)
                link = app.binding(path)
                if direction is not None:
                    if set(phases) != {CONTINUOUS,ROUNDED}:
                        raise ValueError('ready proposal did not expose both directional products')
                    self.pending = self.current,dict(record=link,phases=phases)
                return dict(value,trace_diagnostic=link),direction
            finally:
                self.active = None
        app.acquire,app.propose,app.residuals = fresh,proposal,requested
        try:
            yield
        finally:
            app.acquire,app.propose,app.residuals = acquire,propose,residuals


def from_declaration(app, run, declaration):
    binding = declaration['trace_diagnostic']['policy']
    spec = registration(binding)
    reader = app.load_module('registered_native_trace_reader',baseline.ROOT/spec['reader']['record'])
    artifact = baseline.read(baseline.ROOT/spec['artifact']['record'])
    return Diagnostic(app,run,binding,reader,artifact)
