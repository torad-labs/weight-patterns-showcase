"""Fresh cohort acquisitions and one joint native state transition per update."""

from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from functools import wraps
import ctypes
import inspect
import math
from pathlib import Path
import signal
import sys
from unittest.mock import patch

import torch

from pipeline.weight_edit import final_experiment as baseline, group_contrast_objective as objective
from pipeline.weight_edit import full_graph, group_disk_state, native_key_driver, retired_admission, trace_response
from pipeline.weight_edit.common_prefix_driver import moments
from pipeline.weight_edit.common_prefix_objective import checked


def give_back_freed_memory():
    """Return freed malloc pages to the kernel; a phase's temporaries must not stay resident."""
    ctypes.CDLL('libc.so.6').malloc_trim(0)


def response(shards):
    value, slope, curvature = (math.fsum(getattr(step, key) for step in shards)
                               for key in ('objective', 'slope', 'curvature'))
    if value == 0. or curvature == 0. or slope >= 0.:
        status = 'zero_residual' if value == 0. else 'zero_response' if curvature == 0. else 'non_descent'
        return trace_response.Step(value, slope, curvature, None, value, status)
    length = -slope/(2.*curvature)
    if not math.isfinite(length):
        raise ValueError('The cohort response gives a nonfinite length')
    return trace_response.Step(value, slope, curvature, length, max(0., value+length*slope/2.), 'proposal')


def aggregate(rows):
    groups = {role: [op for row in rows for op in row['operations'] if op['role'] == role]
              for role in objective.ROLES}
    result = {role: dict(objective=math.fsum(value for op in ops
                             if (value := op.get('measured_contribution', op['objective'])) is not None),
                         represented_cells=sum(op['represented_cells'] for op in ops),
                         supported_cells=sum(op['supported_cells'] for op in ops),
                         baseline_required_cells=sum(op['baseline_required_cells'] for op in ops),
                         unmeasured_operations=sum(op['status'] == 'unmeasured' for op in ops),
                         incomplete_operations=sum(op['status'] != 'complete' for op in ops))
              for role, ops in groups.items()}
    unmeasured = sum(row['unmeasured_operations'] for row in result.values())
    measured = math.fsum(row['objective'] for row in result.values())
    ended = all(row['complete'] for row in rows)
    complete = ended and not any(row['incomplete_operations'] for row in result.values())
    for row in result.values():
        if row['incomplete_operations'] or not ended:
            row['measured_contribution'], row['objective'] = row['objective'], None
    return dict(roles=result, unmeasured_operations=unmeasured, complete=complete, trajectories_ended=ended,
                measured_contribution=measured, objective=measured if complete else None, members=rows)


def paired_comparison(previous, current):
    def losses(summary):
        cells = {}
        for member in summary['members']:
            for op in member['operations']:
                for row in op['cells']:
                    key = (member['id'], op['operation'], op['role'], row['source_event'], row['merge'])
                    cells.setdefault(key, []).append(row['weighted_loss'])
        return {key: math.fsum(values) for key, values in cells.items()}

    earlier, later = losses(previous), losses(current)
    common = earlier.keys() & later.keys()
    matched = {}
    for role in objective.ROLES:
        keys = sorted(key for key in common if key[2] == role)
        matched[role] = dict(common_cells=len(keys),
            source=math.fsum(earlier[key] for key in keys) if keys else None,
            successor=math.fsum(later[key] for key in keys) if keys else None,
            change=math.fsum(later[key] - earlier[key] for key in keys) if keys else None)
    same = earlier.keys() == later.keys()
    return dict(scope='common_registered_source_cells', same_operation_support=same,
                full_comparison_valid=same and previous['complete'] and current['complete'], matched=matched,
                lost_cells=sorted(earlier.keys()-later.keys()), new_cells=sorted(later.keys()-earlier.keys()))


def checked_forward(graph, forward):
    for field, anchors in (('incoming', graph.native_in), ('merged', graph.native_out)):
        for actual, expected in zip(forward[field], anchors, strict=True):
            primal, _ = torch.autograd.forward_ad.unpack_dual(actual)
            if not torch.equal(primal.detach(), expected):
                raise ValueError('The differentiable cohort origin must reproduce the current native trace')
    return forward


class Training:
    def __init__(self, app, run, spec, config, field, working=None):
        self.app, self.run, self.spec, self.config, self.field = app, run, spec, config, field
        self.working = working   # the run's disk-backed arrays, when it declares them
        self.model = self.current = self.measurement = self.generation = None
        self.member_previous = {}
        self.baseline_support = {}
        self.cases = {m['id']:baseline.read(baseline.ROOT/m['case']['record'])[0] for m in spec.get('members', [])}

    def settle(self):
        """A phase has ended: its temporaries and the working files' pages go back to the machine."""
        give_back_freed_memory()
        if self.working is not None:
            self.working.settle()

    def observe(self, actual_acquire, invocation, main):
        app, root = self.app, self.run
        args = invocation.arguments
        generation, primary = args['number'], self.config['primary']
        panel, rows = {}, []
        for member in self.spec['members']:
            name = member['id']
            case = self.cases[name]
            if name == primary:
                native = main
            else:
                target = root/'group-members'/name
                target.mkdir(parents=True, exist_ok=True)
                params = dict(args, run=target, case=case, cases_binding=member['case'],
                              previous=self.member_previous.get(name), inputs=dict(args['inputs'], member=name))
                with args['recorder'].context(training_member=name):
                    native = actual_acquire(**params)
            registrations = self.field.register(name, native[0], prompt_events=len(case['tokens']),
                                                baseline_support=self.baseline_support.get(name),
                                                full_trace_pairs=native[1])
            if generation == 0:
                self.baseline_support[name] = {op.name: op.required for op in registrations}
            with torch.no_grad():
                values = [op.measure(native[0], member['first_source_event']) for op in registrations]
            self.member_previous[name] = native[5]
            rows.append(dict(id=name, complete=native[-2], measurement=native[5], operations=values))
            panel[name] = dict(native=native[0], history=native[2], raw=native[4], registrations=registrations)
            self.settle()
        summary = aggregate(rows)
        if self.measurement:
            old = baseline.read(baseline.ROOT/self.measurement['record'])
            summary['comparison_to_previous'] = paired_comparison(old, summary)
        path = root/'loop'/str(generation)/'group-measurement.json'
        app.save(path, dict(summary, model_identity=args['model_identity'], generation=generation,
                            policy=self.config['policy'], previous_group_measurement=self.measurement))
        self.measurement, self.current, self.generation = app.binding(path), panel, generation
        for role, row in summary['roles'].items():
            for name in ('objective', 'represented_cells', 'supported_cells', 'baseline_required_cells', 'unmeasured_operations'):
                if row[name] is not None:
                    args['recorder'].metric(f'native.group.{role}.{name}', row[name], step=generation)
        if summary['objective'] is not None:
            args['recorder'].metric('native.group.objective', summary['objective'], step=generation)
        if objective.support_policy(self.spec) == objective.LIVE_SUPPORT and any(
                row['represented_cells'] for row in summary['roles'].values()):
            args['recorder'].metric('native.group.measured_contribution', summary['measured_contribution'], step=generation)
            stop = self.declaration.get('group_loss_stop')
            if stop is not None and summary['measured_contribution'] <= stop['threshold']:
                app.save(root/'loss-threshold-reached.json', dict(rule=stop, observed=summary['measured_contribution'],
                         measurement=self.measurement, generation=generation, model_identity=args['model_identity']))
                signal.raise_signal(signal.SIGUSR1)
        self.settle()
        return (*main[:-2], main[-2] and summary['trajectories_ended'], main[-1])

    @contextmanager
    def graph_rows(self, state):
        required = state.required_rows
        union = sorted({token for row in self.current.values() for token in row['history']})
        with patch.object(state, 'required_rows', lambda name, unused: required(name, union)):
            yield

    def evaluate(self, primary_graph, state, visit):
        for member, row in self.current.items():
            if not any(op.groups for op in row['registrations']):
                continue
            graph = primary_graph
            if member != self.config['primary']:
                graph = self.app.graph_for(self.model, state, row['history'], row['native'], row['raw'], self.declaration)
            if graph.parameter_rows != primary_graph.parameter_rows or tuple(graph.leaves) != tuple(primary_graph.leaves):
                raise ValueError('Cohort members did not share the eligible coordinate inventory and physical row union')
            yield visit(member, row, graph)
            del graph

    @staticmethod
    def requests(row, forward):
        for operation in row['registrations']:
            for residual in operation.residuals(forward):
                yield operation, residual

    def gradients(self, primary, state, recorder):
        combined, connections = {}, set()

        def accumulate(name, gradient):
            # Joint direction is the negative summed gradient. Each member's gradient is
            # folded in as backward finishes it and freed, so one gradient set exists.
            # It stays in RAM: in a file, every member's pass over it went to the disk.
            if name in combined:
                combined[name].sub_(gradient)
            else:
                combined[name] = gradient.neg_()

        def component(member, row, graph):
            def loss(forward):
                terms = [residual.square().sum() for _, residual in self.requests(row, checked_forward(graph, forward))]
                return torch.stack(terms).sum(), {}
            with recorder.context(training_member=member):
                return full_graph._derivative(graph, row['history'], loss, sink=accumulate)

        for _, _, _, inventory in self.evaluate(primary, state, component):
            for name in inventory:
                if not inventory[name]['gradient_was_none']:
                    connections.add(name)
            for name, row in inventory.items():
                if name not in combined:
                    combined[name] = torch.zeros(row['shape'], dtype=torch.float32).neg_()
        diagnostics = {}
        for name, value in combined.items():
            finite = bool(torch.isfinite(value).all())
            if not finite:
                raise ValueError(f'Nonfinite joint group derivative: {name}')
            norm = float(value.double().square().sum())
            count = int(torch.count_nonzero(value))
            diagnostics[name] = dict(status=('connected_nonzero' if count else 'connected_zero' if name in connections
                                             else 'outside_registered_computational_support'),
                gradient_was_none=name not in connections, norm_squared=norm, nonzero_entries=count,
                finite=finite, shape=list(value.shape), device=str(value.device), row_ids=primary.parameter_rows.get(name))
        self.settle()
        return combined, diagnostics, math.fsum(row['norm_squared'] for row in diagnostics.values())

    def responses(self, primary, state, direction, reference, pairs, recorder, kind):
        ordinary, groups = [], {}

        def component(member, row, graph):
            def read(forward):
                checked_forward(graph, forward)
                if member == self.config['primary']:
                    for _ in self.app.residuals(reference, pairs, ordinary)(forward):
                        pass
                for operation, residual in self.requests(row, forward):
                    key = (member, operation.name, operation.role)
                    groups.setdefault(key, []).append(moments(None, residual))
                    yield residual
            with recorder.context(training_member=member, response_kind=kind), recorder.span('solver.jvp', detached=True):
                return full_graph.directional_response(graph, row['history'], direction, read)

        collective = response(list(self.evaluate(primary, state, component)))
        self.settle()
        grouped = [dict(member=member, operation=name, role=role,
            **{key: math.fsum(row[key] for row in group)
               for key in ('baseline', 'loss_cross', 'response_norm_squared')})
            for (member, name, role), group in groups.items()]
        return collective, ordinary, grouped

    def propose(self, graph, history, reference, pairs, measured, state, eta, recorder):
        if tuple(history) != tuple(self.current[self.config['primary']]['history']):
            raise ValueError('Cohort proposal requires its own current registered primary history')
        summary = baseline.read(baseline.ROOT/self.measurement['record'])
        base = dict(group_measurement=self.measurement, group_objective=summary['objective'],
                    available_group_contribution=summary['measured_contribution'],
                    eligible_tensors=len(state.specs), addressable_rows=graph.parameter_rows, eta=eta)
        live = objective.support_policy(self.spec) == objective.LIVE_SUPPORT
        if (live and not any(row['represented_cells'] for row in summary['roles'].values())
                or not live and summary['unmeasured_operations']):
            return dict(base, status='unmeasured_registered_operations', gradient_norm_squared=0.), None
        with recorder.span('proposal.weight_derivative', detached=True):
            direction, inventory, norm = self.gradients(graph, state, recorder)
        base.update(gradient_diagnostics=inventory, gradient_norm_squared=norm)
        if set(direction) != set(state.specs):
            raise ValueError('The cohort derivative omitted an eligible physical tensor')
        if norm == 0.:
            return dict(base, status='zero_group_gradient'), None
        continuous, continuous_ordinary, continuous_groups = self.responses(graph, state, direction, reference, pairs,
                                                                            recorder, 'continuous_direction')
        if not math.isclose(continuous.objective, summary['measured_contribution'], rel_tol=8*torch.finfo(torch.float64).eps):
            raise ValueError('Held response does not reproduce the available fresh registered cohort contribution')
        base.update(continuous_response=asdict(continuous), continuous_moments=continuous_ordinary,
                    continuous_group_moments=continuous_groups)
        if continuous.length is None:
            return dict(base, status=continuous.status), None
        length = eta*continuous.length
        with recorder.span('proposal.native_rounding', detached=True):
            displacement, updates, hashes = self.app.preview(state, graph, direction, length)
        installed, installed_ordinary, installed_groups = self.responses(graph, state, displacement, reference, pairs,
                                                                       recorder, 'actual_native_displacement')
        del displacement
        return dict(base, status='ready', applied_length=length, candidate_chunk_hashes=hashes, update_inventory=updates,
            changed_values=sum(row['changed_entries'] for row in updates.values()),
            continuous_prediction=continuous.objective+length*continuous.slope+length**2*continuous.curvature,
            installed_displacement_response=asdict(installed), installed_moments=installed_ordinary,
            installed_group_moments=installed_groups, installed_displacement_coefficient=1.,
            installed_displacement_prediction=installed.objective+installed.slope+installed.curvature), direction

    def archive_members(self, previous, current, declaration):
        app, root = self.app, baseline.ROOT
        archive = app.archive_source.legacy
        active = declaration['consumed_trace_archive']['active_runs']
        previous_path = self.run/'loop'/str(previous)/'group-measurement.json'
        old = baseline.read(previous_path)
        successor = baseline.read(checked(self.measurement))
        proposal = baseline.read(self.run/'loop'/str(current)/'proposal.json')
        if (successor['previous_group_measurement'] != app.binding(previous_path)
                or proposal['group_measurement'] != app.binding(previous_path)
                or successor['generation'] != current or old['generation'] != previous
                or not successor['trajectories_ended']):
            raise ValueError('Cohort archive requires the exact consumed source and a complete successor acquisition')
        needs, problems = archive.archive.needed(root, active)
        if problems:
            raise ValueError(f'Cannot determine active cohort archive inputs: {problems}')
        pending = []
        output = self.run/'trace-archives'/str(previous)/'group-members'
        for member in old['members']:
            name = member['id']
            if name == self.config['primary']:
                continue
            measurement = baseline.read(checked(member['measurement']))
            capture = baseline.read(checked(measurement['capture']))
            if (not member['complete'] or measurement['model_identity'] != old['model_identity']
                    or capture['model_identity'] != old['model_identity'] or measurement['member'] != name):
                raise ValueError('Cohort archive source identity differs from the consumed native observation')
            folder = output/name
            membership = dict(sources=[measurement['capture']['record'], member['measurement']['record'],
                app.binding(self.run/'declaration.json')['record'], app.binding(previous_path)['record']],
                answers=[name], pooled={'.':[name]})
            archive.save(folder/'membership.json', membership)
            for stream in archive.STREAMS:
                binding = capture['raw'][stream]
                target = root/binding['record']
                expected = self.run/'group-members'/name/'loop'/str(previous)/'capture/raw'/name/stream
                if (target != expected or target.resolve()!=target
                        or any(target == root/path or target.is_relative_to(root/path) for path in needs)):
                    raise ValueError('Only an exact inactive consumed cohort stream can be released')
                record_name, record, allowlist = archive.inventoried(root, binding, membership, folder)
                if target.exists():
                    with archive.archive_environment():
                        report, failed = archive.archive.bucket(root, 'upload', record, sorted(record['files']))
                    (folder/(stream+'-upload.log')).write_text(report)
                    if failed:
                        raise RuntimeError('Cohort stream upload failed before local release')
                    with archive.archive_environment(allowlist, record_name):
                        errors = archive.archive.verify(root, record_name)
                    if errors:
                        raise RuntimeError(f'Cohort stream readback failed before local release: {errors}')
                elif record.get('remote_bytes_verified') is None:
                    raise ValueError('Missing cohort stream lacks verified remote bytes')
                pending.append(record_name)
        receipts = []
        for name in pending:
            result, problems = archive.archive.release(root, name, active)
            if problems or result is None or result['files_released']+result['already_gone'] != 1:
                raise RuntimeError(f'Guarded cohort archive release failed: {problems}')
            receipts.append(dict(record=name, release=result))
            archive.save(output/'release.json', dict(previous_step=previous, current_step=current, archives=receipts))
        return receipts

    @contextmanager
    def connected(self, declaration):
        app, self.declaration = self.app, declaration
        original_acquire, original_graph, original_execute = app.acquire, app.graph_for, app.execute
        original_budget = app.storage_budget

        def storage_budget(specs, state, context, inherited):
            budget = original_budget(specs, state, sum(self.config['maximum_events_per_member'].values()), inherited)
            if self.working is not None and self.working.limits is not None:
                # The working files' own growth is reserved with each step's checkpoint and capture.
                growth = self.working.pending()
                budget = dict(budget, working_files_bytes=growth,
                              free_required_bytes=budget['free_required_bytes'] + growth)
            return budget

        @wraps(original_acquire)
        def acquire(*args, **kwargs):
            invocation = inspect.signature(original_acquire).bind(*args, **kwargs)
            self.current = None
            primary_id = self.config['primary']
            [member] = [m for m in self.spec['members'] if m['id'] == primary_id]
            invocation.arguments.update(case=self.cases[primary_id], cases_binding=member['case'])
            primary = original_acquire(*invocation.args, **invocation.kwargs)
            return self.observe(original_acquire, invocation, primary)

        @wraps(original_graph)
        def graph_for(model, state, history, native, raw, declaration):
            self.model = model
            with self.graph_rows(state):
                return original_graph(model, state, history, native, raw, declaration)

        @wraps(original_execute)
        def execute(run, declaration, recorder, capture, completion):
            archive = app.archive_source.archive_previous
            reached = False

            def archived(run, previous_step, current_step, declaration, *, stop_requested=None):
                nonlocal reached
                try:
                    receipt = archive(run, previous_step, current_step, declaration, stop_requested=stop_requested)
                except InterruptedError:
                    if not stop_requested or not stop_requested():
                        raise
                    folder = run/'trace-archives'/str(previous_step)
                    pending = baseline.read(folder/'release.json')
                    if (pending.get('phase') != 'waiting_for_publication' or pending.get('archives') != []
                            or (pending['previous_step'], pending['current_step']) != (previous_step, current_step)):
                        raise
                    retained = dict(pending, phase='retained_locally_on_stop', members=self.config['members'])
                    app.save(folder/'joint-retained-on-stop.json', retained)
                    return retained
                secondary = self.archive_members(previous_step, current_step, declaration)
                if self.spec['updates'] != objective.OBSERVED_STOP and current_step == self.spec['updates']:
                    reached = True
                    signal.raise_signal(signal.SIGUSR1)
                return dict(receipt, archives=[*receipt['archives'], *secondary])

            with patch.object(app.archive_source, 'archive_previous', archived):
                result = original_execute(run, declaration, recorder, capture, completion)
            final = baseline.read(run/'final-state.json')
            group = baseline.read(baseline.ROOT/self.measurement['record'])
            paired = group.get('comparison_to_previous', {})
            acceptable = ((reached or self.spec['updates'] == objective.OBSERVED_STOP)
                          and group['complete'] and paired.get('full_comparison_valid') is True)
            app.save(run/'group-boundary.json', dict(status='completed_interval' if reached else 'native_terminal',
                complete_group_comparison=acceptable,
                installed_updates=final['iteration'], group_measurement=self.measurement,
                state=app.binding(run/'final-state.json'), launcher_exit=app.binding(run/'launcher-exit.json')))
            return result

        with (patch.object(app, 'acquire', acquire), patch.object(app, 'graph_for', graph_for),
              patch.object(app, 'propose', self.propose), patch.object(app, 'execute', execute),
              patch.object(app, 'storage_budget', storage_budget)):
            yield


def execute(run):
    from pipeline.weight_edit import group_contrast_experiment as launcher

    run = baseline.run_path(run)
    declaration = baseline.read(run/'declaration.json')
    launcher.check(run, declaration)
    config = declaration['group_contrast']
    spec, _ = objective.specification(config['policy'])
    field = objective.Objective(spec)
    app = baseline.controller(baseline.MEASURED)
    recorder = app.Recorder

    def recording(path, attributes):
        return recorder(path, dict(attributes, entrypoint=config['driver']['record'], group_objective_sha256=config['policy']['sha256']))

    with ExitStack() as stack:
        working = None
        if declaration.get('disk_working_state'):
            working = stack.enter_context(group_disk_state.connected(app, run, launcher.working_limits(declaration)))
        stack.enter_context(patch.object(sys, 'argv', [str(__file__), str(run)]))
        stack.enter_context(patch.object(app, 'Recorder', recording))
        stack.enter_context(native_key_driver.connected(app, declaration))
        if working is not None:
            stack.enter_context(working.native_traces())
        stack.enter_context(Training(app, run, spec, config, field, working).connected(declaration))
        if declaration.get('retired_input_admission'):
            retired_admission.bind(app, declaration)
        app.main()


if __name__ == '__main__':
    execute(Path(sys.argv[1]))
