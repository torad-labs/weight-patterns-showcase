"""Opt-in original-native pre-cache K capture and graph assembly for the existing learner."""
from contextlib import contextmanager
from functools import wraps
import fcntl
import re
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace, FunctionType
from unittest.mock import patch

from pipeline.weight_edit import final_experiment as baseline, native_key_anchor, native_key_finite, trace_diagnostic

# Keep exactly the residual stream plus original key producers. 'projection' without
# this filter also records unrelated, much larger attention and feed-forward arrays.
LABELS = '((inp_scaled|attn_post_norm|attn_out|ffn_post_norm|pe_in|per_layer_embd_out|pe_out|l_out|Kcur_pos)-[0-9]+|inp_scaled|result_norm)'


@contextmanager
def residual_index(path):
    # The historical residual reader verifies every record it sees against the
    # two-axis ADD layout. Route only the three-axis K records to their own reader.
    with open(path) as stream:
        native_key_label = re.compile(r'"label"\s*:\s*"Kcur_pos-[0-9]+"')
        yield (line for line in stream if not native_key_label.search(line))


def residual_forward(held, raw, merges, events, width):
    # The exact historical rows_in, taken, and raw_forward still perform bounds,
    # coverage, shape, addition and source checks. Isolate their file view without
    # copying a stream/index or changing process-global open.
    original=held.states_reading.rows_in
    filtered=FunctionType(original.__code__,dict(original.__globals__,open=residual_index),original.__name__)
    view=SimpleNamespace(rows_in=filtered,taken=held.states_reading.taken)
    forward=held.raw_forward
    return FunctionType(forward.__code__,dict(forward.__globals__,states_reading=view),forward.__name__)(
        raw,merges,events,width)


def storage_receipt(app, declaration, fd, read_fd):
    if not declaration.get('disk_working_state'):
        return dict(model_storage='one anonymous memfd; read-only child descriptor',
                    seals=fcntl.fcntl(fd, app.factory.F_GET_SEALS))
    [output] = declaration['outputs']
    path = baseline.ROOT / output / 'working-files' / 'model.gguf'
    on_disk, writable, reader = path.lstat(), os.fstat(fd), os.fstat(read_fd)
    if (not stat.S_ISREG(on_disk.st_mode) or stat.S_IMODE(on_disk.st_mode) != 0o600
            or on_disk.st_nlink != 1 or on_disk.st_uid != os.getuid()
            or not os.path.samestat(on_disk, writable) or not os.path.samestat(on_disk, reader)
            or reader.st_size != declaration['disk_working_state']['model_bytes']
            or fcntl.fcntl(read_fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY):
        raise ValueError('Native capture must read the private, fixed-size working GGUF through a read-only descriptor')
    return dict(model_storage='private run-local GGUF; read-only child descriptor',
                working_file=str(path.relative_to(baseline.ROOT)), file_bytes=reader.st_size)


def make_capture(app, card, *, declaration=None):
    declaration = declaration or {}
    def capture(fd, folder, identity, cases_file, engine, inputs):
        [case] = baseline.read(cases_file)
        if set(case) != {'id','tokens','max_tokens'} or case['id']!=inputs['member']:
            raise ValueError('Expected one unrestricted free-generation prompt for native-K capture')
        folder.mkdir()
        actual = folder/'recorder-cases.json'
        app.save(actual,[dict(case,stream_labels=LABELS)])
        read_fd = os.open(f'/proc/self/fd/{fd}',os.O_RDONLY|os.O_CLOEXEC)
        held = app.held
        try:
            before = held.working_identity.fingerprint(f'/proc/self/fd/{fd}')
            storage = storage_receipt(app, declaration, fd, read_fd)
            command = [str(held.generation_engine.RECORDER),f'/proc/self/fd/{read_fd}',str(folder/'raw'),
                       str(actual),str(app.ROOT/'dependencies/llama-cpp-build'),'capture','projection','stream']
            with Path(f'/run/user/{os.getuid()}/gpu{card}.lock').open('r+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                held.frozen(inputs)
                with (folder/'stdout.jsonl').open('x') as stdout,(folder/'stderr.log').open('x') as stderr:
                    child=subprocess.Popen(command,env=engine.env,stdout=stdout,stderr=stderr,
                                           pass_fds=(read_fd,lock.fileno()))
                    try:
                        watched=held.complete_watch(engine,child)
                        code=child.wait()
                    except BaseException:
                        if child.poll() is None:
                            child.terminate()
                        child.wait()
                        raise
                if code:
                    raise subprocess.CalledProcessError(code,command)
                account=engine.ended(watched,folder/'stderr.log',cases=[case['id']])
            if before != held.working_identity.fingerprint(f'/proc/self/fd/{fd}'):
                raise ValueError('Installed weights changed during native key capture')
            if declaration.get('disk_working_state') and storage_receipt(app, declaration, fd, read_fd) != storage:
                raise ValueError('Disk working file changed during native key capture')
            held.frozen(inputs)
            raw=folder/'raw'/case['id']
            result=baseline.read(raw/'result.json')
            if (result['scope']!='projection' or result['stream_labels']!=LABELS or result['teacher_token_count']
                    or result['conditioned_continuation'] or result['prompt_tokens']!=len(case['tokens'])):
                raise ValueError('Recorder did not produce the declared unrestricted projection stream')
            receipt=dict(model_identity=identity,command=command,engine=engine.described(passes=[account]),
                         cases=app.binding(cases_file),recorder_cases=app.binding(actual),
                         **storage,
                         file_identity_during_pass=before,raw={f:app.binding(raw/f) for f in ('stream.jsonl','stream.f32')})
            app.save(folder/'capture.json',receipt)
            return receipt
        finally:
            os.close(read_fd)
    return capture


@contextmanager
def connected(app, declaration):
    original_graph = app.graph_for
    @wraps(original_graph)
    def graph_for(model,state,history,native,raw,declaration):
        graph_type=app.full_graph.Graph
        def anchored_graph(wrapped, tokens, source):
            return graph_type(native_key_anchor.wrap(wrapped,tokens,source,raw),tokens,source)
        # graph_for constructs the original ParameterModel, FFN and projection adapters.
        # Substitute only its final constructor, before it computes the mathematical origin.
        with patch.object(app,'full_graph',SimpleNamespace(Graph=anchored_graph)):
            return original_graph(model,state,history,native,raw,declaration)
    # Keep a reference to the historical function before binding the projection reader.
    residual_source=SimpleNamespace(states_reading=app.held.states_reading, raw_forward=app.held.raw_forward)
    finite_read=native_key_finite.read_at_cells
    @wraps(finite_read)
    def passive(*args,**kwargs):
        try:
            return finite_read(*args,**kwargs)
        except (ValueError,IndexError,KeyError,TypeError,RuntimeError) as error:
            # read_at_cells restores the derivative mapping on every exit. An
            # unavailable optional forecast cannot select or veto an installation.
            return dict(status='finite_read_unavailable',error=f'{type(error).__name__}: {error}',sites=[])
    def read_raw(raw, nodes, count, width):
        return residual_forward(residual_source,raw,nodes,count,width)
    with (patch.object(app.factory,'capture',make_capture(app,declaration['card'],declaration=declaration)),
          patch.object(app,'graph_for',graph_for), patch.object(app.held,'raw_forward',read_raw),
          patch.object(trace_diagnostic,'Diagnostic',native_key_finite.Diagnostic),
          patch.object(native_key_finite,'read_at_cells',passive)):
        yield


def execute(run):
    from pipeline.weight_edit import native_key_experiment as launcher
    run=baseline.run_path(run)
    declaration=baseline.read(run/'declaration.json')
    launcher.check(run,declaration)
    app=baseline.controller(baseline.MEASURED)
    with connected(app,declaration), patch.object(sys,'argv',[str(__file__),str(run)]):
        if declaration.get('trace_diagnostic') or declaration.get('validation_stopping'):
            from pipeline.weight_edit import validation_driver
            with patch.object(baseline,'controller',return_value=app):
                validation_driver.main()
        else:
            if declaration.get('retired_input_admission'):
                from pipeline.weight_edit import retired_admission
                retired_admission.bind(app,declaration)
            app.main()


if __name__=='__main__':
    execute(Path(sys.argv[1]))
