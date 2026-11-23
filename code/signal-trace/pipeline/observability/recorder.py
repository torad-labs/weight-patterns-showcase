"""Append small operation events for the independent MLflow observer.

No arguments, return objects, tensors or response text are captured implicitly.
Logging failures cannot replace a research result or exception.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
from pathlib import Path
import os
import resource
import sys
import threading
import time
import uuid
from typing import Any

FORMAT = 'weight-patterns-observability/1'
ORGANIZATION = 'research-operations/1'
# These repeated operations stay independently exportable, but can be filtered
# out of the phase overview. The operation name never contains a dynamic ID.
DETAIL_OPERATIONS = frozenset(('solver.jvp', 'solver.vjp', 'proposal.evaluate'))


def scalars(values):
    """Select explicit scalar metadata without converting arbitrary objects."""
    return {key: value for key, value in values.items()
            if isinstance(key, str) and isinstance(value, (str, bool, int, float))
            and (not isinstance(value, float) or math.isfinite(value))}


class Recorder:
    def __init__(self, path, attributes=None):
        self.path = Path(path)
        self.attributes = scalars(attributes or {}) | {'organization': ORGANIZATION}
        self._stack: ContextVar[tuple[dict[str, Any], ...]] = ContextVar('research_span_stack', default=())
        self._context: ContextVar[dict[str, Any]] = ContextVar('research_span_attributes', default={})
        self._lock = threading.RLock()
        self._seq = 0
        self._file = None
        self._warned = False
        self.status = 'FINISHED'
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open('x', encoding='utf-8')
        except OSError as error:
            self.warning(error)
        self.write('session_start', attributes=self.attributes)

    def warning(self, error):
        if not self._warned:
            self._warned = True
            try:
                print(f'observability unavailable ({type(error).__name__}); research continues',
                      file=sys.stderr, flush=True)
            except Exception:
                pass

    def write(self, event, **fields):
        if self._file is None:
            return
        try:
            with self._lock:
                self._seq += 1
                row = dict(format=FORMAT, event=event, seq=self._seq,
                           at_ns=time.time_ns(), pid=os.getpid(), **fields)
                self._file.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
                self._file.flush()
        except Exception as error:
            self.warning(error)
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    @property
    def context_attributes(self):
        return self._context.get()

    def annotate_context(self, **attributes):
        """Update this scoped invocation, e.g. after an actual solver callback.

        The enclosing context() restores the caller's context on return.
        No global, cross-run iteration counter is used.
        """
        try:
            self._context.set(self._context.get() | scalars(attributes))
        except Exception as error:
            self.warning(error)

    @contextmanager
    def context(self, **attributes):
        try:
            token = self._context.set(self._context.get() | scalars(attributes))
        except Exception as error:
            self.warning(error)
            yield
            return
        try:
            yield
        finally:
            try:
                self._context.reset(token)
            except Exception as error:
                self.warning(error)

    @contextmanager
    def span(self, name, *, attributes=None, detached=False):
        try:
            stack = self._stack.get()
            parent = stack[-1] if stack else None
            span_id = uuid.uuid4().hex
            trace_id = parent['trace_id'] if parent and not detached else uuid.uuid4().hex
            attrs = self.attributes | self._context.get() | scalars(attributes or {})
            attrs.update(phase=name.split('.', 1)[0],
                         trace_level='detail' if name in DETAIL_OPERATIONS else 'phase')
            if parent and detached:
                attrs = attrs | {'parent_span_id': parent['span_id']}
            row = dict(span_id=span_id, trace_id=trace_id,
                       parent_id=parent['span_id'] if parent and not detached else None,
                       parent_trace_id=parent['trace_id'] if parent and detached else None,
                       name=name, attributes=attrs)
            started = time.perf_counter()
            self.write('span_start', **row)
            token = self._stack.set(stack + (row,))
        except Exception as error:
            self.warning(error)
            yield {}
            return
        status, outcome = 'OK', {}
        try:
            yield outcome
        except BaseException as error:
            status = 'ERROR'
            outcome['error.type'] = type(error).__name__
            raise
        finally:
            try:
                self._stack.reset(token)
                self.write('span_end', span_id=span_id, trace_id=trace_id, status=status,
                           attributes=scalars(outcome), duration_seconds=time.perf_counter() - started)
            except Exception as error:
                self.warning(error)

    def metric(self, name, value, *, step=0, attributes=None):
        if (name.split('.', 1)[0] not in ('native', 'solver', 'proposal', 'ops', 'reference')
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or type(step) is not int or step < 0):
            return
        self.write('metric', name=name, value=value, step=step,
                   attributes=self._context.get() | scalars(attributes or {}))

    def close(self, status='FINISHED', **attributes):
        self.status = status
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            self.metric('ops.process_cpu_user_seconds', usage.ru_utime)
            self.metric('ops.process_cpu_system_seconds', usage.ru_stime)
            self.metric('ops.process_peak_rss_bytes', usage.ru_maxrss * 1024)
        except Exception as error:
            self.warning(error)
        self.write('session_end', status=status, attributes=scalars(attributes))
        if self._file is not None:
            try:
                self._file.close()
            except Exception as error:
                self.warning(error)
            self._file = None
