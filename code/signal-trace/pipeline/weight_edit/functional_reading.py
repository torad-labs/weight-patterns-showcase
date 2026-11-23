"""Load a declared native reading in causal order for functional comparisons.

Only the caller's explicit member inventory is admitted. Tokens identify the
acquired input history; no token comparison grades an answer or selects loss
support. The reader's answer-first storage layout is not computational order.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from . import working_identity


def digest(path: Path) -> str:
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def tokens_hash(tokens: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(value) for value in tokens], separators=(',', ':')).encode()).hexdigest()


def identity(path: Path) -> tuple[int, ...]:
    found = path.stat()
    return found.st_dev, found.st_ino, found.st_size, found.st_mtime_ns, found.st_ctime_ns


@dataclass(frozen=True)
class Case:
    history: tuple[int, ...]
    native: dict[str, list[torch.Tensor]]
    question: int
    complete: bool | None
    ended: bool


class Reading:
    def __init__(self, folder, model, members: Mapping[str, Sequence[int]], *,
                 expected_reading_sha256: str, expected_model_sha256: str | None = None,
                 expected_model_identity: dict | None = None):
        self.folder, self.model = Path(folder), model
        self.members = {name: tuple(int(value) for value in prompt) for name, prompt in members.items()}
        if not self.members or any(not name or name in ('.', '..') or '/' in name or '\\' in name for name in self.members):
            raise ValueError('the reading needs an explicit safe member inventory')
        self._verified: dict[Path, tuple[tuple[int, ...], str]] = {}
        self._arrays: dict[int, np.ndarray] = {}
        self.record = json.loads(self._file('reading.json', expected_reading_sha256).read_text())
        if self.record.get('format') != 'whole-gap-reading/1':
            raise ValueError('not a whole native reading')
        self.model_state = None
        if expected_model_identity is not None or self.record.get('model_identity') is not None:
            if expected_model_identity is None or expected_model_sha256 is not None:
                raise ValueError('a working-model reading needs its revision identity, not a file sha256')
            self.model_state = Path(model.file).parent / 'state.json'
            state = working_identity.load_state(self.model_state, model=model.file,
                                                 expected_identity=expected_model_identity)
            if working_identity.from_record(self.record) != state['model_identity']:
                raise ValueError('the reading and ready working model have different identities')
            self.model_identity = state['model_identity']
            self._model_stat = identity(Path(model.file))
        else:
            if (not expected_model_sha256 or self.record.get('model_sha256') != expected_model_sha256
                    or digest(Path(model.file)) != expected_model_sha256):
                raise ValueError('the reading and actual model file do not have the declared model identity')
            self.model_identity = working_identity.from_record(self.record)
            self._model_stat = identity(Path(model.file))
        if (len(self.record['answers_read']) != len(self.members)
                or set(self.record['answers_read']) != set(self.members)):
            raise ValueError('the native reading membership is not the explicit fitting inventory')
        row_binding = self.record['row_map']
        if row_binding['file'] != 'row-map.json':
            raise ValueError('the native row-map location is not canonical')
        self.row_map = json.loads(self._file(row_binding['file'], row_binding['sha256']).read_text())
        if self.row_map.get('order') != 'all_answers_then_all_questions':
            raise ValueError('unsupported native array row order')
        self.entries = {entry['id']: entry for entry in self.row_map['answers']}
        self.table = {entry['id']: entry for entry in self.record['table']}
        if (set(self.entries) != set(self.members) or len(self.entries) != len(self.row_map['answers'])
                or set(self.table) != set(self.members) or len(self.table) != len(self.record['table'])):
            raise ValueError('the native row and answer membership is incomplete or duplicated')
        rows = self.row_map['rows']
        if type(rows) is not int or rows < 1 or rows != self.record['rows'] or self.record['width'] != model.width:
            raise ValueError('the native reading dimensions do not match its model and rows')
        coverage = np.zeros(rows, dtype=np.int64)
        for entry in self.entries.values():
            for start_key, count_key in (('answer_rows_from', 'steps'), ('question_rows_from', 'question')):
                start, count = entry[start_key], entry[count_key]
                if type(start) is not int or type(count) is not int or start < 0 or count < 0 or start + count > rows:
                    raise ValueError('the native row ranges do not partition the capture')
                coverage[start:start + count] += 1
        if not bool((coverage == 1).all()):
            raise ValueError('the native row ranges do not partition the complete capture')
        self.nodes = self.record['merges']
        if [node['merge'] for node in self.nodes] != list(range(3 * model.layers)):
            raise ValueError('the native reading does not hold every model merge exactly once')
        for node in self.nodes:
            if (node['file'] != f"nodes/{node['merge']}/state.npy" or node['shape'] != [rows, model.width]
                    or node['dtype'] != '<f4'):
                raise ValueError('a native merge has a noncanonical file, shape or dtype')
        self.scales = tuple(model.w[f'blk.{layer}.layer_output_scale.weight'].detach().clone()
                            for layer in range(model.layers))

    @contextmanager
    def _model_access(self):
        """The sole controller owns the file; every read checks its ready revision."""
        self._check_model()
        yield
        self._check_model()

    def _check_model(self):
        if identity(Path(self.model.file)) != self._model_stat:
            raise ValueError('the native model file changed after verification')
        if self.model_state is not None:
            working_identity.load_state(self.model_state, model=self.model.file,
                                        expected_identity=self.model_identity)

    def _file(self, relative: str, expected: str) -> Path:
        part = Path(relative)
        if part.is_absolute() or any(item in ('', '.', '..') for item in part.parts):
            raise ValueError('a native artifact path leaves its reading')
        path = self.folder
        if path.is_symlink():
            raise ValueError('a native reading is not reached through a link')
        for item in part.parts:
            path = path / item
            if path.is_symlink():
                raise ValueError('a native artifact is not reached through a link')
        before = identity(path)
        if path in self._verified:
            if self._verified[path] != (before, expected):
                raise ValueError('a native artifact changed after sha256 verification')
            return path
        if digest(path) != expected or identity(path) != before:
            raise ValueError(f'native artifact sha256 differs or changed during reading: {relative}')
        self._verified[path] = before, expected
        return path

    def acquisition(self, name: str) -> tuple[tuple[int, ...], int, bool | None, bool]:
        with self._model_access():
            if name not in self.members:
                raise ValueError('the requested answer is not in the explicit fitting membership')
            entry, table = self.entries[name], self.table[name]
            answer = json.loads(self._file(f'answers/{name}/answer.json', table['answer_json_sha256']).read_text())
            prompt, text = self.members[name], answer['tokens']
            question, steps = entry['question'], entry['steps']
            if (len(prompt) != question + 1 or tokens_hash(prompt) != entry['prompt_sha256']
                    or answer['prompt_sha256'] != entry['prompt_sha256']):
                raise ValueError('the supplied prompt is not the native capture input')
            if (steps < 1 or len(text) != steps or answer['steps'] != steps or answer['question'] != question
                    or tokens_hash(text) != entry['text_sha256'] or answer['text_sha256'] != entry['text_sha256']):
                raise ValueError('the native acquired history differs from its bound record')
            history = prompt + tuple(int(value) for value in text[:-1])
            complete = table.get('the_whole_answer_is_this_models_own')
            if complete is not None and not isinstance(complete, bool):
                raise ValueError('native acquisition completeness is not a boolean or unresolved')
            if answer.get('first_step_not_its_own') is not None or answer.get('steps_not_its_own') != 0:
                complete = False
            return history, question, complete, answer.get('ended_with_an_end_token') is True

    def state(self, name: str, index: int) -> torch.Tensor:
        """One observed merge in causal order; no other merge array is opened."""
        with self._model_access():
            if name not in self.members or type(index) is not int or not 0 <= index < len(self.nodes):
                raise ValueError('unknown fitting member or native merge')
            node, entry = self.nodes[index], self.entries[name]
            path = self._file(node['file'], node['sha256'])
            if index not in self._arrays:
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if array.dtype != np.dtype('<f4') or list(array.shape) != node['shape']:
                    raise ValueError('native state bytes have another shape or dtype')
                self._arrays[index] = array
            array = self._arrays[index]
            q, a = entry['question_rows_from'], entry['answer_rows_from']
            values = np.concatenate((array[q:q + entry['question']], array[a:a + entry['steps']]))
            if not bool(np.isfinite(values).all()):
                raise ValueError('a native state has nonfinite values')
            return torch.from_numpy(values).to(self.model.device)

    def incoming(self, name: str, index: int) -> torch.Tensor:
        with self._model_access():
            if type(index) is not int or not 0 <= index < len(self.nodes):
                raise ValueError('unknown native merge')
            if index == 0:
                history, _, _, _ = self.acquisition(name)
                return self.model.rows('token_embd.weight', history) * math.sqrt(self.model.width)
            previous = self.state(name, index - 1)
            return previous * self.scales[index // 3 - 1] if index % 3 == 0 else previous

    def read(self, name: str) -> Case:
        with self._model_access():
            history, question, complete, ended = self.acquisition(name)
            merged = [self.state(name, index) for index in range(len(self.nodes))]
            incoming = [self.model.rows('token_embd.weight', history) * math.sqrt(self.model.width)]
            for index in range(1, len(merged)):
                incoming.append(merged[index - 1] * self.scales[index // 3 - 1]
                                if index % 3 == 0 else merged[index - 1])
            return Case(history, dict(incoming=incoming, merged=merged), question, complete, ended)
