"""Read verified excerpt bytes in the original capture's coordinates."""
from bisect import bisect_right
from collections import OrderedDict
import json
import mmap
from pathlib import Path

from pipeline.native_records.group_observation import NativeReader


class ExcerptBytes:
    def __init__(self, file, index):
        self.packed = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self.packed) != index['packed_bytes']:
            self.packed.close()
            raise ValueError('Excerpt length differs from its published index')
        self.size = index['source']['bytes']
        self.segments = sorted(index['segments'], key=lambda s:s['source_offset'])
        self.starts = [s['source_offset'] for s in self.segments]

    def __len__(self):
        return self.size

    def __getitem__(self, span):
        if not isinstance(span, slice) or span.step not in (None, 1):
            raise ValueError('Only exact source-byte intervals are readable')
        start, stop = span.start, span.stop
        i = bisect_right(self.starts, start)-1
        while i >= 0:
            segment = self.segments[i]
            if segment['source_offset'] + segment['bytes'] >= stop:
                begin = segment['packed_offset'] + start - segment['source_offset']
                return self.packed[begin:begin+stop-start]
            i -= 1
        raise ValueError(f'Original byte interval {start}:{stop} is not present in this excerpt')

    def close(self):
        self.packed.close()


class ExcerptReader(NativeReader):
    def __init__(self, directory):
        directory = Path(directory)
        self.index = json.loads((directory/'excerpt.json').read_text())
        if self.index['complete_source'] is not False:
            raise ValueError('Excerpt identity must explicitly retain its partial-source scope')
        self.file = (directory/'chunks.bin').open('rb')
        try:
            self.data = ExcerptBytes(self.file, self.index)
        except BaseException:
            self.file.close()
            raise
        self.verified = OrderedDict()
