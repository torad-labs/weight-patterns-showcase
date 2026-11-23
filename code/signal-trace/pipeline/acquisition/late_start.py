"""What a sweep has to settle before a late-started pass (observe-stream-late, STREAM-RECORDER.md) may supply rows after a matrix was
written: which kinds of step a pass is, the complete comparison of a late pass with the ordinary one, and the record of the kinds that
have held. The recorder cannot do this for itself: it holds a seed to the engine at one position and takes every other row from the file.

Astra's admission (records/retrospectives/2026-11-14-late-start-admission-astra.md): the comparison is made inside the first declared
sweep, on its real writes, with the ordinary pass as comparator, once for each of the driver's distinct kinds of step, over the complete
captured output and metadata of every fitting answer; until a kind has held, a step of that kind takes its rows from the ordinary pass;
a mismatch fails the kind and stays visible. Nothing here runs a pass, writes a model or chooses a matrix: the driver does (Fable's),
and it binds what a seed is (text, positions, the tensors upstream of it, recorder, engine, card).

  kinds(written, requested)        the kinds of step a pass is: the matrix written since its seeds were recorded, and the matrix whose
                                   rows it reads. Only the two neighbours of the path order are steps; anything else is refused.
  same(ordinary, late, ids)        every answer's stream.f32 and stream.jsonl byte for byte, and result.json equal but for the late
                                   pass's own `late_start` record
  Held(path, scope)                the sweep's record. .may(kinds) says whether a late pass alone may supply a step's rows;
                                   .note(kinds, written, requested, compared, **bound) writes one comparison under every kind it was
                                   a step of that had not been compared yet.

How a driver uses it, a step at a time:
    step = late_start.kinds(written, requested)
    if held.may(step):   rows from the late pass alone
    else:                rows from the ordinary pass; where seeds exist, the late pass too, kept raw, then
                         held.note(step, written, requested, late_start.same(ordinary_raw, late_raw, ids), ...)"""
import filecmp, json, time
from pathlib import Path

ATTENTION, FEEDFORWARD = 'attn_output', 'ffn_down'   # the two matrices a layer has, as their tensors are named
# The layers from this one on keep no cache of their own and read layer 13's or 14's (STREAM-RECORDER.md, stop_after_last_label: "the
# layers that share another layer's cache (15 to 34) read layer 13 or 14"): a late pass from here on is also seeded with the two states
# those caches are made from, l_out-12 and l_out-13, which is one more dependency a write may or may not have left as it was.
SHARED_FROM = 15
KINDS = ('attention written, its feed-forward read', 'feed-forward written, the next attention read', 'the shared caches\' two states handed on')


def of_name(name):
    parts = name.split('.')
    if len(parts) != 4 or parts[0] != 'blk' or parts[3] != 'weight' or parts[2] not in (ATTENTION, FEEDFORWARD) or not parts[1].isdigit():
        raise ValueError(f'{name} is no matrix a sweep writes')
    return int(parts[1]), parts[2]


def kinds(written, requested, shared_from=SHARED_FROM):
    """The kinds of step a late pass is that reads `requested`'s rows with seeds recorded before `written` was written. A late pass is
    seeded with the two outputs the pass before it recorded, which lie before the written matrix: so `requested` is the matrix right
    after `written` in the path order, and any other pair is no step a seed exists for."""
    (wrote, what), (reads, which) = of_name(written), of_name(requested)
    if (what, which, reads - wrote) == (ATTENTION, FEEDFORWARD, 0):
        step = [KINDS[0]]
    elif (what, which, reads - wrote) == (FEEDFORWARD, ATTENTION, 1):
        step = [KINDS[1]]
    else:
        raise ValueError(f'{requested} is not the matrix after {written} in the path order: a late start is seeded from the pass before it and from no other')
    return step + ([KINDS[2]] if reads >= shared_from else [])


def same(ordinary, late, ids):
    """A late pass against the ordinary pass of the same cases, answer by answer, whole: the two stream files byte for byte and the
    result equal but for the late pass's own record. Returns what was found for every answer; nothing is summarized away."""
    found = []
    for name in ids:
        a, b = Path(ordinary)/name, Path(late)/name
        whole, started = json.loads((a/'result.json').read_text()), json.loads((b/'result.json').read_text())
        record = started.pop('late_start', None)
        found.append(dict(id=name, stream_f32=filecmp.cmp(a/'stream.f32', b/'stream.f32', shallow=False), stream_jsonl=filecmp.cmp(a/'stream.jsonl', b/'stream.jsonl', shallow=False),
                          result=whole == started, started_late=bool(record) and record['started_late'] > 0, bytes=(a/'stream.f32').stat().st_size,
                          seeds=[dict(label=s['label'], used=s['used'], compared=s['compared_in_the_whole_compute'], written=s['written']) for s in (record or {}).get('seeds', [])]))
    return found


def held_by(compared):
    """Whether a comparison held: every answer the same in all three files, and every one of them actually started late (a pass that
    read every compute whole is the ordinary pass twice and shows nothing)."""
    return bool(compared) and all(each['stream_f32'] and each['stream_jsonl'] and each['result'] and each['started_late'] for each in compared)


class Held:
    """A sweep's record of the kinds of step compared on its own writes. `scope` is what the comparisons are good for and nothing
    else: the two recorders, the engine's libraries, the card, the model the sweep starts from, the cohort and how its text is read.
    A record made under another scope answers no question here."""

    def __init__(self, path, scope):
        self.path, self.scope = Path(path), scope
        if self.path.exists():
            self.record = json.loads(self.path.read_text())
            if self.record['scope'] != scope:
                raise ValueError(f'{self.path} was made under another scope (recorders, libraries, card, model, cohort or reading): its comparisons are not this sweep\'s')
        else:
            self.record = dict(what='The kinds of step of this sweep at which a late-started pass was compared, whole, with the ordinary pass after a real write '
                                    '(pipeline/acquisition/late_start.py; Astra, records/retrospectives/2026-11-14-late-start-admission-astra.md). A kind that is not here '
                                    'has not been compared: its steps take the ordinary pass. A kind that did not hold stays here as it was found and is never compared again.',
                               scope=scope, kinds={})

    def may(self, step):
        """Whether a late pass alone may supply the rows of a step of these kinds: each of them compared on this sweep's writes, and held."""
        return all(kind in self.record['kinds'] and self.record['kinds'][kind]['held'] for kind in step)

    def open(self, step):
        """The kinds of a step that have not been compared yet. A kind that failed is not open: it stays failed."""
        return [kind for kind in step if kind not in self.record['kinds']]

    def note(self, step, written, requested, compared, **bound):
        """One comparison, written under every kind of the step not compared before. `bound` is what the driver holds the pair to
        (the states before and after the write, the seeds' origin) and is kept as given."""
        fresh = self.open(step)
        if not fresh:
            raise ValueError('Every kind of this step has its comparison already: a kind is compared once, on the first write that makes it a step')
        entry = dict(held=held_by(compared), written=written, requested=requested, answers=compared, bytes_compared=sum(each['bytes'] for each in compared),
                     at=time.strftime('%Y-%m-%d %H:%M:%S %Z'), **bound)
        for kind in fresh:
            self.record['kinds'][kind] = entry
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.record, indent=1) + '\n')
        return entry['held']
