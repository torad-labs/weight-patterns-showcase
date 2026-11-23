# observe-stream: reduced-retention recorder

`observe-stream` is built from `observe_stream.cpp`, a copy of the
physics-weight-edit `observe.cpp` with one added retention mode. Existing
binaries (`observe`, `observe-selected`) and `signal_trace.hpp` are unchanged.

```
observe-stream MODEL OUT CASES BACKENDS capture residual|projection stream
```

Run with the usual environment (`CUDA_DEVICE_ORDER=PCI_BUS_ID`,
`CUDA_VISIBLE_DEVICES=N`, `GGML_CUDA_DISABLE_FUSION=1`) and the GPU flock.

## What it records

At every token, outputs only. No input preimages and no per-operation
`cudaDeviceSynchronize`; the scheduler computes up to each requested node and
the after-callback reads the finished output.

- `residual`: `inp_scaled`, `attn_post_norm`, `attn_out` (attention merge),
  `ffn_post_norm`, `pe_in` (feed-forward merge), `per_layer_embd_out`,
  `pe_out` (the unnamed per-layer-embedding merge ADD), `l_out`, `result_norm`.
  This gives all three residual merges per layer: incoming state, added
  contribution and result.
- `projection`: adds the weighted-projection inputs `attn_norm`, `ffn_norm`,
  `kqv_out`, `ffn_geglu`, and `Qcur_pos`, `Kcur_pos`, `Vcur_normed` (enough to
  rebuild the KV cache and attention offline).

At any `trace_input_positions` declared in a case, it runs the unchanged
complete `signal_trace` capture (all operations, input preimages, cache links)
and also writes the stream records, so the stream is complete for every token.

Files per case: `stream.f32` (logical x-fast float32 values), `stream.jsonl`
(one record per output: label, native name, operation, shape, strides,
positions, byte offset, source names), plus the usual `result.json`, and
`signal-data.bin` / `signal-events.jsonl` when trace positions are declared.

## Opt-in case fields (2026-11-09)

Both are per case and change nothing when absent.

- `watched_tokens`: a list of token ids. Each step in `result.json` gets
  `watched_logits`, the scores of those tokens at that decision, next to the
  top eight. Works in every mode, including `plain`.
- `stream_positions`: a list of decode input positions. The stream is written
  only at those positions; at every other token the callback requests nothing,
  so the token runs as ordinary inference. The decision that chooses generated
  token `s` is computed at input position `prompt_tokens + s - 1`, which is
  `steps[s].input_position`.

Qualification of both is in
`studies/04-round-2-overlap/runs/frame-decisions/qualification/`.

- `stream_labels` (2026-11-10): a regular expression. The stream keeps only the
  outputs whose label matches it whole, such as `kqv_out-3|l_out-2`; nothing
  else is requested from the scheduler. With an answer's own tokens appended to
  the question as `tokens` and `max_tokens: 1`, the whole text is read in the
  prefill microbatches and the named outputs are written at every position,
  which is how the path-ordered edit reads one matrix's rows per pass. Checked
  on the original model: after the whole text of `discovery-05-channel-h1` the
  model chooses the answer's recorded last token, and six answers with six
  labels take 11 s including the model load. Stream retention only.

- `patch`: `{"position": p, "vectors": file, "targets": [{"label": L, "row": k}, ...]}`.
  At decode input position `p`, row `k` of the float32 file (rows of the
  residual width) is added to the output named `L`, for every target, and the
  rest of the pass runs on the changed values. `L` is a stream label such as
  `attn_out-16`, `pe_in-16` or `pe_out-16`. Nothing else is observed during
  that pass. Stream retention only.
- `patch_sweep`: `{"vectors": file, "passes": [[targets], [targets], ...]}`,
  with `teacher_tokens` holding an answer's own tokens up to and including the
  token of the decision. After the replay, the decision's own pass is removed
  from memory and run again once per entry of `passes`, with that entry's
  targets applied; an empty entry runs the pass unchanged. Each pass records
  the score hash, the top eight and the watched scores under
  `result.json: patch_sweep.passes`. An unchanged pass must reproduce the
  natural score hash of that decision; each run that uses a sweep checks this.

- `stop_after_last_label` (2026-11-11): `true` ends each compute once the last
  output the case asked for is written. Without it every token runs all 35
  layers, the final norm and the output head whatever `stream_labels` names;
  with it a pass that reads layer 0 does layer 0's work. It is for one kind of
  case only, an answer's own tokens supplied as `teacher_tokens` and read under
  `stream_labels`; with trace or stream positions, a patch, a patch sweep,
  watched tokens, a temperature, or without supplied tokens, the recorder
  throws. A case without the field takes the path it always took.
  - How it stays exact. The engine builds one graph per microbatch shape (token
    count, with or without an output: four shapes in an answer). The first
    compute of each shape runs whole and lists the wanted outputs in the order
    they were written. Every later compute of that shape must write the same
    labels in the same order, or the recorder throws, and it ends at the last
    of them. Each layer's cache is written before it is read, the layers that
    share another layer's cache (15 to 34) read layer 13 or 14, always below
    them, and every case starts from a cleared cache, so nothing a wanted
    output depends on lies after the stop.
  - What a stopped case no longer has. No vocabulary scores are computed, so
    `result.json` steps carry no `logits_sha256`, `top`, `greedy_token` or
    `selected_logit`, and `score_storage` says so; the supplied tokens are fed
    as they are. The finite check of the scores is replaced by one on every
    recorded value. `result.json: stop_after_last_label` records the computes,
    how many ran whole (one per shape), how many stopped, and the callbacks
    seen after a stop (0). A case in which a compute after the first of its
    shape did not stop throws.
  - Not for free generation: the model's own answers are read without the
    field (Astra, November 11,
    `records/retrospectives/2026-11-11-native-reference-decision-astra.md`,
    'Subsequent recorder decision: E's own answers').

This is **reduced retention**, not a complete weighted trace. Between declared
positions it does not keep input preimages, cache links or intermediates other
than those listed. Label analyses that use it accordingly.

## Qualification (2026-11-08, original BF16, GPU0 RTX 5080)

Records in `runs/legs/gemma4-e2b/signal-trace-A-2026-11-07/stream-qualification-2026-11-08/`.
Checked with `qualify_stream.py` against the historical baseline and the
verified full-recorder excerpt of `block-changed`.

| Run | Tokens / score hashes | Bit-identical vs full recorder | Different | Wall time |
| --- | --- | ---: | ---: | ---: |
| residual, natural, 480 decisions | equal / all 480 equal | 55,081 | 0 | 35.3 s (includes cold model load) |
| projection, natural, 480 decisions | equal / all 480 equal | 100,796 | 0 | 14.9 s |
| residual + trace at 432–436 (Astra's selected qualification case) | equal / all 265 equal | `signal-events.jsonl` and `signal-data.bin` SHA-256 identical to `observe-selected` | none | 13.1 s |

Full recorder on a comparable response: ~375 s, 9.3 GB native bytes.
Stream outputs outside the excerpt's passages (64,714 residual, 118,424
projection records) have no stored full-recorder counterpart and are counted,
not assumed equal.

GPU1 qualification (`expanded-043-h3`, projection scope, against its GPU1
baseline and excerpt) is queued on the GPU1 lock; check with:

```
python3 code/signal-trace/qualify_stream.py <Q>/gpu1/expanded-043-h3 \
  <R>/dual-gpu/gpu1-baseline/expanded-043-h3/result.json \
  <R>/individual-observation/expanded-043-h3 <Q>/gpu1-qualification.json
```

Until that passes, treat GPU1 stream captures as unqualified.

## Qualification of `stop_after_last_label` (2026-11-11, original BF16, GPU0 RTX 5080)

Records in `studies/04-round-2-overlap/runs/controlled-bf16-2/recorder-qualification/`
(`prepared.json`, `comparison.json`, `execution-whole.json`,
`execution-stopped.json`), made under that run's declaration before any of its
passes, by
`records/development/2026-11-10-speed-review/recorder-early-stop/reviewed/qualify_early_stop.py`.
The change is `observe_stream-early-stop.consolidated.patch` in that folder,
after three independent readers and a consolidator (`review.json`).

The pair compared differs by the patch and nothing else: the recorder's compile
line on the source before the patch gives the binary that was installed until
now, `f9dab98d...`, whole digest; on the patched source (sha256 `525851af...`)
it gives `0b4ea036c2c33711e76586d28f773bbfa5ba5463671656ec46869908e74fe060`,
which is the binary installed here since 8:26 AM CT on November 11.

Nine label sets as `tube` asks for them (a build read at layer 0's attention,
layer 14's attention and feed-forward, layer 15's and layer 19's attention, the
closing read, and the baseline groups starting at layers 0, 15 and 30) on four
answers, read through the original file by both binaries on card 0.

| Compared | Cases | Result |
| --- | ---: | --- |
| stopped case against the same case read whole by `f9dab98d` | 36 | `stream.f32` and `stream.jsonl` byte-identical in all; the stop record is the expected one in all (every compute but the first of each shape stopped, no callback after a stop) |
| the same case without the field, through `0b4ea036`, against `f9dab98d` | 9 | `result.json` equal whole, every `logits_sha256` included, and both stream files byte-identical |

The nine controls are replays with `teacher_tokens` and `stream_labels`; they
are not free generations. That the unfiltered natural path is unchanged rests
on the source (the field defaults to false and every changed line is behind
it), as Astra's decision above says.

Seconds a case on card 0, from the times the cases' results were written (the
first case of each process left out: it carries the model's load):

| Label set | Last layer read | `f9dab98d` | `0b4ea036`, stopped |
| --- | ---: | ---: | ---: |
| a build read at layer 0's attention | 0 | 4.67 | 0.13 |
| a baseline group, layers 0 to 4 | 4 | 4.64 | 0.85 |
| a build read at layer 14's attention | 14 | 4.29 | 1.10 |
| a build read at layer 14's feed-forward | 14 | 4.27 | 1.10 |
| a build read at layer 15's attention | 15 | 4.29 | 1.15 |
| a build read at layer 19's attention | 19 | 4.30 | 1.53 |
| a baseline group, layers 15 to 19 | 19 | 4.71 | 2.04 |
| the closing read | 34 | 4.25 | 3.03 |
| a baseline group, layers 30 to 34 | 34 | 4.73 | 3.53 |

The whole pass of 36 cases took 166.7 s and the stopped pass with its nine
controls 88.7 s. The same comparison on GPU1 (RTX 5070 Ti) held in all 45 cases
the night before (`reviewed/card1-development/`); that was a development
reading, and GPU1 stream captures stay unqualified as said above.

Scope: this pair of programs, this model, this engine build and card 0. The
kernels are compiled for the card and a stopped compute launches a piece of the
graph, so another engine build, card or model needs the comparison again.
`qualify_early_stop.py` holds the hashes of this pair and refuses another.

## `observe-stream-late`: a pass started late (2026-11-14, a second program beside the first)

`observe-stream` and `observe_stream.cpp` are not changed by this: the binary
`0b4ea036...` and its source `525851af...` stay what every declaration that
names them pinned. The late start is a second program, installed beside it at
11:11 AM CT on November 14:

| File | sha256 |
| --- | --- |
| `observe_stream.late.cpp` (629 lines) | `c095fb1a55bf46399063417319a78d1ebedbe3ea0e4e629e912fe70f454a21c8` |
| `observe-stream-late` | `13cc39349560a22519da8b14457049cc36cd4bd14b7cd45368c817d4cba12843` (`observe-stream-late.identity.json` holds its second identity, without the library folder's name) |

The source is `observe_stream.cpp` with five blocks added and nothing removed:
`records/development/2026-11-10-speed-review/late-start/make_late.py` makes it
from the installed source and refuses another source, and
`observe_stream-late-start.patch` in that folder is the difference. The binary
is the recorder's compile line on that source and is, byte for byte, the binary
the qualification below ran (the source keeps the file name it was compiled
under then, because the name is in the binary's symbol table).

A case without the new field takes the path it always took and gives the
bytes `observe-stream` gives (the mixed controls below). Nothing uses this
program by default: a run names it and its sha256 in its own declaration.

- `late_start` (2026-11-14): `{"seeds": [{"label": L, "vectors": file, "width": w, "first_row": r}, ...]}`,
  for one kind of case only, the one `stop_after_last_label` is for (an answer's
  own tokens supplied as `teacher_tokens`, read under `stream_labels`, with
  `stop_after_last_label: true`); without that field the recorder throws. A
  seed is one recorded output of an earlier pass over the same text through the
  same weights below it: float32 rows of `width`, one for every input position
  of the case in position order, beginning at row `first_row` of the file. A
  seed may not be one of the wanted outputs, and a label is named once.
  - What a late case computes. The prefill is computed whole, as always. The
    case's first single-token compute is computed whole too: every node the
    scheduler asks is listed (operation, name, label, flags, sources, the cache
    leafs it reads and writes), and each seed's node is compared with the
    seed's row at that position byte for byte; a difference throws. From the
    list a plan is made once: the nodes the wanted outputs need, walking back
    from them and stopping at the seeds, plus every node that writes a cache a
    needed node reads. In every later single-token compute a node the plan
    does not need, from `attn_norm-0` on, is not computed; a needed seed's node
    is not computed either and gets the seed's row for that position written
    into it; everything else is computed as always. Each compute must ask the
    listed nodes in the listed order (operation and name), or the recorder
    throws, and the compute flags it cleared are put back after each decode.
  - A seed no wanted output is computed from is compared in the whole compute
    and written nowhere (`used: false`), so a driver can always pass the two
    states the shared caches are made from, `l_out-12` and `l_out-13`, from
    layer 15 on.
  - `result.json: late_start` records each seed (label, node, used, compared,
    rows written), the single-token computes, how many ran whole (one) and how
    many started late, the nodes listed, needed, skipped and seeded in a late
    compute, and the flags put back.
  - What a seed has to be is the driver's to bind, not the recorder's to know:
    the recorder holds a seed to the engine at one position and takes the other
    rows from the file. A seed belongs to a pass only when the text, the
    positions, the model's tensors upstream of it, the recorder, the engine and
    the card are the ones it was recorded under
    (`records/retrospectives/2026-11-14-late-start-admission-astra.md`, "What
    proves the later rows belong").

### Qualification of `late_start` (2026-11-14, original BF16, GPU0 RTX 5080)

Records in `records/development/2026-11-10-speed-review/late-start/`, made by
`late_start_check.py` there; the reading of them is the speed review's
`README.md`, "A pass started late gives the same bytes". Every comparison is
of a late case against the same case read by `observe-stream` under
`stop_after_last_label`: `stream.f32` and `stream.jsonl` byte for byte, and
`result.json` equal but for the `late_start` record.

| Compared | Cases | Result |
| --- | ---: | --- |
| the 70 matrices in path order, each pass's seeds the two outputs the LATE pass before it wrote (and `l_out-12`, `l_out-13` from layer 15 on), three answers (`chain-three-answers.json`) | 207 | the same bytes in all (69 of 69 matrices after the first) |
| one process reading a late case, a case without the field, then a late case, at three places of that chain | 9 | the same bytes in all; the case without the field carries no `late_start` record |
| the thirty fitting answers at layer 14's feed-forward and layer 15's attention (`cohort-thirty-answers-28-30.json`) and at layer 34's feed-forward (`cohort-thirty-answers-68-69.json`) | 90 | the same bytes in all |
| seeds given as a sweep's driver gives them (`SEEDS=shared`): one file a label holding every answer's rows, the answers laid in the reverse of the pass's order, each case told its `first_row` (1,738, 940 and 0); layer 15's attention and feed-forward, three answers (`shared-seeds-three-answers-29-31.json`, 11:46 AM CT) | 6 | the same bytes in all |

A late compute holds 11 to 59 needed nodes of the 1,496 a compute lists. On the
thirty answers a pass after its first answer is 9.5 to 11.9 s where the whole
one is 38 to 102 s at the same load.

Admitted by Astra as an opt-in revision
(`records/retrospectives/2026-11-14-late-start-admission-astra.md`, 774a4c04),
with what is not yet shown stated there and here: both passes of every pair
above read the same file. A pass whose seeds were recorded before a matrix was
written and which reads the file after the write has not been compared; that
comparison is made inside the first declared sweep, on its real writes, with
the ordinary pass as comparator, at the driver's three kinds of step (an
attention output written and the feed-forward after it read, a feed-forward
written and the next attention read, and the shared caches' two states handed
on), and until a kind of step has held, its rows come from the ordinary pass.
It is a way to read a matrix's rows on a supplied text: it computes no
vocabulary scores and says nothing about whether the text is the model's own.

How a pass takes it, and how a sweep settles a kind of step
(`pipeline/acquisition/`, 14 tests in `tests/test_late_start.py`):

- `run_a_observation.run(..., program='observe-stream-late')` runs this
  program; without `program` every pass runs the recorder it always ran, and a
  pass whose cases carry `late_start` without it is refused (`observe-stream`
  would read such a case whole and say nothing). `execution.json` records the
  program's sha256 as `observer_sha256`, as for every pass.
- `late_start.kinds(written, requested)` names the kinds of step a late pass
  is (the matrix written since its seeds were recorded, the matrix whose rows
  it reads): only the two neighbours of the path order are steps, and from
  layer 15 on the shared caches' two states are a kind too.
  `late_start.same(ordinary, late, ids)` is the whole comparison (both stream
  files byte for byte, the result equal but for the late record, every answer
  actually started late). `late_start.Held(path, scope)` is the sweep's record:
  `.may(kinds)` says whether a late pass alone may supply a step's rows, and
  `.note(...)` writes one comparison under each kind not compared before. A
  kind that did not hold stays in the record as found and is not compared
  again; a record made under another scope (recorders, libraries, card, model,
  cohort, reading) is refused.

Scope: this pair of programs, this model, this engine build and card 0.
