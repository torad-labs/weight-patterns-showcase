"""The whole model as the engine computes it, in torch, with gradients. Files and a card's arithmetic only: no engine, no pass, no loss.

  (from code/signal-trace)
  python torch_forward.py describe <out architecture.json>
  python torch_forward.py check <run or pass that holds reading/> <run that holds capture/ with each answer's case.json> <out .json> [answers=<n>] [device=cuda] [cache=float16] [model=<a candidate file, from the repository's root>]
  python torch_forward.py write <run or pass that holds reading/> <run that holds capture/ with each answer's case.json> <out .json> [answers=<n>] [device=cuda] [model=<a candidate file, from the repository's root>]

Why it exists. Marcos's statement of the loss (relayed November 15, 3:44 to 3:52 AM CT; journal/2026-11-15-computation-not-state-values.md):
the trace is what each merge does to what it is given, the shortfall of a run's trace from the correct answers' overlap is the loss, and the
loss is taken back through the weights, the model run again and the trace measured again. Taking a loss back through the weights needs the
model's computation as one differentiable graph, which the engine does not give and nothing here had: the tree held one block of it
(fixed_input_movement.Block, the feed-forward block) and the last merge (field.Final).

What it is. `Model` reads a model file's tensors at the original file's offsets and computes, for one text held fixed, what the engine's
graph computes (dependencies/llama.cpp/src/models/gemma4.cpp, with build_attn_mha and the key-value layers of llama-graph.cpp and
llama-model.cpp, and the card's rotary kernel, ggml-cuda/rope.cu):
    x          = embedding[token] * sqrt(width)                                                       inp_scaled
    per layer  : h = rms(x) * attn_norm
                 q = rope(rms(q h) * q_norm), eight heads; k = rope(rms(k h) * k_norm), v = rms(v h): ONE head of keys and values
                 the first 15 layers make k and v; a later layer reads layer 13's (a windowed layer) or layer 14's (a whole-text layer)
                 (the engine's cache holds k and v in float16 and its attention kernels compute in float16; here they are float32
                 unless cache='float16' is asked: see check, and the record page, for why that is the nearer forward)
                 attn_out = x + rms(o . softmax(q k, causal, 512 back on a windowed layer) v) * post_attention_norm      merge 3L
                 pe_in    = attn_out + rms(down(gelu(gate f) * up f)) * post_ffw_norm,  f = rms(attn_out) * ffn_norm      merge 3L+1
                 pe_out   = pe_in + rms(proj(gelu(inp_gate pe_in) * per_layer_input[L])) * post_norm                      merge 3L+2
                 x        = pe_out * layer_output_scale                                                                   l_out
    per_layer_input = (rms(per_layer_model_proj x0 / sqrt(width)) * per_layer_proj_norm + per_layer_token_embd[token] * 16) / sqrt(2)
`forward(tokens)` gives every merge's incoming and merged state at every position, as one graph whose leaves are the matrices an edit
writes (the 35 attn_output and 35 ffn_down by default; `leaves` takes any choice of the file's tensors). A function of those pairs is
one number, and one backward gives its derivative in every leaf. Each layer is made again during the backward (torch's checkpoint), so
the graph held at once is one layer's.

`write(prompts, limit, ends)` writes the text a model answers each prompt with, as the recorder does (the vocabulary's first choice at
each position), several prompts side by side with a cache of keys and values between positions: the thirty answers in seconds where
the engine takes minutes. It is this forward's own run of the model: at a near tie of two tokens its choice can be the other one than
the engine's on a card (as the two cards' are of each other), and from there the text is another text.

What it is not. `forward` is the computation of a GIVEN text and reads no answer. `scores` and `write` pick tokens to write a text and
are held to the recorder's by `write` below; no token, score or wording is a measure of anything here. The engine stays the recorder of
the canonical protocol. It defines no loss, target or update: which function of the pairs is the loss is the objective's (Astra), what
is done with a derivative is the solver's (Fable), and which text and which trace a declared run measures on is the objective's too.

check  The original's file against the original's kept reading (every merge's output at every position of every fitting answer, as the
       recorder read it off the card), on the text the reading was made on. Per merge, each over the engine's own size: the merge
       ALONE (what this forward adds to the ENGINE's own input of that merge, against what the engine added), told apart by how the
       engine made the position (the prompt's positions in batches, an answer's one at a time: its attention runs another kernel for
       each); the state end to end (every layer fed by this forward's own layer before); the turn end to end, and its cosine; the
       positions whose worst merge stands far from the engine, beside how far the SAME forward stands from itself when only the cache's
       float16 rounding is changed (a change of about 1e-4 of k and v): the positions the model itself computes unstably. Then a
       backward of a stand-in number (the sum of every merge's squared turn: no loss of anything) on every text, the derivatives
       added up, for the time and the card's memory. With model=<file> the file is a candidate (the original's layout, other values)
       and the reading is the engine's of that candidate on its own answers: the text is the reading's, the prompts the cohort's.
write  The writing held to the recorder's, on the model and the text of a kept reading. First, under the recorded text held fixed: this
       forward's scores at every decision beside the eight the recorder kept (where the capture's result.json is of that text), how
       near the engine's own first two choices stand at each decision, and the decisions where this forward's first choice is another
       token. Then the answers written by `write` from the cohort's prompts, the time, and for each answer the step at which its
       text leaves the recorded one. Last, each written text laid on the original's exported turns of its question (`registered`,
       the capture's rule of a registration from the forward's pairs): a text written as recorded must lie one on one, and a text
       that left it is held to the capture's own arithmetic on the processor. A comparison of two acquisitions of one model;
       nothing is judged.
describe  The file's architecture keys, read with the engine's own reader (gguf-py), written once beside the checks."""
import hashlib, json, math, sys, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

A, C = 0.044715, 0.79788456080286535587989211986876           # ggml's gelu
TABLES = ('token_embd.weight', 'per_layer_token_embd.weight')  # read by row from the file, never held whole
EDITABLE = lambda name: name.endswith(('.attn_output.weight', '.ffn_down.weight'))


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(C * x * (1 + A * x * x)))


def rms(x, weight=None, eps=1e-6):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y if weight is None else y * weight


def half(x):
    """What the engine's key and value cache holds: float16. The derivative passes straight through the rounding."""
    return x + (x.half().float() - x).detach()


def rope(x, cos, sin):
    """Pairs (i, i + n/2) of the last axis turned by each position's angle (rope_neox)."""
    n = x.shape[-1] // 2
    a, b = x[..., :n], x[..., n:]
    return torch.cat((a * cos - b * sin, a * sin + b * cos), -1)


def stored(model, t):
    """A tensor of a model file as it is stored, on the processor: BF16 as bfloat16 (the same bits), F32 as float32; a matrix as (out, in)."""
    shape = tuple([n for n in t['shape'] if n > 1][::-1] or [1])
    if t['native_type'] == 'bf16':
        return torch.from_numpy(np.array(np.memmap(model, dtype='<i2', mode='r', offset=t['offset_bytes'], shape=shape))).view(torch.bfloat16)
    if t['native_type'] != 'f32':
        raise ValueError(f"{t.get('name')}: {t['native_type']}")
    return torch.from_numpy(np.array(np.memmap(model, dtype='<f4', mode='r', offset=t['offset_bytes'], shape=shape)))


class Model:
    def __init__(self, model, tensors, architecture, device='cuda', leaves=EDITABLE, cache=None):
        a = self.a = architecture
        self.cache = cache                                       # None: keys and values as computed; 'float16': rounded as the engine's cache holds them
        self.kept = None                                         # inside `holding`: what is made once and held (the token table)
        self.file, self.tensors, self.device = Path(model), tensors, torch.device(device)
        self.layers, self.width, self.heads, self.per = a['layers'], a['width'], a['heads'], a['per_layer_width']
        self.windowed, self.keys_from = [bool(w) for w in a['windowed']], a['layers'] - a['layers_sharing_keys']
        if abs(a['norm_eps'] - 1e-6) > 1e-9:
            raise ValueError(f"the norms here add 1e-6, and the file says {a['norm_eps']}")
        if a['key_heads'] != 1:
            raise ValueError(f"this forward is written for one head of keys and values, and the file has {a['key_heads']}")
        unread = lambda name: name.startswith('blk.') and int(name.split('.')[1]) >= self.keys_from and name.split('.')[2] in ('attn_k', 'attn_v', 'attn_k_norm')
        self.w = {}
        for name, t in tensors.items():
            if name in TABLES or unread(name):                 # a late layer's own k and v are in the file and the engine never reads them
                continue
            x = stored(self.file, t).to(self.device)
            self.w[name] = x.float().requires_grad_(True) if leaves(name) else x
        self.leaves = {name: x for name, x in self.w.items() if x.requires_grad}

    def weight(self, name):
        x = self.w[name]
        return x if x.dtype == torch.float32 else x.float()

    @contextmanager
    def holding(self):
        """While inside, every stored matrix is its float32 on the device in place of its stored bits, and the token table is held
        once made: a text written one token at a time reads each of them at every token. On leaving, each matrix is its stored
        bits again (BF16 to float32 and back loses nothing) and the table is let go."""
        was = {name: x.dtype for name, x in self.w.items() if x.dtype != torch.float32}
        for name in was:
            self.w[name] = self.w[name].float()
        self.kept = {}
        try:
            yield self
        finally:
            self.kept = None
            for name, kind in was.items():
                self.w[name] = self.w[name].to(kind)

    def rows(self, name, tokens):
        """A token table's rows for `tokens`, float32 on the device, read from the file."""
        t = self.tensors[name]
        table = np.memmap(self.file, dtype='<u2', mode='r', offset=t['offset_bytes'], shape=(t['shape'][1], t['shape'][0]))
        return torch.from_numpy((np.asarray(table[np.asarray(tokens)]).astype(np.uint32) << 16).view(np.float32).copy()).to(self.device)

    def head(self, layer):
        return self.a['head_width_windowed'] if self.windowed[layer] else self.a['head_width']

    def keys_of(self, layer):
        """The layer whose keys and values a layer reads: its own, or for a layer past the last that makes them, the last windowed or whole-text one that does."""
        return layer if layer < self.keys_from else self.keys_from - (2 if self.windowed[layer] else 1)

    def rotation(self, layer, positions):
        """cos and sin of every position's angle at every pair, as the card's kernel makes them: pos * theta_scale^i / freq_factor, in float32."""
        n = self.head(layer)
        base = self.a['rope_base_windowed'] if self.windowed[layer] else self.a['rope_base']
        scale = torch.tensor(base, dtype=torch.float32, device=self.device).pow(torch.tensor(-2.0 / n, dtype=torch.float32, device=self.device))
        theta = positions.float()[..., None] * torch.pow(scale, torch.arange(n // 2, dtype=torch.float32, device=self.device))
        if not self.windowed[layer]:
            theta = theta / self.w['rope_freqs.weight'][:n // 2]
        return torch.cos(theta), torch.sin(theta)

    def masked(self, layer, positions, keys_at=None):
        """[query, key]: a key after the query, or on a windowed layer a key the window's length or more before it. `keys_at` is the
        keys' positions where they are not the queries' own (a cache read by one new position)."""
        apart = positions[..., :, None] - (positions if keys_at is None else keys_at)[..., None, :]
        return (apart < 0) | (apart >= self.a['window']) if self.windowed[layer] else apart < 0

    def per_layer_input(self, x, tokens):
        """[position, layer, per-layer width]: what each layer's third merge is given beside its state (project_per_layer_inputs)."""
        T = x.shape[0]
        own = self.rows('per_layer_token_embd.weight', tokens).view(T, self.layers, self.per) * math.sqrt(self.per)
        shared = (F.linear(x, self.weight('per_layer_model_proj.weight')) * (1 / math.sqrt(self.width))).view(T, self.layers, self.per)
        return (rms(shared, self.w['per_layer_proj_norm.weight']) + own) * (1 / math.sqrt(2))

    def keys(self, layer, x, positions):
        """A key-making layer's keys and values."""
        name = lambda part: f'blk.{layer}.{part}.weight'
        h = rms(x, self.w[name('attn_norm')])
        k = F.linear(h, self.weight(name('attn_k')))
        v = F.linear(h, self.weight(name('attn_v'))) if name('attn_v') in self.w else k
        held = half if self.cache == 'float16' else (lambda x: x)
        return held(rope(rms(k, self.w[name('attn_k_norm')]), *self.rotation(layer, positions))), held(rms(v))

    def attention(self, layer, x, positions, k, v, keys_at=None):
        """What a layer's attention merge adds to its input. k and v are the cache's rows it reads, at `keys_at` (the queries' own
        positions when not given). Any leading axes are texts side by side: x [..., position, width], k and v [..., key, head width]."""
        name = lambda part: f'blk.{layer}.{part}.weight'
        n = self.head(layer)
        cos, sin = self.rotation(layer, positions)
        h = rms(x, self.w[name('attn_norm')])
        q = rope(rms(F.linear(h, self.weight(name('attn_q'))).view(*x.shape[:-1], self.heads, n), self.w[name('attn_q_norm')]), cos[..., None, :], sin[..., None, :])
        scores = torch.einsum('...thd,...sd->...hts', q, k).masked_fill(self.masked(layer, positions, keys_at)[..., None, :, :], float('-inf'))
        mixed = torch.einsum('...hts,...sd->...thd', torch.softmax(scores, -1), v).reshape(*x.shape[:-1], self.heads * n)
        return rms(F.linear(mixed, self.weight(name('attn_output'))), self.w[name('post_attention_norm')])

    def feed_forward(self, layer, attn_out):
        """What a layer's feed-forward merge adds to the attention merge's output."""
        name = lambda part: f'blk.{layer}.{part}.weight'
        f = rms(attn_out, self.w[name('ffn_norm')])
        f = gelu(F.linear(f, self.weight(name('ffn_gate')))) * F.linear(f, self.weight(name('ffn_up')))
        return rms(F.linear(f, self.weight(name('ffn_down'))), self.w[name('post_ffw_norm')])

    def third(self, layer, pe_in, per):
        """What a layer's third merge adds: the per-layer input, gated by the state."""
        name = lambda part: f'blk.{layer}.{part}.weight'
        e = gelu(F.linear(pe_in, self.weight(name('inp_gate')))) * per
        return rms(F.linear(e, self.weight(name('proj'))), self.w[name('post_norm')])

    def block(self, layer, x, per, positions, k, v, keys_at=None):
        """One layer on its input: its three merges' outputs and its exit."""
        attn_out = x + self.attention(layer, x, positions, k, v, keys_at)
        pe_in = attn_out + self.feed_forward(layer, attn_out)
        pe_out = pe_in + self.third(layer, pe_in, per)
        return attn_out, pe_in, pe_out, pe_out * self.w[f'blk.{layer}.layer_output_scale.weight']

    def forward(self, tokens, given=None, again=True):
        """Every merge of one text held fixed: `incoming[m]` and `merged[m]`, each [position, width], for the 3 x layers merges, and the
        last layer's exit. `given(layer)`, if passed, is a layer's input taken from elsewhere (the engine's own exit of the layer before):
        each layer then stands alone. `again` makes each layer again during a backward instead of holding every layer's graph."""
        tokens = [int(t) for t in tokens]
        positions = torch.arange(len(tokens), device=self.device)
        x = self.rows('token_embd.weight', tokens) * math.sqrt(self.width)
        per = self.per_layer_input(x, tokens)
        incoming, merged, cache = [], [], {}
        run = (lambda f, *a: checkpoint(f, *a, use_reentrant=False)) if again and torch.is_grad_enabled() else (lambda f, *a: f(*a))
        for layer in range(self.layers):
            if given is not None and layer:
                x = given(layer)
            if layer < self.keys_from:
                cache[layer] = run(lambda x, layer=layer: self.keys(layer, x, positions), x)
            k, v = cache[self.keys_of(layer)]
            attn_out, pe_in, pe_out, out = run(lambda x, p, k, v, layer=layer: self.block(layer, x, p, positions, k, v), x, per[:, layer], k, v)
            incoming += [x, attn_out, pe_in]
            merged += [attn_out, pe_in, pe_out]
            x = out
        return dict(incoming=incoming, merged=merged, exit=x, result_norm=rms(x, self.w['output_norm.weight']), keys=cache)

    def scores(self, result_norm):
        """The vocabulary's scores of a final state: its product with the token table, which the file ties to the output, soft-capped
        (gemma4.cpp:397-402). Used to pick the next token of a text being written and by the checks, never as a measure of an answer."""
        table = None if self.kept is None else self.kept.get('token table')
        if table is None:
            table = stored(self.file, self.tensors['token_embd.weight']).to(self.device).float()
            if self.kept is not None:
                self.kept['token table'] = table
        cap = float(self.a['final_logit_softcapping'])
        return cap * torch.tanh(F.linear(result_norm, table) / cap)

    @torch.no_grad()
    def write(self, prompts, limit, ends=()):
        """The text this model answers each prompt with, as the recorder writes one: the vocabulary's first choice at each position
        (temperature 0), until a token of `ends` or `limit` tokens; the ending token is the text's last. The prompts are written side
        by side, one new position of each at a time, reading a cache of the key-making layers' keys and values. Returns each
        prompt's tokens. Acquisition only: nothing here reads what a text says."""
        ends = {int(t) for t in ends}
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        with self.holding():
            room = max(len(p) for p in prompts) + limit
            K = {layer: torch.zeros(len(prompts), room, self.head(layer), device=self.device) for layer in range(self.keys_from)}
            V = {layer: torch.zeros_like(k) for layer, k in K.items()}
            last = []
            for b, prompt in enumerate(prompts):                 # each prompt alone, all its positions at once
                said = self.forward(prompt)
                for layer, (k, v) in said['keys'].items():
                    K[layer][b, :len(prompt)], V[layer][b, :len(prompt)] = k, v
                last.append(said['result_norm'][-1])
            texts, live = [[] for _ in prompts], list(range(len(prompts)))
            at = torch.tensor([len(p) for p in prompts], device=self.device)      # the position each text's next token is given at
            chosen = self.scores(torch.stack(last)).argmax(-1)
            for step in range(limit):
                tokens = chosen.tolist()
                for b, token in zip(live, tokens):
                    texts[b].append(token)
                going = [i for i, token in enumerate(tokens) if token not in ends]
                if not going or step + 1 == limit:
                    break
                if len(going) < len(live):                       # the texts that ended leave the cache
                    index = torch.tensor(going, device=self.device)
                    K, V = {layer: k[index] for layer, k in K.items()}, {layer: v[index] for layer, v in V.items()}
                    at, live, tokens = at[index], [live[i] for i in going], [tokens[i] for i in going]
                x = self.rows('token_embd.weight', tokens) * math.sqrt(self.width)
                per = self.per_layer_input(x, tokens)
                row, upto = torch.arange(len(live), device=self.device), int(at.max()) + 1
                there = torch.arange(upto, device=self.device)
                for layer in range(self.layers):
                    if layer < self.keys_from:
                        K[layer][row, at], V[layer][row, at] = self.keys(layer, x, at)
                    source = self.keys_of(layer)
                    x = self.block(layer, x[:, None], per[:, None, layer], at[:, None], K[source][:, :upto], V[source][:, :upto], there)[3][:, 0]
                chosen = self.scores(rms(x, self.w['output_norm.weight'])).argmax(-1)
                at = at + 1
            return texts


def unit(x):
    return x / x.norm(dim=-1, keepdim=True)


def turns(pairs, precision=torch.float64):
    """Every merge's turn, unit(merged) - unit(incoming), from the forward's float32 pairs in the precision asked."""
    return [unit(m.to(precision)) - unit(i.to(precision)) for i, m in zip(pairs['incoming'], pairs['merged'])]


def registered(pairs, question, original):
    """A text's steps laid on the original's recording of the same question, by the capture's rule of a registration
    (capture_bound.laid_on_original: each step's turns at all merges as one unit vector, the product of the two answers' steps, and
    complete_overlap.distance_path over it), from the forward's pairs of that text: [step, step of the original]. `question` is the
    position of the text's first step; `original` the original's exported turns of the question, [step, merge, width]. The turns are
    made as the export makes them (unit(merged) - unit(incoming) in float64, kept in float32) and the product is the card's."""
    from pipeline.overlap.complete_overlap import distance_path
    flat = lambda x: x.reshape(len(x), -1) / x.reshape(len(x), -1).norm(dim=-1, keepdim=True)
    mine = torch.stack([t[question:].float() for t in turns(pairs)], 1)
    theirs = torch.as_tensor(np.asarray(original), dtype=torch.float32, device=mine.device)
    if mine.shape[1:] != theirs.shape[1:]:
        raise ValueError(f'the text has turns of {tuple(mine.shape[1:])} a step and the original of {tuple(theirs.shape[1:])}')
    return distance_path((flat(mine) @ flat(theirs).T).cpu().numpy())


def architecture_of(model):
    """The architecture keys of a model file, read with the engine's own reader."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'dependencies/llama.cpp/gguf-py'))
    from gguf import GGUFReader
    fields = GGUFReader(model).fields
    key = lambda name: fields[f'gemma4.{name}'].contents()
    return dict(format='torch-forward-architecture/1', read_with='gguf-py of dependencies/llama.cpp (GGUFReader)', architecture=fields['general.architecture'].contents(),
                layers=key('block_count'), width=key('embedding_length'), heads=key('attention.head_count'), key_heads=key('attention.head_count_kv'),
                head_width=key('attention.key_length'), head_width_windowed=key('attention.key_length_swa'), window=key('attention.sliding_window'),
                windowed=[bool(w) for w in key('attention.sliding_window_pattern')], layers_sharing_keys=key('attention.shared_kv_layers'),
                rope_base=key('rope.freq_base'), rope_base_windowed=key('rope.freq_base_swa'), rope_pairs=key('rope.dimension_count'), rope_pairs_windowed=key('rope.dimension_count_swa'),
                norm_eps=key('attention.layer_norm_rms_epsilon'), per_layer_width=key('embedding_length_per_layer_input'), feed_forward=list(key('feed_forward_length')),
                final_logit_softcapping=key('final_logit_softcapping'))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def spread(values):
    v = torch.cat([x.flatten() for x in values]).double()
    return dict(largest=float(v.max()), median=float(v.median()))


KINDS = ('attention', 'feedforward', 'layer')


def check(model, reading, capture, answers, record):
    """Fill `record` with the forward's distances from a kept reading, answer by answer. Returns every answer's tokens."""
    reading, capture = Path(reading), Path(capture)
    rows = json.loads((reading/'row-map.json').read_text())
    state = [np.load(reading/'nodes'/str(m)/'state.npy', mmap_mode='r') for m in range(3 * model.layers)]
    scale = [float(model.w[f'blk.{l}.layer_output_scale.weight']) for l in range(model.layers)]
    over = lambda ours, theirs: (ours.double() - theirs.double()).norm(dim=-1) / theirs.double().norm(dim=-1)
    merges = range(len(state))
    whole, turn, cosine, alone, batched, single = ([[] for _ in merges] for _ in range(6))
    cases, seconds, texts, far, far_engine, far_other = [], [], [], [], [], []
    for a in rows['answers'][:answers]:
        case = next(iter(capture.glob(f"batch-*/{a['id']}/case.json")), None)
        if case is None:
            raise ValueError(f"{a['id']}: no case.json under {capture}")
        prompt, text = json.loads(case.read_text())['tokens'], json.loads((reading/'answers'/a['id']/'answer.json').read_text())['tokens']
        if len(prompt) != a['question'] + 1 or len(text) != a['steps']:
            raise ValueError(f"{a['id']}: a prompt of {len(prompt)} tokens and a text of {len(text)} against question {a['question']} and {a['steps']} steps")
        tokens = prompt + text[:-1]                              # position question + s is given the text's token s - 1; the last token is given to nothing
        texts.append(tokens)
        at = np.concatenate((a['question_rows_from'] + np.arange(a['question']), a['answer_rows_from'] + np.arange(a['steps'])))     # by position
        theirs = [torch.from_numpy(np.asarray(state[m][at])).to(model.device) for m in merges]
        in_a_batch = torch.arange(len(tokens), device=model.device) <= a['question']      # the prompt's positions: the engine made them in batches
        with torch.no_grad():
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            began = time.perf_counter()
            ours = model.forward(tokens)
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            seconds.append(time.perf_counter() - began)
            fed = model.forward(tokens, given=lambda layer: theirs[3 * layer - 1] * scale[layer - 1])
            asked, model.cache = model.cache, 'float16' if model.cache is None else None
            other = model.forward(tokens)                        # the same forward with the cache's rounding the other way: a change of about 1e-4 of k and v
            model.cache = asked
            per = model.per_layer_input(ours['incoming'][0], tokens)
            ours_turns = turns(ours)
            for layer in range(model.layers):
                first, second, last = 3 * layer, 3 * layer + 1, 3 * layer + 2
                entered = fed['incoming'][first]                 # the engine's own exit of the layer before (the embedding at the first layer)
                parts = ((first, fed['merged'][first] - entered, theirs[first].double() - entered.double()),
                         (second, model.feed_forward(layer, theirs[first]), theirs[second].double() - theirs[first].double()),
                         (last, model.third(layer, theirs[second], per[:, layer]), theirs[last].double() - theirs[second].double()))
                for m, added, added_there in parts:              # each merge alone: what it adds to the ENGINE's own input, over what the engine added
                    apart = over(added, added_there)
                    alone[m].append(apart)
                    batched[m].append(apart[in_a_batch])
                    single[m].append(apart[~in_a_batch])
            from_engine = torch.stack([over(ours['merged'][m], theirs[m]) for m in merges])               # [merge, position]
            from_other = torch.stack([over(other['merged'][m], ours['merged'][m]) for m in merges]).max(0).values
            worst, where = from_engine.max(0)
            far_engine.append(worst)
            far_other.append(from_other)
            for p in (worst > 0.1).nonzero().flatten().tolist():
                passed = (from_engine[:, p] > 0.03).nonzero()
                far.append(dict(answer=a['id'], position=p, row='prompt' if p <= a['question'] else 'answer', worst=round(float(worst[p]), 4), at_merge=int(where[p]),
                                first_merge_over_0_03=int(passed[0]), at_the_last_merge=round(float(from_engine[-1, p]), 4),
                                the_other_cache_mode_against_this_one=round(float(from_other[p]), 4)))
            for m in merges:
                whole[m].append(from_engine[m])
                if m:                                            # a layer's exit is its third merge times a positive number: the same direction
                    there = unit(theirs[m].double()) - unit(theirs[m - 1].double())
                    turn[m].append((ours_turns[m] - there).norm(dim=-1) / there.norm(dim=-1))
                    cosine[m].append((ours_turns[m] * there).sum(-1) / (ours_turns[m].norm(dim=-1) * there.norm(dim=-1)))
        cases.append(dict(id=a['id'], positions=len(tokens), forward_seconds=round(seconds[-1], 3)))
        del theirs, ours, fed, other, ours_turns
    least = lambda each: float(torch.cat(each).min())
    record['answers'], record['positions'] = cases, int(sum(c['positions'] for c in cases))
    record['by_merge'] = [dict(merge=m, kind=KINDS[m % 3], layer=m // 3, alone=spread(alone[m]), alone_where_the_engine_ran_a_batch=spread(batched[m]),
                               alone_where_the_engine_ran_one_position=spread(single[m]), state_end_to_end=spread(whole[m]),
                               turn_end_to_end=spread(turn[m]) if m else None, turn_cosine_least=least(cosine[m]) if m else None) for m in merges]
    by_kind = lambda each, first=0: {kind: spread([x for m in range(first, len(each)) if m % 3 == k for x in each[m]]) for k, kind in enumerate(KINDS)}
    record.update(alone=by_kind(alone), alone_where_the_engine_ran_a_batch=by_kind(batched), alone_where_the_engine_ran_one_position=by_kind(single),
                  state_end_to_end=by_kind(whole), turn_end_to_end=by_kind(turn, 1),
                  turn_cosine=dict(least=min(least(c) for c in cosine[1:]), median=float(torch.cat([x for c in cosine[1:] for x in c]).median())))
    windowed = lambda m: model.windowed[m // 3]
    record['attention_alone_by_kind_of_layer'] = {
        name: dict(where_the_engine_ran_a_batch=spread([x for m in merges if m % 3 == 0 and windowed(m) == flag for x in batched[m]]),
                   where_the_engine_ran_one_position=spread([x for m in merges if m % 3 == 0 and windowed(m) == flag for x in single[m]]))
        for name, flag in (('windowed', True), ('whole-text', False))}
    e, o = torch.cat(far_engine), torch.cat(far_other)
    record['positions_by_their_worst_merge'] = {
        f'over_{level}': dict(from_the_engine=int((e > level).sum()), the_other_cache_mode_from_this_one=int((o > level).sum()), both=int(((e > level) & (o > level)).sum()))
        for level in (0.01, 0.03, 0.1, 0.5)}
    record['at_the_last_merge'] = dict(median=float(torch.cat(whole[-1]).median()), largest=float(torch.cat(whole[-1]).max()))
    record['positions_over_0_1_from_the_engine'] = far
    record['forward_seconds'] = dict(first=round(seconds[0], 3), median=round(float(np.median(seconds)), 3), all=round(float(sum(seconds)), 3))
    return texts


def backward_once(model, texts):
    """A backward of a stand-in number through every layer, text by text, the derivatives added up as a step over several answers
    would: the time, the card's memory, and that every leaf has a derivative."""
    if model.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize()
    each = []
    for tokens in texts:
        began = time.perf_counter()
        pairs = model.forward(tokens)
        number = sum((t ** 2).sum() for t in turns(pairs, torch.float32))
        number.backward()
        if model.device.type == 'cuda':
            torch.cuda.synchronize()
        each.append(time.perf_counter() - began)
        del pairs, number
    sizes = {name: float(x.grad.norm()) for name, x in model.leaves.items()}
    said = dict(what='the sum of every merge\'s squared turn, text by text, the derivatives added up: a stand-in, no loss of anything', texts=len(texts),
                positions=int(sum(len(tokens) for tokens in texts)), forward_and_backward_seconds=round(float(sum(each)), 3),
                a_text_seconds=dict(least=round(min(each), 3), median=round(float(np.median(each)), 3), greatest=round(max(each), 3)),
                leaves=len(sizes), leaves_with_a_derivative=sum(v > 0 and math.isfinite(v) for v in sizes.values()),
                card_bytes_at_most=int(torch.cuda.max_memory_allocated(model.device)) if model.device.type == 'cuda' else None)
    for x in model.leaves.values():
        x.grad = None
    return said


def written(model, reading, capture, answers, record):
    """Fill `record` with this forward's writing against the recorder's: the decisions under the recorded text, then the texts written."""
    reading, capture = Path(reading), Path(capture)
    rows = json.loads((reading/'row-map.json').read_text())['answers'][:answers]
    under = lambda values: {str(t): int((np.asarray(values) < t).sum()) for t in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)}
    prompts, texts, limits, apart, theirs_near, ours_near, other, compared = [], [], set(), [], [], [], [], 0
    began = time.perf_counter()
    with model.holding(), torch.no_grad():
        for a in rows:
            folder = next(iter(capture.glob(f"batch-*/{a['id']}")), None)
            if folder is None:
                raise ValueError(f"{a['id']}: no case under {capture}")
            case, text = json.loads((folder/'case.json').read_text()), json.loads((reading/'answers'/a['id']/'answer.json').read_text())['tokens']
            if len(case['tokens']) != a['question'] + 1 or len(text) != a['steps']:
                raise ValueError(f"{a['id']}: a prompt of {len(case['tokens'])} tokens and a text of {len(text)} against question {a['question']} and {a['steps']} steps")
            prompts.append(case['tokens'])
            texts.append(text)
            limits.add(case['max_tokens'])
            said = json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
            if said is None or said['tokens'] != text:           # the recorder's kept scores are of another text than the reading's: nothing to hold the scores to
                continue
            scores = model.scores(model.forward(case['tokens'] + text[:-1])['result_norm'][a['question']:])
            for s, step in enumerate(said['steps']):
                if step['input_position'] != a['question'] + s or step['selection'] != 'greedy':
                    raise ValueError(f"{a['id']}: step {s} at position {step['input_position']}, selected by {step['selection']}")
                kept = torch.tensor([t['logit'] for t in step['top']], dtype=torch.float64, device=model.device)
                first = torch.topk(scores[s], 2)
                apart.append(float((scores[s, [t['id'] for t in step['top']]].double() - kept).abs().max()))
                theirs_near.append(step['top'][0]['logit'] - step['top'][1]['logit'])
                ours_near.append(float(first.values[0] - first.values[1]))
                if int(first.indices[0]) != step['greedy_token']:
                    other.append(dict(answer=a['id'], step=s, of=len(text), between_the_engine_s_first_two=round(theirs_near[-1], 6), between_ours=round(ours_near[-1], 6), scores_apart=round(apart[-1], 6)))
            compared += 1
            del scores
    if model.device.type == 'cuda':
        torch.cuda.synchronize()
    record['under_the_recorded_text'] = dict(
        what="this forward's scores at every decision of the recorded text held fixed, beside the eight scores the recorder kept at that decision",
        answers=compared, decisions=len(apart), seconds=round(time.perf_counter() - began, 1),
        scores_apart=dict(median=float(np.median(apart)), ninety_ninth=float(np.quantile(apart, 0.99)), largest=float(np.max(apart))) if apart else None,
        decisions_with_the_engine_s_first_two_nearer_than=under(theirs_near), decisions_with_ours_nearer_than=under(ours_near),
        decisions_where_the_first_choice_is_another_token=len(other), answers_with_one=len({o['answer'] for o in other}), those=other)
    if len(limits) != 1:
        raise ValueError(f'the cases have several limits: {sorted(limits)}')
    limit = limits.pop()
    ends = sorted({text[-1] for text in texts if len(text) < limit})
    if model.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(model.device)
    began = time.perf_counter()
    ours = model.write(prompts, limit, ends)
    if model.device.type == 'cuda':
        torch.cuda.synchronize()
    seconds, leaves = time.perf_counter() - began, []
    for a, mine, text in zip(rows, ours, texts):
        same = next((s for s, (x, y) in enumerate(zip(mine, text)) if x != y), min(len(mine), len(text)))
        if same < max(len(mine), len(text)):
            leaves.append(dict(answer=a['id'], leaves_at_step=same, recorded_tokens=len(text), written_tokens=len(mine)))
    peak = int(torch.cuda.max_memory_allocated(model.device)) if model.device.type == 'cuda' else None
    native = capture.parent/'exported'                        # the original's exported turns, which a capture's registration reads
    if all((native/a['id']/'turns.npy').is_file() for a in rows):
        from pipeline.overlap.complete_overlap import distance_path, unit_steps
        began, one_on_one, as_the_capture = time.perf_counter(), 0, []
        for a, prompt, mine in zip(rows, prompts, ours):
            theirs = np.load(native/a['id']/'turns.npy', mmap_mode='r')
            with torch.no_grad():
                said = model.forward(prompt + mine[:-1])
                path = registered(said, a['question'], theirs)
            one_on_one += len(mine) == len(theirs) and bool(np.array_equal(path, np.column_stack([np.arange(len(mine))] * 2)))
            if any(leaf['answer'] == a['id'] for leaf in leaves):    # a text that left the recorded one: the capture's own arithmetic on the same turns, on the processor
                kept = torch.stack([t[a['question']:].float() for t in turns(said)], 1).cpu().numpy()
                as_the_capture.append(bool(np.array_equal(path, distance_path(unit_steps(kept) @ unit_steps(theirs).T))))
            del said
        if model.device.type == 'cuda':
            torch.cuda.synchronize()
        record['laid_on_the_original'] = dict(
            what="each written text laid on the original's exported turns of its question by `registered` (the forward of the text, its turns, the path)",
            original_turns=f'{capture.parent.name}/exported', answers=len(rows), seconds=round(time.perf_counter() - began, 2),
            one_on_one=int(one_on_one), of_the_texts_written_as_recorded=len(rows) - len(leaves),
            texts_that_left_the_recorded=len(leaves), their_path_is_the_capture_s_arithmetic_on_the_processor=int(sum(as_the_capture)))
    record['written'] = dict(
        what="the answers written by `write` from the cohort's prompts, side by side, beside the recorded texts", answers=len(rows), limit=limit, ending_tokens=ends,
        seconds=round(seconds, 2), tokens_written=int(sum(len(t) for t in ours)), recorded_tokens=int(sum(len(t) for t in texts)),
        a_token_of_the_longest_text_seconds=round(seconds / max(len(t) for t in ours), 4),
        answers_written_as_recorded=len(rows) - len(leaves), answers_whose_text_leaves_the_recorded=leaves, card_bytes_at_most=peak)
    return ours


if __name__ == '__main__':
    from pipeline.overlap import fed_replay as base
    from pipeline.paths import ORIGINAL_MODEL, ROUND_TWO_RUNS
    here = Path(__file__).resolve().parent/'torch-forward'
    if len(sys.argv) == 3 and sys.argv[1] == 'describe':
        said = dict(architecture_of(ORIGINAL_MODEL), model_sha256=digest(ORIGINAL_MODEL))
        Path(sys.argv[2]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[2]).write_text(json.dumps(said, indent=1) + '\n')
        sys.exit(print(json.dumps(said)))
    if len(sys.argv) < 5 or sys.argv[1] not in ('check', 'write'):
        sys.exit(__doc__)
    options = dict(o.split('=', 1) for o in sys.argv[5:])
    architecture = json.loads((here/'architecture.json').read_text())
    reading = ROUND_TWO_RUNS/sys.argv[2]/'reading'
    manifest = json.loads((reading/'reading.json').read_text())
    file = ROUND_TWO_RUNS.parent.parent.parent/options['model'] if 'model' in options else ORIGINAL_MODEL       # a candidate: the original's layout, other values
    if manifest['model_sha256'] != (digest(file) if 'model' in options else architecture['model_sha256']):
        raise ValueError('the reading is of another model than the file given')
    began = time.perf_counter()
    model = Model(file, json.loads(base.TENSORS.read_text()), architecture, options.get('device', 'cuda'), cache=options.get('cache'))
    if sys.argv[1] == 'write':
        record = dict(what="the torch forward's writing of a model's answers against the recorder's kept text of that model", model_sha256=manifest['model_sha256'], model=options.get('model', 'the original'),
                      reading=f'{sys.argv[2]}/reading', reading_sha256=hashlib.sha256((reading/'reading.json').read_bytes()).hexdigest(), prompts_from=f'{sys.argv[3]}/capture',
                      device=torch.cuda.get_device_name(model.device) if model.device.type == 'cuda' else 'the processor', torch=torch.__version__, load_seconds=round(time.perf_counter() - began, 1))
        written(model, reading, ROUND_TWO_RUNS/sys.argv[3]/'capture', int(options.get('answers', len(manifest['answers_read']))), record)
        Path(sys.argv[4]).parent.mkdir(parents=True, exist_ok=True)
        Path(sys.argv[4]).write_text(json.dumps(record, indent=1) + '\n')
        show = dict(record['under_the_recorded_text'], those=len(record['under_the_recorded_text']['those']))
        sys.exit(print(json.dumps(dict(under_the_recorded_text=show, written=record['written'], laid_on_the_original=record.get('laid_on_the_original')), indent=1)))
    record = dict(what="the torch forward of a model file against the engine's kept reading of that model: distances over the engine's own size", model_sha256=manifest['model_sha256'],
                  model=options.get('model', 'the original'),
                  cache=model.cache or 'as computed (float32)', reading=f'{sys.argv[2]}/reading', reading_sha256=hashlib.sha256((reading/'reading.json').read_bytes()).hexdigest(), text_from=f'{sys.argv[3]}/capture',
                  device=torch.cuda.get_device_name(model.device) if model.device.type == 'cuda' else 'the processor', torch=torch.__version__,
                  leaves=len(model.leaves), leaf_weights=int(sum(x.numel() for x in model.leaves.values())), load_seconds=round(time.perf_counter() - began, 1))
    texts = check(model, reading, ROUND_TWO_RUNS/sys.argv[3]/'capture', int(options.get('answers', len(manifest['answers_read']))), record)
    record['card_bytes_after_the_forwards'] = int(torch.cuda.max_memory_allocated(model.device)) if model.device.type == 'cuda' else None
    record['backward'] = backward_once(model, texts)
    Path(sys.argv[4]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[4]).write_text(json.dumps(record, indent=1) + '\n')
    print(json.dumps({k: record[k] for k in ('cache', 'alone', 'attention_alone_by_kind_of_layer', 'state_end_to_end', 'turn_end_to_end', 'turn_cosine', 'positions_by_their_worst_merge', 'at_the_last_merge', 'forward_seconds', 'backward')}, indent=1))
