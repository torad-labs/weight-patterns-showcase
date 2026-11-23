"""torch_forward.py on a made-up model file: a temporary folder, the processor, no real model, no card, no recorder, no real answer."""
import sys, tempfile, unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_forward as tf

WIDE, PER, VOCABULARY, FEED = 16, 4, 20, 24
ARCHITECTURE = dict(layers=4, width=WIDE, heads=2, key_heads=1, head_width=8, head_width_windowed=4, window=3, windowed=[True, False, True, False],
                    layers_sharing_keys=2, rope_base=1000000.0, rope_base_windowed=10000.0, norm_eps=1e-6, per_layer_width=PER, final_logit_softcapping=30.0)


def made(root, seed=0):
    """A model file of the engine's tensor names at this small size, and its index: matrices BF16 (in, out), norms and scales F32."""
    rng, parts, tensors, at = np.random.default_rng(seed), [], {}, 0

    def put(name, shape, kind, values=None):
        nonlocal at
        x = rng.normal(size=shape[::-1]).astype(np.float32) if values is None else np.asarray(values, dtype=np.float32).reshape(shape[::-1])
        raw = (x.view(np.uint32) >> 16).astype('<u2').tobytes() if kind == 'bf16' else x.astype('<f4').tobytes()
        tensors[name] = dict(name=name, native_type=kind, shape=list(shape) + [1] * (4 - len(shape)), offset_bytes=at, bytes=len(raw))
        parts.append(raw)
        at += len(raw)

    norm = lambda n: rng.uniform(0.5, 1.5, size=n)
    put('rope_freqs.weight', [4], 'f32', rng.uniform(0.5, 2.0, size=4))
    put('token_embd.weight', [WIDE, VOCABULARY], 'bf16')
    put('per_layer_token_embd.weight', [PER * 4, VOCABULARY], 'bf16')
    put('output_norm.weight', [WIDE], 'f32', norm(WIDE))
    put('per_layer_model_proj.weight', [WIDE, PER * 4], 'bf16')
    put('per_layer_proj_norm.weight', [PER], 'f32', norm(PER))
    for layer, windowed in enumerate(ARCHITECTURE['windowed']):
        head, name = 4 if windowed else 8, lambda part: f'blk.{layer}.{part}.weight'
        for part in ('attn_norm', 'post_attention_norm', 'post_ffw_norm', 'post_norm', 'ffn_norm'):
            put(name(part), [WIDE], 'f32', norm(WIDE))
        put(name('layer_output_scale'), [1], 'f32', rng.uniform(0.5, 1.5, size=1))
        for part, shape in (('ffn_down', [FEED, WIDE]), ('ffn_gate', [WIDE, FEED]), ('ffn_up', [WIDE, FEED]), ('inp_gate', [WIDE, PER]), ('proj', [PER, WIDE]),
                            ('attn_k', [WIDE, head]), ('attn_v', [WIDE, head]), ('attn_q', [WIDE, 2 * head]), ('attn_output', [2 * head, WIDE])):
            put(name(part), shape, 'bf16')
        for part in ('attn_k_norm', 'attn_q_norm'):
            put(name(part), [head], 'f32', norm(head))
    model = Path(root)/'model.gguf'
    model.write_bytes(b''.join(parts))
    return model, tensors


TEXT = [3, 7, 1, 19, 4, 4, 11, 0, 8]
number = lambda pairs: sum((t ** 2).sum() for t in tf.turns(pairs))


class TheForwardIsTheEnginesGraph(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.file, self.tensors = made(self.folder.name)
        self.model = tf.Model(self.file, self.tensors, ARCHITECTURE, 'cpu')

    def tearDown(self):
        self.folder.cleanup()

    def test_each_merge_is_given_the_merge_before_it_and_a_layer_exits_scaled(self):
        with torch.no_grad():
            said = self.model.forward(TEXT)
        self.assertEqual((len(said['incoming']), len(said['merged']), tuple(said['merged'][0].shape)), (12, 12, (len(TEXT), WIDE)))
        for layer in range(4):
            self.assertIs(said['incoming'][3 * layer + 1], said['merged'][3 * layer])
            self.assertIs(said['incoming'][3 * layer + 2], said['merged'][3 * layer + 1])
            if layer:
                scale = self.model.w[f'blk.{layer - 1}.layer_output_scale.weight']
                torch.testing.assert_close(said['incoming'][3 * layer], said['merged'][3 * layer - 1] * scale)
        torch.testing.assert_close(said['incoming'][0], self.model.rows('token_embd.weight', TEXT) * 4.0)
        self.assertTrue(all(bool(torch.isfinite(m).all()) for m in said['merged']))

    def test_a_text_s_beginning_computes_alone_what_it_computes_inside_the_text(self):
        with torch.no_grad():
            whole, start = self.model.forward(TEXT), self.model.forward(TEXT[:5])
        for a, b in zip(whole['merged'], start['merged']):
            torch.testing.assert_close(a[:5], b, rtol=1e-4, atol=1e-5)

    def test_a_windowed_layer_does_not_read_past_its_window_and_no_layer_reads_ahead(self):
        positions = torch.arange(6)
        windowed, whole = self.model.masked(0, positions), self.model.masked(1, positions)
        self.assertEqual((bool(windowed[5, 1]), bool(windowed[5, 3]), bool(windowed[2, 4])), (True, False, True))     # 4 back is past a window of 3
        self.assertEqual((bool(whole[5, 1]), bool(whole[2, 4]), bool(whole[3, 3])), (False, True, False))
        with torch.no_grad():
            changed = self.model.forward([5] + TEXT[1:])
            same = self.model.forward(TEXT)
        self.assertFalse(torch.allclose(changed['merged'][0][0], same['merged'][0][0]))
        self.assertGreater(float((changed['merged'][3][1] - same['merged'][3][1]).abs().max()), 0)                  # a whole-text layer reads the first position

    def test_a_late_layer_reads_the_keys_of_the_last_layer_of_its_kind_that_makes_them(self):
        self.assertEqual([self.model.keys_of(layer) for layer in range(4)], [0, 1, 0, 1])
        self.assertNotIn('blk.2.attn_k.weight', self.model.w)
        self.assertNotIn('blk.3.attn_v.weight', self.model.w)
        self.assertIn('blk.1.attn_k.weight', self.model.w)

    def test_the_leaves_are_the_matrices_an_edit_writes_and_any_tensor_on_request(self):
        self.assertEqual(sorted(self.model.leaves), sorted(f'blk.{layer}.{part}.weight' for layer in range(4) for part in ('attn_output', 'ffn_down')))
        number(self.model.forward(TEXT)).backward()
        self.assertTrue(all(float(x.grad.abs().sum()) > 0 for x in self.model.leaves.values()))
        other = tf.Model(self.file, self.tensors, ARCHITECTURE, 'cpu', leaves=lambda name: name in ('blk.1.attn_q.weight', 'blk.0.ffn_norm.weight'))
        number(other.forward(TEXT)).backward()
        self.assertEqual(sorted(other.leaves), ['blk.0.ffn_norm.weight', 'blk.1.attn_q.weight'])
        self.assertTrue(all(float(x.grad.abs().sum()) > 0 for x in other.leaves.values()))

    def test_the_derivative_is_the_same_with_each_layer_made_again_as_with_every_layer_held(self):
        number(self.model.forward(TEXT, again=True)).backward()
        again = {name: x.grad.clone() for name, x in self.model.leaves.items()}
        for x in self.model.leaves.values():
            x.grad = None
        number(self.model.forward(TEXT, again=False)).backward()
        for name, x in self.model.leaves.items():
            torch.testing.assert_close(again[name], x.grad, rtol=1e-4, atol=1e-6)

    def test_the_derivative_says_what_a_small_change_of_a_matrix_does(self):
        leaf = self.model.leaves['blk.1.ffn_down.weight']
        before = number(self.model.forward(TEXT, again=False))
        before.backward()
        step = 1e-3 * leaf.grad / leaf.grad.norm()
        with torch.no_grad():
            leaf += step
            after = number(self.model.forward(TEXT))
        self.assertAlmostEqual(float(after - before) / float((leaf.grad * step).sum()), 1.0, delta=0.1)

    def test_a_layer_given_its_input_from_elsewhere_stands_alone(self):
        with torch.no_grad():
            own = self.model.forward(TEXT)
            exits = {layer: own['merged'][3 * layer - 1] * self.model.w[f'blk.{layer - 1}.layer_output_scale.weight'] for layer in range(1, 4)}
            fed = self.model.forward(TEXT, given=lambda layer: exits[layer])
            other = self.model.forward(TEXT, given=lambda layer: exits[layer] * 2 if layer == 2 else exits[layer])
        for a, b in zip(own['merged'], fed['merged']):
            torch.testing.assert_close(a, b)
        torch.testing.assert_close(other['merged'][5], own['merged'][5])
        self.assertFalse(torch.allclose(other['merged'][8], own['merged'][8]))


class ATextWrittenOneTokenAtATime(unittest.TestCase):
    PROMPTS = ([3, 7, 1, 19], [5, 2, 2, 9, 14, 0, 11], [8])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.file, self.tensors = made(self.folder.name)
        self.model = tf.Model(self.file, self.tensors, ARCHITECTURE, 'cpu')

    def tearDown(self):
        self.folder.cleanup()

    def first_choices(self, prompt, text):
        """The forward's first choice at every position of the text held fixed, from the prompt's last position on."""
        with torch.no_grad():
            said = self.model.forward(list(prompt) + list(text[:-1]))
            return self.model.scores(said['result_norm'][len(prompt) - 1:]).argmax(-1).tolist()

    def test_the_scores_are_the_token_table_s_product_soft_capped(self):
        state = torch.randn(3, WIDE)
        table = tf.stored(self.file, self.tensors['token_embd.weight']).float()
        torch.testing.assert_close(self.model.scores(state), 30.0 * torch.tanh(state @ table.T / 30.0))
        self.assertEqual(tuple(self.model.scores(state).shape), (3, VOCABULARY))

    def test_each_token_written_is_the_forward_s_first_choice_on_the_text_so_far(self):
        texts = self.model.write(self.PROMPTS, 9)
        self.assertEqual([len(t) for t in texts], [9, 9, 9])            # past the window of 3 on the windowed layers, and several positions from the cache
        for prompt, text in zip(self.PROMPTS, texts):
            self.assertEqual(text, self.first_choices(prompt, text))

    def test_texts_written_side_by_side_are_the_texts_written_alone(self):
        together = self.model.write(self.PROMPTS, 7)
        for prompt, text in zip(self.PROMPTS, together):
            self.assertEqual(self.model.write([prompt], 7), [text])

    def test_a_text_ends_at_an_ending_token_which_is_its_last_and_the_others_go_on(self):
        free = self.model.write(self.PROMPTS, 8)
        end = free[1][2]                                                 # the second text's third token, made an ending token
        texts = self.model.write(self.PROMPTS, 8, ends=[end])
        self.assertEqual(texts[1], free[1][:free[1].index(end) + 1])
        for text, alone in zip(texts, free):
            cut = alone.index(end) + 1 if end in alone else len(alone)
            self.assertEqual(text, alone[:cut])
        self.assertEqual(self.model.write(self.PROMPTS, 1), [t[:1] for t in free])

    def test_a_matrix_is_float32_while_writing_and_its_stored_bits_again_after_it(self):
        before = {name: x.clone() for name, x in self.model.w.items()}
        with self.model.holding():
            self.assertEqual({x.dtype for x in self.model.w.values()}, {torch.float32})
            self.assertIs(self.model.weight('blk.0.attn_q.weight'), self.model.w['blk.0.attn_q.weight'])
            self.model.scores(torch.randn(1, WIDE))
            self.assertIn('token table', self.model.kept)
        self.model.write(self.PROMPTS[:1], 2)
        self.assertIsNone(self.model.kept)
        for name, x in self.model.w.items():
            self.assertEqual(x.dtype, before[name].dtype)
            self.assertTrue(torch.equal(x.view(torch.int16) if x.dtype == torch.bfloat16 else x, before[name].view(torch.int16) if x.dtype == torch.bfloat16 else before[name]), name)
        self.assertTrue(all(self.model.w[name] is x for name, x in self.model.leaves.items()))

    def test_a_written_text_is_laid_on_an_original_by_its_turns(self):
        prompt = list(self.PROMPTS[1])
        text = self.model.write([prompt], 6)[0]
        with torch.no_grad():
            said = self.model.forward(prompt + text[:-1])
            other = self.model.forward(prompt + text[:-1] + [4, 4])          # the same text and two steps more
        own = torch.stack([t[len(prompt) - 1:].float() for t in tf.turns(said)], 1).numpy()      # [step, merge, width]: what an export keeps
        longer = torch.stack([t[len(prompt) - 1:].float() for t in tf.turns(other)], 1).numpy()
        np.testing.assert_array_equal(tf.registered(said, len(prompt) - 1, own), np.column_stack([np.arange(6)] * 2))
        path = tf.registered(said, len(prompt) - 1, longer)
        np.testing.assert_array_equal(path[:6], np.column_stack([np.arange(6)] * 2))               # its six steps on the six they are, then the two more
        self.assertEqual((int(path[-1, 0]), int(path[-1, 1])), (5, 7))
        with self.assertRaisesRegex(ValueError, 'turns of'):
            tf.registered(said, len(prompt) - 1, own[:, :3])

    def test_attention_of_texts_side_by_side_is_each_text_s_attention(self):
        with torch.no_grad():
            said = self.model.forward(TEXT)
            x, positions = said['incoming'][3], torch.arange(len(TEXT))
            k, v = said['keys'][1]
            alone = self.model.attention(1, x, positions, k, v)
            both = self.model.attention(1, torch.stack((x, x * 2)), positions.expand(2, -1), torch.stack((k, k)), torch.stack((v, v)), keys_at=positions)
            one = self.model.attention(1, x[None, -1:], positions[None, -1:], k[None], v[None], keys_at=positions)
        torch.testing.assert_close(both[0], alone, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(one[0, 0], alone[-1], rtol=1e-5, atol=1e-6)


class TheFilesValues(unittest.TestCase):
    def test_a_matrix_is_read_as_out_by_in_with_the_file_s_own_bits(self):
        with tempfile.TemporaryDirectory() as tmp:
            file, tensors = made(tmp)
            x = tf.stored(file, tensors['blk.0.attn_q.weight'])
            self.assertEqual((tuple(x.shape), x.dtype), ((8, WIDE), torch.bfloat16))
            raw = np.frombuffer(file.read_bytes(), dtype='<u2', count=8 * WIDE, offset=tensors['blk.0.attn_q.weight']['offset_bytes'])
            np.testing.assert_array_equal(x.float().numpy().ravel(), (raw.astype(np.uint32) << 16).view(np.float32))
            self.assertEqual(tuple(tf.stored(file, tensors['blk.0.layer_output_scale.weight']).shape), (1,))

    def test_the_cache_s_rounding_is_asked_for_and_changes_the_keys_only_by_float16(self):
        with tempfile.TemporaryDirectory() as tmp:
            file, tensors = made(tmp)
            plain, rounded = tf.Model(file, tensors, ARCHITECTURE, 'cpu'), tf.Model(file, tensors, ARCHITECTURE, 'cpu', cache='float16')
            with torch.no_grad():
                x = plain.rows('token_embd.weight', TEXT) * 4.0
                (k, v), (kr, vr) = plain.keys(0, x, torch.arange(len(TEXT))), rounded.keys(0, x, torch.arange(len(TEXT)))
            torch.testing.assert_close(kr, k.half().float())
            torch.testing.assert_close(vr, v.half().float())
            self.assertFalse(torch.equal(k, kr))
            number(rounded.forward(TEXT)).backward()
            self.assertTrue(all(bool(torch.isfinite(x.grad).all()) and float(x.grad.abs().sum()) > 0 for x in rounded.leaves.values()))

    def test_what_the_cache_holds_is_float16_and_the_derivative_passes_through(self):
        x = torch.tensor([0.1, 1.0000001, -3.3333333], requires_grad=True)
        y = tf.half(x)
        torch.testing.assert_close(y.detach(), x.detach().half().float())
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones(3))


if __name__ == '__main__':
    unittest.main()
