import { expect, test } from "bun:test";
import { captureMatch, validateSpan, matrixRoute } from "./weight-study";

test("a diagnostic decode consumes token i and predicts i+1, without claiming trajectory equivalence", () => {
  const reply = { id: "reply:1", promptTokens: 10, tokenIds: [7, 8, 9] };
  expect(captureMatch(reply, 11, 8, 9)).toEqual({ replyId: "reply:1", inputIndex: 1, outputIndex: 2, inputMatches: true, outputMatches: true });
  expect(captureMatch(reply, 11, 8, 99).outputMatches).toBe(false);
  expect(captureMatch(reply, 12, 9, 10).outputMatches).toBe(false);
});

test("token selections are nonempty bounded half-open intervals", () => {
  expect(validateSpan({ replyId: "r", start: 0, end: 3 }, [{ id: "r", tokenIds: [1, 2, 3] }])).toBe(true);
  for (const [start, end] of [[-1, 2], [2, 2], [1, 4], [0.5, 2]]) expect(() => validateSpan({ replyId: "r", start, end }, [{ id: "r", tokenIds: [1, 2, 3] }])).toThrow();
});

test("matrix relationships name their architectural basis and leave unrecognized weights unmapped", () => {
  expect(matrixRoute("model.language_model.layers.19.mlp.down_proj")).toMatchObject({ layer: 19, input: "geglu", output: "mlp_down", recordId: "l19:main:10" });
  expect(matrixRoute("model.language_model.layers.2.per_layer_projection").recordId).toBeNull();
});
