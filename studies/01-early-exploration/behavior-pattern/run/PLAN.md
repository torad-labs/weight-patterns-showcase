# Varied lookup behavior: twenty original-model questions

Purpose: test whether the same position-instead-of-associated-value behavior repeats across varied keys, value categories, and row locations, before deriving any weight edit.

Twenty questions are frozen before generation. Cases 01 and 02 preserve Marcos's examples, including capitalization and the double space before FOotball. Cases 03–20 cover six additional tables, each with the requested association moved through all three rows. This is twenty responses over eight table families, not twenty independent semantic templates. Target rows have counts 7, 6, 7. Every question is exactly a three-row table followed by What is <key>? No answer is supplied in the prompt.

Run the saved original qgre-infer through the production planner/megakernel, original published TQ4 E2M1 checkpoint, greedy, fresh process per case, maximum128 new tokens, natural stop respected. Save complete text and emitted token IDs from stderr. The current project GPU rule requires physical device0 through gpu_card_run0; earlier discovery captures used device1, so cross-study comparisons do not assert same-device control.

Primary observation: number of responses whose answer is the one-based requested row instead of its associated value. Report also correct associated answers, other numeric answers, wrong values, and incomplete/other output. Read complete responses manually; a number in an explanation is not a position answer. A pattern limited to one family is not cross-family replication. Mixed or absent position errors refute treating this prompt family as an established universal position mechanism. Successful behavior is a required comparison, not an excluded sample.

Original weights only. No edit formula, layer or direction selected in advance. A repeated output pattern alone does not prove a unique internal cause or justify a particular edit. This small behavior run makes no performance claim and does not instrument internal states; capture relevant matched trajectories next only if observations justify it. Existing held-out edit confirmation cases remain unused.
