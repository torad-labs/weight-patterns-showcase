# Resolve incomplete completions and preserve the Silver contrast

The frozen twenty-case run found three answers cut at eight tokens: probe-08, probe-09 and probe-16. Re-run those exact prompts with the same original checkpoint, executable, device and greedy policy, cap128. Compare their emitted first eight token IDs with the saved short runs before treating the extension as the same continuation. A continued cap hit remains explicitly incomplete.

A matched Silver contrast has emerged from the observations: Nek=Silver in row2 returns24; the same association in row3 returnsSilver. Capture the answer-boundary state of these two cases, with all35 named KV views, using cap8 as in their original runs. Compare both emitted token sequences and actual prompt counts against the originals. A behavior-changing diagnostic capture is not used as the original's state. The standard tap-layout limitations remain; these captures do not assert complete measurement of all internal operations or causal identification.

Purpose: retain actual completions and an observed failure/success relationship for mechanism inspection. No weights edited; no vector, component or edit rule chosen in advance. These are reporting and observation operations, not a second model forward implementation.
