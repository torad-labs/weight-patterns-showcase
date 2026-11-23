# Direct computational weight update

Launched November 16, 2026, 03:56:14 AM CT under the sole controller
`weight-patterns-vector-gradient.service`. Declaration SHA256
`0b429a835053ac6362dc8a45295542ae20a5302b1d3cd58a1fa4d2180afbdef4`.

One complete weight derivative, one measured directional response, one native
BF16 rounding/installation, then fresh native generation and full computational
remeasurement. Original reference/tolerance, all 105 merges and 70 matrices are
retained. No CGLS, candidate search, token comparison or output scoring.

The revised method is recorded in `journal/2026-11-16-direct-computational-update.md`.
The independent reviewer's source-bound YES and 21 tests plus 5 subtests are bound in the declaration.
`startup.json` records the actual command; `handoff.json` records the explicitly
stopped unsaved predecessor with no installed updates. Its records are intact.

The fresh native baseline reproduced exactly at 03:59:41 AM CT: loss
0.00029234223350039747 and 4,386 outside cells. The process exited at
03:59:47 before differentiation because its CUDA allocator was uninitialized
when resetting peak statistics. No gradient, proposal or installed update
exists. The separately declared `vector-chance-gradient-2-original` fixes
initialization; all these records remain. No improvement or closure is claimed.
Actual observations, GPU phase timings/peak memory and reached-state identities
are written under `loop/`; the existing port-5000 observer publishes this run
in `computational-training` with phases separate from native computational loss.
