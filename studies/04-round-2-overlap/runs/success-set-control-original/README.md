# Original full-reference scale and successful leave-self-out coverage

Prospectively declared November 15, 2026 (CT), as an original-only step-3 diagnostic.
It neither changes nor gates `success-sets-loop-from-mates`.

Two separate readings use all thirty original fitting members, all 19,751 native
frames and all 105 merges. `full-score.json` measures the unchanged full reference;
`leave-self-out-score.json` removes each successful recipient's own donor points
from its sets, keeps all other points and the original mass, and leaves the wrong
members' sets unchanged. Its distinct descriptor is `leave-self-out.json`.
The second is contextual coverage evidence, not the loop's baseline or a closure
threshold. No model runs, no new labels, no no-fit data and no edited states enter.

The reused scorer/projector and all its source bindings stay unchanged. Four new
fixture tests pass: self-only exclusion across both domains, immutable source
memberships and weights, distinct full/leave-self-out readings, undefined empty
hulls, and unknown group mass retained without publishing a partial total.
The complete pipeline suite passes: 852 tests and 115 subtests. The declaration's
first revision only fixes the declarer's printed input count's static typing; the
preceding declaration is retained before any output. Editor import-resolution
warnings remain; no clean static-typecheck result is claimed.

Run on physical card 0 under its existing lock, from the repository root:

```
python3 code/signal-trace/pipeline/research_guard/verify_round2.py
flock /run/user/1000/gpu0.lock env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code/signal-trace PYTHONDONTWRITEBYTECODE=1 uv run -q --with torch --with numpy python -u -m pipeline.weight_edit.success_set_control studies/04-round-2-overlap/runs/success-set-control-original
```

Current state: completed with exit code zero before the 3:46 AM CT stop relay. All
30 fitting members, 19,751 frames and 105 merges are represented; both readings
have zero missing mass and zero unresolved projections. This diagnostic preserves
the state-valued construction's scale, not a result about the newly clarified
input-output observable. The later stop is `journal/2026-11-15-computation-not-state-values.md`.
The generated result's continuation wording describes its original scope, not a
current execution admission; the bound output is retained unchanged.

| Original-model reading | Wrong-group contribution | Correct-group contribution |
|---|---:|---:|
| Unchanged full reference | 0.02056196633077416 | 4.848181021245653e-35 |
| Each success omitted from its own sets | 0.02056196633077416 | 0.007281305407612131 |

Each group retains mass 0.5. These are weighted contributions to the declared loss,
not per-member means, answer grades, a repair verdict or an attainable floor.

Successful leave-self-out contributions, each against the other fourteen donors:

| Member | Contribution |
|---|---:|
| discovery-01-channel-h1 | 0.0000123466376334 |
| discovery-06-channel-h1 | 0.000572936061998 |
| discovery-15-channel-h1 | 0.000819292047449 |
| discovery-00-channel-h3 | 0.00205313926481 |
| expanded-004-h1 | 0.0000118958335307 |
| discovery-08-channel-h3 | 0.000223486360088 |
| discovery-10-channel-h3 | 0.000352314682361 |
| expanded-029-h1 | 0.000326692024354 |
| discovery-11-channel-h3 | 0.000291264271508 |
| expanded-064-h1 | 0.00067923324564 |
| expanded-071-h1 | 0.000606692916437 |
| expanded-039-h3 | 0.000317646598895 |
| expanded-076-h1 | 0.000297557838451 |
| expanded-078-h1 | 0.000374805290346 |
| expanded-023-h1 | 0.000342002334111 |

`result.json` binds both detailed score files and the distinct leave-self-out view
by hash. The detailed per-merge reports remain local for the existing bucket backup
path; Git keeps this summary, result, descriptor and declarations. The diagnostic
held card 0 during the running sweep; Performance reported that wait beginning at
3:36 AM CT. This one-off delay is not included in an ordinary loop pass's cost.

## Declaration-order interruption

Astra created this declaration before adding its status-page entry. Fable reports
that the running loop's next late pass was refused at 3:31:10 AM CT after thirty of
seventy matrices because the new run was not yet named on `ROUND-2.md`. The status
entry is now present and the unchanged verifier reports no drift. The loop's reached
state was preserved and its owner received that result for normal continuation.
This was Astra's publication-order error, not a numerical result or a reason to
bypass the verifier. No bound loop implementation or reader protocol changed.
