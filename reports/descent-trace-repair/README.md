# Training on Local Divergence Between Computational Traces

[Read the PDF](paper.pdf) · [Read on a phone](paper.html) · [Edit the manuscript](paper.typ) · [Editorial review](review.md) · [All manuscripts](../README.md)

This November 18, 2026 working paper follows one repaired descent in Gemma 4 E2B, the separate operations it changed and preserved, and the tests in which a candidate detector missed the error or would have moved correct computation. The paper includes six figures, all 74 consecutive fitted observations from updates 317 through 390, and the eleven-update prospective continuation to 401. The 35 recordings used to audit friction include repeat observations of the same prompt.

| To inspect or reproduce | Entry point |
| --- | --- |
| Fitted case study and its limits | [PDF](paper.pdf) · [mobile HTML](paper.html) |
| Computational objective and native update | [Final Experiment](../../FINAL-EXPERIMENT.md) · [running the baseline](../../code/signal-trace/TRAINING.md) · [entry point](../../code/signal-trace/pipeline/weight_edit/final_experiment.py) |
| N · Training the shared direction and testing it on four additional configurations | [Eleven native updates](../../records/development/2026-11-18-prefix-direction-training/README.md) · [Paired transfer study](../../studies/04-round-2-overlap/runs/prefix-direction-generalization/) · [Independent physics audit](../../records/retrospectives/2026-11-18-prefix-direction-blind-physics-review.md) |
| S · Two successful interventions and a failed edit on identical activity histories | [measurements, directional figure and controls](../../studies/04-round-2-overlap/runs/same-input-operation-comparison/README.md) · [independent reconstruction](../../records/retrospectives/2026-11-18-same-input-operation-review.md) |
| Read the evidence in research order | [research record](../../final-experiment/README.md) |
| Inspect the historical weight slice and paired trace field | [interactive figure](../../studies/04-round-2-overlap/runs/one-pair-landscape-and-traces/explorer.html) |
| Register and evaluate future operation-level predictions | [prospective protocol](transfer-protocol.md) · [offline scorer](score_forecasts.py) |

## Questions and their roles

| Role in this study | Identifier | Independently calculated available / required energy |
| --- | --- | --- |
| Fixed reference, also regenerated separately to assess preservation | `expanded-004-h1` | 44 / 20 J, YES |
| Only response supplying a fitted loss | `discovery-05-channel-h1` | 91 / 20 J, YES |
| Development transfer and crossed-context question | `discovery-10-channel-h1` | 115 / 30 J, YES |
| Separately reserved transfer for the one-update comparison | `discovery-02-reordered-h3` | 47 / 60 J, NO |

The two transfer questions are distinct. The latter had screened positive for the original model’s descent error on a prior public screen. It was selected before reading its trace, sign margin or edited outcome for this comparison. No sealed cohort supplies a result in this paper.

## Source map

| Key | Records |
| --- | --- |
| M | [Final Experiment specification](../../FINAL-EXPERIMENT.md), including model and native identity, reference selection, correspondence, tolerance, loss, gradient, parameter inventory and installation. Fourteen nonself witnesses supply variability calibration. They are distinct from fourteen additional training targets. |
| H | [Checkpoint arithmetic and route changes](../../records/retrospectives/2026-11-17-one-pair-correction/answer-audit.md), [audited observations 317–362](../../records/retrospectives/2026-11-17-one-pair-correction/fitted-outcomes.json), [forecast comparisons and geometry](../../records/retrospectives/2026-11-17-one-pair-correction/README.md), [derivation of the alignment-conditional bound](../../records/retrospectives/2026-11-17-one-pair-correction/bound-review.md), [original development control distributions](../../records/retrospectives/2026-11-17-one-pair-stopping-calibration/README.md), [figure data for updates 363–390 with hashes](figure-1-continuation.json). |
| W | [Parameter displacement and fixed-history controls](../../records/retrospectives/2026-11-17-direct-weight-read/README.md), [reciprocal history intervention](../../records/development/2026-11-17-crossed-history/README.md), [route-selecting weight-family experiment](../../records/development/2026-11-17-family-swap/README.md), [arithmetic-repairing weight-family experiment](../../records/development/2026-11-17-descent-sign-swap/README.md). |
| C | [Reference arithmetic and selection](../../records/retrospectives/2026-11-17-descent-reference-adequacy.md), [frozen checkpoint results](../../records/development/2026-11-17-reference-preservation/profile-result.md), [reciprocal context interventions](../../records/development/2026-11-17-descent-context/native-result.json), [one-word intervention](../../records/development/2026-11-17-descent-context/heading.md), [sign readout and local loss](../../records/retrospectives/2026-11-17-descent-transfer/README.md). |
| G | [Earlier pre-sign measurement and support sensitivity](../../records/retrospectives/2026-11-17-pre-sign-localization-history-review.md), [945-cell survey](../../records/retrospectives/2026-11-17-native-tolerance-recurrence/analysis.json), [all distances](../../records/retrospectives/2026-11-17-native-tolerance-recurrence/radial-separation.json), [frozen direction and calibration](../../records/retrospectives/2026-11-17-native-tolerance-recurrence/axis-row312-merge61.json), [separate component and activation measurements](../../records/retrospectives/2026-11-17-natural-descent-decisions/README.md). The selected original aligned measurements use F32 exported turns with the saved F64 reference and calibration. The fresh device-local acquisition has its own observed baseline. |
| U | [Prospective reserved case selection](../../records/development/2026-11-17-isolated-pre-error-update/transfer-selection.md), [native installation and all losses](../../records/development/2026-11-17-isolated-pre-error-device-continuation/native-result.json), [matched experiment report and device correction](../../records/development/2026-11-17-isolated-pre-error-device-continuation/RESULTS.md), [nine generated answers](../../records/retrospectives/2026-11-17-natural-descent-decisions/updated-answers.json), [paired row, actual sign and pre-sign scan](../../records/retrospectives/2026-11-17-natural-descent-decisions/transfer-correspondence.json). |
| | [Fixed-history native projection and complete local losses](../../studies/04-round-2-overlap/runs/directional-intervention-fixed-history/README.md) and [competing gradient calculation](../../studies/04-round-2-overlap/runs/directional-intervention-gradient-competition/README.md). |
| | [Finite forward and native engine](../../studies/04-round-2-overlap/runs/directional-intervention-anchored-replay/README.md), [fitted and correct free regenerations](../../studies/04-round-2-overlap/runs/directional-intervention-projected-native/README.md), [all eight Q/K/V substitutions](../../studies/04-round-2-overlap/runs/directional-intervention-block0-qkv-factorial/README.md), [key prediction using the original weights’ native keys](../../studies/04-round-2-overlap/runs/directional-intervention-block0-native-k-anchor/README.md), the [full downstream native-key forecast](../../studies/04-round-2-overlap/runs/directional-intervention-full-native-k-qualification-retry/README.md), and its [second-displacement replication](../../studies/04-round-2-overlap/runs/directional-intervention-native-k-second-displacement/result.json). |
| P | [Earlier repair retrospective](../../records/retrospectives/2026-11-09-round-two-repairs/STUDY.md), [D1 report](../../studies/03-round-1-single-pair-edits/D-background-sensitivity/run/REPORT.md), [single-pair paper](../fixing-a-reasoning-mistake-by-editing-the-weights.pdf), [method guide](../how-the-trace-derived-edit-works.pdf), [many-trace paper](../how-the-path-edit-works.pdf), [cohort-specific counts from its native result](../../studies/04-round-2-overlap/runs/a2-path/result.json), [conditional climb overlap](../../studies/04-round-2-overlap/runs/joint-climb-conditional-overlap/README.md), [iterative learning proposal](../Iterative_Learning_from_Computational_Traces.docx). |
| R | [Paired landscape and full decode](../../studies/04-round-2-overlap/runs/one-pair-landscape-and-traces/result.json), [gain and repair controls](../../final-experiment/descent-repair-controls.md), [reference registration and downstream comparisons](../../final-experiment/landmark-route-comparison.md). |
| O | [Operation audit and stage analysis](../../final-experiment/operand-identity.md), [annotated arithmetic](../../studies/04-round-2-overlap/runs/descent-operand-identity/annotations/events.json), [local and prefix scores](../../studies/04-round-2-overlap/runs/descent-operand-identity/result.json), [stage scores](../../studies/04-round-2-overlap/runs/descent-operand-identity/stage-results.json), [one-factor physical contrasts](../../studies/04-round-2-overlap/runs/descent-operand-identity/contrast/README.md). |
| T | [Native donor comparison by matrix](../../studies/04-round-2-overlap/runs/descent-donor-on-descendants/result.json), [immutable source identities](../../studies/04-round-2-overlap/runs/descent-donor-on-descendants/declaration.json), and [the independent transplant and joint-training design](../../final-experiment/accumulating-repairs.md). |
| V | [Prespecified eight-formulation panel](../../studies/04-round-2-overlap/runs/frozen-energy-generalization/declaration.json), [blind arithmetic judgments](../../studies/04-round-2-overlap/runs/frozen-energy-generalization/blinded-readings.json), [paired outcomes with native source digests](../../studies/04-round-2-overlap/runs/frozen-energy-generalization/paired-outcomes.json), [same-device fitted measurements](../../studies/04-round-2-overlap/runs/frozen-energy-generalization/matched-fits.json) and [development stopping rule](../../code/signal-trace/TRAINING.md). |
| P, outside-interval prediction | [Signed prediction before the many-trace edit](../../studies/04-round-2-overlap/runs/a2-path/declaration.json) and [measured changes inside and outside the interval](../../studies/04-round-2-overlap/runs/a2-path/result.json). |

## Rebuilding the manuscript

From this directory, run the following commands in the research checkout. The first command uses only the Python standard library. The second uses the Python Typst binding. The SVGs are included, so rendering the manuscript alone requires only the source and its figures.

```sh
python3 make_figures.py
uv run --no-project --with typst==0.15.2 python render.py
```

The continuation supplement records SHA-256 digests of the native measurement and complete generated result for each added point. Its local generation 15 corresponds to cumulative update 363. Figure 1 includes seven wrong generations in the full 74-point interval. The last two appear at 371 and 387. Figure 5 pairs the actual energies at 387 and 388 with event-level counts in the original cohort. The two panels have different units of observation.

Figure 2 uses four final model score differences at checkpoint 175. Figure 3 uses one observed value per member at the selected development cell. Figure 4 shows two individual positions and one measured maximum on the reserved question. Its markers do not interpolate a measured path.

Rebuilding the figures from raw measurements requires the research checkout and corresponding native records. Reading the PDF or self-contained HTML requires neither model weights nor a tracking service. The local source map also distinguishes repeated checkpoint observations from distinct evaluation questions.
