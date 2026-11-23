# Original-model overlap analysis

All commands run from `code/signal-trace`. Read the repository AGENTS.md and
ROUND-2.md first. Every calculation uses the existing declaration and the drift
verifier. No command here edits model weights or launches inference.

The run is `studies/04-round-2-overlap/runs/overlap/` relative to the research root.
Its declaration must be committed before a new output phase starts. Completed
phases refuse overwrite. Do not rerun them into the same directory.

## Data flow

1. `python -m pipeline.overlap.analyze RUN`: separate group means, dispersion,
   member omission behavior, then raw group differences. Whole-vector agreement
   is descriptive and does not filter individual components.
2. `python -m pipeline.overlap.measure_common RUN`: independent coordinate
   envelopes, their common signed gap, coverage, gain differences and changes
   under member omission. This is a conservative native-coordinate observation.
3. `python -m pipeline.overlap.figures RUN` and
   `python -m pipeline.overlap.common_figures RUN`: views of those measurements.
4. `python -m pipeline.overlap.compact RUN`: lossless packing of member arrays
   with equality checked on read-back. Indexes preserve the member mappings.
5. `python -m pipeline.overlap.attribute RUN ORIGINAL_MODEL`: offline weighted
   contribution accounting. It requires the original model to be an explicitly
   permitted read-only input in the declaration and method guard. That allowance
   is recorded in `round-2-method.json`. It verifies the model's stored digest,
   decodes native BF16/f32 according to the bindings, reconstructs each native
   turn, and separately averages each group's components. Common-axis shares
   describe the observed mean contrast along the common axis; they do not
   pretend the synthetic interval gap was an execution.
6. `python -m pipeline.overlap.summarize_components RUN` and
   `python -m pipeline.overlap.component_figures RUN`: signed component shares,
   concentration, cross-passage profiles, reconstruction errors and coverage.
   These profiles are not controlled for speed and are not ablation results.
7. `python -m pipeline.overlap.measure_cross_validation RUN`: leave each member
   out before estimating the group centroids, direction and midpoint; repeat
   inside exact-speed strata. Record undefined/small strata explicitly. Keep
   within-speed predictive support separate from speed-standardized group
   differences. Neither purifies the full-data vector into an edit target.
8. `python -m pipeline.overlap.summarize_cross_validation RUN` and
   `python -m pipeline.overlap.cross_validation_figures RUN`: all aligned
   subsets as the main view, full-population positions separately, individual
   group scores and margins, stability, and matched population counts. The
   compressed CSV exports every position/merge row. Plot modules need matplotlib
   (`uv run --with matplotlib --with numpy python -m ...` in this environment).
9. `python -m pipeline.overlap.member_scores RUN` and
   `python -m pipeline.overlap.member_figures RUN`: one score per response over
   its own climb and descent positions, with speed, missing strata and
   equal-response summaries. These reuse saved held-out margins and do not
   estimate new directions. Figures live in the new phase's own
   `cross-validated/per-response-figures/` folder.

The original inputs are under
`shared/observation/original-model-traces/full-response-turns` and
`shared/observation/original-model-traces/matrix-inputs`, together with the
preserved correspondence and original native graph metadata. Bulk derived
arrays are local and reproducible from those original records. No reserved
question outcomes are read.

Current coverage: direction and size at all 105 merges; weighted component
inputs at 46 attention/feed-forward merges in 23 layers. The other merges must
remain visibly unattributed unless existing original records supply them.

Tests: `python -m unittest discover -s tests -p 'test_overlap_*.py' -t .`.
The envelope tests include member-only coordinates, varying-strength common
directions, interval overlap, and the every-cross-group-difference guarantee.
The component tests pin residual normalization and signed accounting.
The held-out tests cover rotations, exclusion before averaging, unequal speed
mixes, missing strata, and a member-specific component that survives in a
full-data mean despite positive held-out predictions.
