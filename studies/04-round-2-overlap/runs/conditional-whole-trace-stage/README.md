# Input-conditioned transformations and the timing of descent

**Neither incoming-conditioned rule gave an on-time warning on either wrong descent in the held physical family or on the reordered problem.** Conditioning improved class separation at retrospectively aligned sites. Those sites could seldom be identified in the free-running computation by the frozen input matcher. Removing one successful witness also reversed the aligned separation for one alternative valid route.

## What was measured

We inherited 25 discovery executions from 14 physical groups and held out all three executions in group 64–40. We kept the previously missed reordered problem as a separate challenge. The eligibility scan covered all 387 fully shared generated reference events and all 105 merges. The fixed comparison used the selected original reference. The conditional comparison fitted the actual incoming direction and log norm using a convex mixture of successful executions, excluding the query's physical group. The same coefficients predicted the witnesses' actual directional turns and log gains. The fitted reference itself belongs to the discovery set. Frozen plan · [Source bindings](declaration.json)

For every queried event the embedding supplied the original starting direction. Adding the exported directional increments reconstructed the incoming directions at successive merges. The actual model's 35 between-block scale values were all positive. Direct readings of all 105 native ADDs at all 388 generated events of the reference gave a maximum discrepancy of 0.0000002001 in unit-vector L2 distance. On the reordered trajectory the maximum was 0.0000002046. The endpoint log norms came from each retained native measurement. All files used were already local. Reference reconstruction · Reordered reconstruction

Per-cell alternatives all used the same reference correspondence. A many-to-one match was represented by the witnessed row with highest whole-turn cosine to the reference. Each rule selected its earliest reference row with a strictly positive gap between the largest successful and smallest failed discovery score, then the largest normalized gap at that row. To find the corresponding computation online, the rule scanned the current and previous incoming directions with a group-excluded threshold. The arithmetic annotations were opened after all four sites and their banks had been saved. Sites and calibration · Every registered event and audit

| Discrepancy | Separating sites | Selected reference row / merge | Largest success | Smallest failure | Discovery target matches admitted |
| --- | ---: | --- | ---: | ---: | ---: |
| Fixed direction | 421 | 314 / 23 | 0.05351 | 0.05579 | 2 / 88 |
| Fixed direction and log gain | 351 | 299 / 65 | 0.15096 | 0.15128 | 0 / 77 |
| Conditional direction | 1,352 | 314 / 41 | 0.05856 | 0.32389 | 0 / 88 |
| Conditional direction and log gain | 1,195 | 314 / 11 | 0.00668 | 0.03382 | 4 / 88 |

The reference made its sign choice at native row 313. Both conditional selections land at row 314. The earlier row 299 from fixed direction and log gain had no accepted discovery matches under the frozen causal matcher. The calibration denominators count DTW-matched events across discovery, including repeated matches to one reference event. The solver certified all 25 group-excluded mixtures at 38,635 of 40,635 sites. We excluded the other 2,000 sites from conditional selection. Of 1,015,875 individual fits, 2,682 missed the numerical optimality certificate. They remain marked in the complete map. The complete score map also has a byte-verified copy at `torad-research:traces/conditional-whole-trace-stage/discovery-maps.npz`.

## What happened on the held routes

| Free generation | Independently audited descent | Conditional direction | Conditional direction and log gain |
| --- | --- | --- | --- |
| discovery-05-channel-h1 | Wrong | No match | No match |
| expanded-020-h1 | Wrong | No match | No match |
| expanded-023-h1 | Correct | No match | One match two events after the annotated precursor, below the failure cutoff |
| discovery-02-reordered-h3 | Wrong | No match | No match |

Neither fixed-reference rule registered a site in those four trajectories. All four trajectories used valid conditional friction subtraction. None raised an alert at that operation, but missing coverage also accounts for silence. These data cannot establish friction preservation or broad correct-route specificity. The known 387–388 reversal uses another saved history and was outside this bounded frozen-group analysis. [Outcome counts](result.json)

## What the stronger aligned signal depends on

At conditional-direction row 314, all 14 successful queries scored between 0.00268 and 0.05856, while all 11 failures scored between 0.32389 and 0.33462. The successful family-excluded banks were affinely independent at the selected site under NumPy's double-precision matrix-rank criterion; fitted solutions had two to six positive weights. Two separate donor deletions erased the separation. For example, removing `expanded-071-h1` raised the correct `expanded-039-h3` from 0.00742 to 0.39479, above the smallest failure. Neither matching the input nor achieving a strict discovery gap established coverage of this successful route without that witness. Per-query weights, values and donor deletions

The existing whole-history correspondence can retrospectively locate a class difference. Its ability to place that difference does not automatically provide a prefix-only method to locate it during generation. A next selection rule could require measured group-excluded stage coverage while it chooses the candidate, and compare support across successful witnesses. That would be a new rule.

The run executed `measure.py` with system Python, NumPy, a C++ compiler and two OpenBLAS threads. It assembled one small simplex problem per query and candidate site on CPU. The experiment made no model update. File sizes and SHA-256 digests
