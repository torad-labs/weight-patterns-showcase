# What the unbanded loss measures on known weight repairs

Both earlier repairs of the fitted 91 J question receive much lower downhill loss than their common starting checkpoint under the 27-donor objective. The unsuccessful edit receives higher downhill loss than its own starting checkpoint. This comparison reuses native free generations and the exact registration and residual functions used by training. All five answers subtract friction correctly.

| Natural generation on the 91 J question | Downhill mean squared residual | Friction mean squared residual |
|---|---:|---:|
| Early checkpoint 27 | 0.0170374 | 0.0011767 |
| Attention repair of 27 | 0.00000312 | 0.0026003 |
| Feed-forward repair of 27 | 0.00000140 | 0.0033049 |
| Original model | 0.0174526 | 0.0012613 |
| Unsuccessful projected edit of original | 0.0178241 | 0.0004407 |

The table holds 1,132 downhill source-event/merge pairs and 1,050 friction pairs fixed across all five executions. Each pair receives equal weight after averaging its squared residual over its registered visits. These numbers describe the fitted question. The joint training loss also includes the 44 J preservation question and role weights. On the larger intersection containing just checkpoint 27 and its two repairs, downhill error falls 99.981% and 99.990% and the friction error rises 43.79% and 77.22%. A higher friction residual can coexist with valid friction on a corrected trajectory. The endpoint comparison does not establish opposition between two local weight gradients. [All measurements](result.json) · [Production objective](../../../code/signal-trace/pipeline/weight_edit/group_contrast_objective.py) · [Independent native operation audits](../../../studies/04-round-2-overlap/runs/same-input-operation-comparison/README.md).

## Where feedback disappears

On the common support, only two downhill measurements precede the first symbolic downhill sign in all five historical controls. Both belong to one source event and diminish under each genuine repair. Both increase under the unsuccessful edit. At the original training weights they registered three predictions before the downhill sign. After the first installation the regular matcher returned no visit for either. Both were still absent after the second installation. Consequently the second weight update received no local gradient from either measurement. Other registered cells could still affect the earlier computation through shared parameters. A cell missing from a free generation provides no measurement of its error.

The first native prediction that selected a different token among the five historical states was at event 244. The downhill loss had no measured cell anywhere in that shared input history. Once the latest training edits are included, only four successive predictions have identical inputs across all eight states. Across the 105 merges at those four predictions, the two successful changes have cosine 0.576. The cumulative two-update training change has cosine 0.028 with the attention repair and 0.078 with the feed-forward repair. Each edit was differenced against its own worker and starting weights. The starting traces themselves differ slightly between workers. Four predictions on one question cannot identify a transferable direction. [Early movement at every shared prediction](early-movement.json).

## Reading the correspondence in both directions

The training matcher asks which reference stage best matches the query prefix observed so far. A reference stage can consequently receive no visit. We also asked the converse question on each completed generation, using the same all-merge similarity matrix and prefix cost. Both directions returned exactly the same event for the selected stage on all five historical controls and the training baseline. In each case that event occurred three predictions before the downhill sign.

For both new weight states the reverse read found event 355, ten predictions before the downhill sign. The forward read at 355 chose the immediately preceding reference stage, 309. At the reverse candidate the first component changed sign and decreased in absolute error, while the second increased in absolute error. On both genuine repairs the absolute errors of both components had decreased. Inspection of the emitted arithmetic locates event 355 at the opening of the post-descent energy equation. The baseline match is farther into that equation. A backward lookup alone cannot qualify the two events as equivalent or show that the changed registration caused the failed repair. [Per-event signed measurements and both mappings](reverse-registration.json).

The downhill group contrast can recognize a successful outcome on this question. Its coverage and its individual components matter when choosing an update. Using the reverse direction in training requires distinguishing an alternative alignment from an operation genuinely absent from the new generation. Independently audited correct and failed routes, including reordered operations, can test that distinction. The friction comparison also shows why agreement across successful computations matters when constructing the preservation objective.

## Reproduction

From the repository root, set `PYTHONPATH=code/signal-trace` with the environment used for the group learner. Each program takes one argument, the root of a restored archive tree. The readers first use a locally present file and otherwise resolve the same relative path under that root. They verify native stream bytes and index digests against the recorded captures.

1. Run compare.py to register five generations with the frozen production objective and read the selected cells back from the three training boundaries.
2. Run early_movement.py to compare each installed change over the input history common to the five controls and three training boundaries.
3. Run reverse_registration.py to look up missing stages with the two arguments of the original prefix matcher exchanged. The complete natural generation is available to this retrospective correspondence check.
