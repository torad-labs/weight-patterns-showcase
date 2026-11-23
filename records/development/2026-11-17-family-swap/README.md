# Which changed weight family selects the later history?

The [crossed-history measurement](../2026-11-17-crossed-history/README.md) established a full fresh loss reduction between native 175 and 189 that follows the later freely generated history on this fitted prompt. Both sets of weights gave that supplied history a substantially lower computational loss. The open question is which differences in the stored model tipped the free generation.

I compared the actual native BF16 codes for every tensor in the complete seed-relative checkpoint override inventories, including locations modified in exactly one checkpoint. Both checkpoints have the same verified underlying GGUF. All 35 attention-output matrices changed, accounting for 54,889 changed native coefficients and squared displacement 0.000001237971026. All 35 feed-forward down matrices changed, accounting for 100,125 changed coefficients and squared displacement 0.000000218116786. Every one of those code differences also changed the represented numerical value. The other model arrays share the same underlying seed bytes. [All 70 matrix comparisons](families.json).

The two new native conditions use independent private copies of that seed. Each physical tensor receives its verified sparse overrides from exactly one chosen donor. One model has the attention-output family from 189 and the FFN-down family from 175. Its reciprocal contains the attention-output family from 175 and the FFN-down family from 189. A small independent numeric fixture exercised the actual native sparse installer on both recipes, including a location present only in the opposite donor that had to stay at its original seed value.

Both hybrids generate freely from the same fitting prompt on physical GPU1 with the same engine and unfiltered native callback geometry as the two verified free endpoint controls. All 105 measured merges and all consumed positions enter fresh independently recomputed computational correspondence and loss against the unchanged reference and chance array. Endpoint histories are used only after generation to measure where choices diverged. This factorial comparison measures family and interaction effects for the specified endpoints on one prompt. It does not identify individual weights, a single historical update, semantic correctness or performance on new questions. [Prospective experiment](../../../studies/04-round-2-overlap/runs/vector-chance-family-swap-175-189/declaration.json).

## Native results

The FFN-down donor determined which complete generated history appeared in all four conditions. With either set of attention-output weights, the step-175 FFN-down weights produced the 175 history and the step-189 FFN-down weights produced the 189 history. The first differing generated choice was index 381, corresponding to consumed position 554. Each hybrid's independently recomputed correspondence also exactly matched that of the endpoint sharing its FFN-down family.

| Attention-output donor | FFN-down donor | Generated history | Consumed positions | Whole-trace loss |
|---|---|---|---:|---:|
| 175 | 175 | 175 | 605 | 0.00009736442 |
| 189 | 175 | 175 | 605 | 0.00009730706 |
| 175 | 189 | 189 | 613 | 0.00004280245 |
| 189 | 189 | 189 | 613 | 0.00004259006 |

On the 175 attention background, substituting the 189 FFN-down family reproduced 99.61% of the full endpoint loss decrease. On the 189 attention background, changing the same family reproduced 99.90%. All 105 individual merge losses decreased in each comparison. The 189 attention-output family also reduced total loss on both routes, by 0.10% and 0.39% of the endpoint decrease, respectively. The signed two-family interaction on the loss scale was −0.00000015503. These conditional percentages share the same denominator and the interaction makes them background-dependent.

The FFN-down family caused the switch and reproduced almost the entire full-model trace-loss change for this fixed prompt and these two checkpoints, despite accounting for only 14.98% of the squared weight movement. This intervention resolves the family responsible for choosing this route. Individual blocks, individual coefficients and transfer to other prompts require separate evidence. Correctness of the physical reasoning is a separate observation from distance to this frozen reference. [Native result with measurement hashes and per-merge comparisons](native-result.json).
