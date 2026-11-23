# How far a computation prefix can place the decision

At each new event the reader uses all of the query computation seen so far. It finds the frozen reference prefix with minimum accumulated cosine distance between the 105-merge turn fields, retaining every exactly tied endpoint. Both paths start at their first generated computation. The reader sees the full fixed reference and has no access to future query computation. A nearest reference prefix exists even for an unfamiliar query, so this experiment measures timing. It supplies no correction gradient or indication of confidence. [Prospective rule](declaration.json) · All prefix endpoints.

| Query | Native pre-choice event | Reference event matched at that moment | When the historical reference pre-choice was reached |
| --- | ---: | ---: | ---: |
| Fitted mistake | 312 | 312 | At pre-choice |
| Other same-template mistake | 325 | 312 | At pre-choice |
| Correct held answer | 312 | 312 | At pre-choice |
| Reordered mistake | 258 | 261 | 88 events after pre-choice |

Event 312 is the independently annotated pre-choice on the reference. It was read from the finished map as a historical control. The reader computed every reference match without using that number. The held physical family had already been inspected in earlier work. These are development controls.

The frozen reference event selected by the earlier fixed-direction-plus-log-gain analysis was reached 15, 15 and 13 events before the three same-template decisions. The conditional-direction candidates were reached two events after all three. In the reordered response both sets of candidates were reached late and repeatedly, even though the model made the same kind of error. Knowing when the reference and query coincide on one template establishes timing only. Evaluating the frozen discrepancy at the matched computation is a separate measurement. [Results by candidate](result.json).
