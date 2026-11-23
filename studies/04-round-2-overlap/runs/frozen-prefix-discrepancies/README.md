# Frozen discrepancy scores at whole-prefix matches

**With correspondence based on the entire available computation, three of four frozen scores separated the two held-family descent mistakes from the valid answer. All three did so two events after the audited prechoice.** An earlier score flagged one of the two errors, 15 events before the prechoice, and stayed below threshold on the valid answer. On the reordered problem every flag came after the wrong descent. [All scores, witness coefficients and event offsets](result.json)

Only correspondence changed. We used the saved prefix endpoints, the four original sites and thresholds, and the original successful witness banks. For each new input, the original scoring function computed its mixture of those witnesses from that input. It used the same domain-group exclusion, score and threshold as before. We kept every matching prefix, including repeated visits and ties. Outcomes and event names were used only to audit the scored chronology. [Source-bound declaration](declaration.json)

| Frozen score | Failure 1 | Failure 2 | Valid answer | Reordered failure |
| --- | ---: | ---: | ---: | --- |
| Fixed direction, cutoff 0.05351 | **0.05591**, +2 | **0.05781**, +2 | 0.00057, +2 | Three flags, +92 to +94 |
| Fixed direction and log gain, cutoff 0.15096 | 0.15041, −15 | **0.15367**, −15 | 0.00209, −13 | Four flags at +50, +59, +60, +61. Two later matches below cutoff |
| Conditional direction, cutoff 0.05856 | **0.32621**, +2 | **0.32161**, +2 | 0.00524, +2 | Three flags, +92 to +94 |
| Conditional direction and log gain, cutoff 0.00668 | **0.03381**, +2 | **0.03387**, +2 | 0.00004, +2 | Three flags, +92 to +94 |

Offsets count generated events relative to each answer's audited prechoice. Bold scores exceed their original discovery cutoff. The first three columns are `discovery-05-channel-h1`, `expanded-020-h1` and `expanded-023-h1` in the held physical family. On the reordered failure, the three matches at +92 through +94 were also two through four events after the valid friction prechoice. That coincidence prevents attributing those flags specifically to the earlier wrong descent. None of the four rules registered the reordered descent by its prechoice. Original timing and split

The local two-input matcher had abstained on both held-family errors. Recovering the same stage from the entire available prefix resolved that registration failure on these two cases. It establishes useful retrospective classification on this small family. Earlier training credit would require a reliable discrepancy at an eligible earlier computation, or another way of assigning the late discrepancy to its upstream cause. [Registration comparison](../conditional-whole-trace-stage/README.md)
