# a2-path: first errors in the model's own answers

Full reading by Astra, November 10, 2026 (CT). **All 43 scored-incomplete responses were read**, plus the originals of all 16 originally-correct regressions. The source is the 88-answer `answers.json` snapshot `b0130628cd01f306f6140437668afcaa670a632caa138f0b2d0890b1449ad387`. This adds explanations, not new generations or a changed score.

## What the remaining targeted failures actually do

Of the **13** overlap answers still failing the final-triple rubric:

- **11** consistently add the descent in the written calculation.
- **1**, `expanded-058-h3`, numerically adds it but retains a contradictory symbolic minus sign; it later counts the descent again.
- **1**, `expanded-046-h3`, still subtracts it numerically.
- **10** state the correct after-friction energy somewhere in their calculation, even though their final answer fails.

The first issues are: five reports of the after-climb remainder as available energy; three additions of climb cost to remaining energy; two initial-kinetic-energy arithmetic errors; one double use of the climb in the reachability comparison; one actual descent subtraction; one contradictory descent formula followed by reference errors.

**The three final-value labels 'descent omitted' are not what these responses did.** `discovery-10-channel-h1`, `expanded-001-h1` and `expanded-040-h1` explicitly add the descent and subtract friction correctly. Their final available fields contain the remaining energy *after the climb*. Because climb cost equals descent gain in those cases, that number coincides with 'descent omitted'. The same coincidence occurs for outside-fit `block-base` and `block-m4-base`; outside-fit `new-sledge-2` really does omit the descent.

## What broke in the originally correct answers

The **12** overlap regressions begin with:

| First new issue | Cases |
| --- | ---: |
| Descent subtracted | 4 |
| Initial kinetic-energy arithmetic (36 becomes 18, or 64 becomes 128) | 4 |
| Descent height added to climb height | 1 |
| Climb cost added to remaining energy | 1 |
| Correct post-climb remainder reported as available energy | 2 |

All three missing-field overlap responses contain substantive calculations and an earlier error; they are capped, not empty. The original responses were checked so that old misleading headings or kinetic/potential labels are not attributed to this edit.

The **4** outside-fit regressions consist of two after-climb reporting errors, one kinetic-energy arithmetic error, and one descent subtraction. The **14** remaining outside-fit failures contain seven initial kinetic-energy arithmetic errors, two direct descent subtractions, two removals of descent energy from a total that should be conserved, one actual descent omission, one mixed absolute/relative energy reference, and one self-corrected descent equation followed by an unresolved reference/velocity error.

## What this does and does not establish

- The overlap's **17 complete repairs and 12 correct-answer regressions remain the recorded score**. This reading does not move the scoring bar.
- Nine scored failures have a correct numerical energy chain and reachability decision but put the after-climb remainder in the available field. Several still have inconsistent explanatory prose. Reporting errors, arithmetic errors and incorrect physical operations should not be pooled as one capability failure.
- A score label inferred from the final number is not a reliable label for the first incorrect computation. The per-answer evidence below and in `answer-error-review.json` retains that distinction.
- The repair is not a general collapse. Nor do these answers alone decide whether to add later matrices or proceed to protection. The native state/turn overlay is a separate result.
- The 45 scored-complete responses were outside this bounded audit; no blanket claim of flawless explanation is made about them.

## Per-answer reading

Equation excerpts below use normalized notation. Full observations and original-response comparisons are in `answer-error-review.json`.

### overlap: originally wrong

- **[discovery-10-channel-h1](answers/discovery-10-channel-h1/result.json)** — The numerical chain reaches 115 J before the climb, then reports the 85 J remaining after the climb as available energy. The reachability comparison uses 115 versus 30 and is correct. The scorer's 'descent omitted' label misidentifies the mechanism; misleading zero-final-speed prose remains.
- **[expanded-001-h1](answers/expanded-001-h1/result.json)** — Correctly reaches 20 J before the climb and zero after it, but reports the zero as available energy. Correctly says it just reaches the destination. 'Descent omitted' is a numerical coincidence, not the written operation.
- **[expanded-001-h3](answers/expanded-001-h3/result.json)** — Adds the uphill potential-energy requirement to the kinetic energy remaining instead of subtracting it, then infers reachability from that sum. Also reports required energy as zero and gives the wrong YES verdict.
- **[expanded-014-h3](answers/expanded-014-h3/result.json)** — Evaluates one-half times 2 times 6 squared as 18 J rather than 36 J. Subsequent descent addition and friction subtraction are consistent with the incorrect initial value.
- **[expanded-020-h1](answers/expanded-020-h1/result.json)** — Evaluates one-half times 2 times 8 squared as 128 J rather than 64 J. Later also reports its after-climb value (136) rather than its before-climb value (156) as available.
- **[expanded-022-h3](answers/expanded-022-h3/result.json)** — After correctly calculating 24 J remaining at the destination, divides that remainder by mg and treats 1.2 m as the maximum height from the climb's bottom. This counts the climb twice for the final NO verdict, despite first stating that positive final energy means it reaches.
- **[expanded-040-h1](answers/expanded-040-h1/result.json)** — Correctly computes 114 J before the climb and 84 J after it, then reports 84 as available. Correct comparison and YES verdict. 'Descent omitted' misidentifies a reporting error; misleading zero-final-speed prose remains.
- **[expanded-045-h1](answers/expanded-045-h1/result.json)** — Reports the 110 J remaining after the climb rather than the correctly computed 140 J at its start. The numerical energy chain and positive-margin YES decision are correct; prose about zero final speed is inconsistent.
- **[expanded-046-h3](answers/expanded-046-h3/result.json)** — Adds the negative potential-energy change to kinetic energy, reducing it during descent. Also adds descent to the climb height (5 m rather than 3 m), requiring 150 instead of 90 J.
- **[expanded-058-h3](answers/expanded-058-h3/result.json)** — The descent equation says KE_i minus PE_1 but substitutes plus 40 and obtains the correct 112; the symbolic rule and arithmetic disagree. Later adds the 40 J descent contribution again to the correctly calculated 92, yielding 132; misplaces the rough section at h=1 and starts reducing the climb height. Capped before required field/verdict.
- **[expanded-070-h3](answers/expanded-070-h3/result.json)** — Reports 68 J remaining after the climb rather than the correctly computed 188 J before it. The positive-margin decision is correct; prose about the minimum final energy is imprecise.
- **[expanded-089-h3](answers/expanded-089-h3/result.json)** — Adds the uphill potential-energy requirement to the remaining energy. Then sets required energy to zero. The YES verdict happens to match.
- **[expanded-093-h1](answers/expanded-093-h1/result.json)** — Adds the uphill requirement to remaining energy instead of subtracting it. Then sets required energy to zero. The YES verdict happens to match.

### overlap: originally correct

- **[discovery-06-channel-h1](answers/discovery-06-channel-h1/result.json)** — Newly subtracts the descent contribution from kinetic energy, yielding -6 J instead of 54 J. Negative-energy rationalization runs to the cap; missing final fields are a consequence, not the first error.
- **[discovery-00-channel-h3](answers/discovery-00-channel-h3/result.json)** — Newly adds the climb requirement to the correctly computed 27 J before the climb and treats the sum as proof of reachability. Edited required energy becomes zero and verdict YES.
- **[expanded-018-h3](answers/expanded-018-h3/result.json)** — Newly evaluates one-half times 2 times 8 squared as 128 rather than 64 J. Later reports its after-climb remainder 68 as available and zero as required.
- **[expanded-024-h1](answers/expanded-024-h1/result.json)** — Newly subtracts the descent contribution, obtaining -6 J instead of 54 J. Later clamps negative available energy to zero and gives the wrong NO verdict.
- **[expanded-028-h1](answers/expanded-028-h1/result.json)** — Newly adds the 2 m descent to the 1 m climb and computes a 90 J requirement rather than 30 J. Then calls the 18 J shortfall 'required energy' and is capped. The text contains the correct 72 J before-climb value even though the final-field parser marks available missing.
- **[expanded-029-h1](answers/expanded-029-h1/result.json)** — Newly subtracts the 60 J downhill gain from 24 J, producing -36 J instead of 84 J. Runs to the cap after propagating negative energy; missing final fields are not the first failure.
- **[expanded-022-h1](answers/expanded-022-h1/result.json)** — Newly computes one-half times 2 times 8 squared as 128 J instead of 64 J. The remaining operations are consistent with the wrong initial kinetic energy.
- **[expanded-010-h3](answers/expanded-010-h3/result.json)** — Newly computes one-half times 2 times 6 squared as 18 J instead of 36 J. The required energy and NO verdict remain right, but the available energy is wrong.
- **[expanded-071-h1](answers/expanded-071-h1/result.json)** — Newly reports the after-climb remainder of 144 J as available instead of the correctly computed 184 J before the climb. The energy chain and positive-remainder YES decision remain correct; final-field meaning changed.
- **[discovery-17-channel-h3](answers/discovery-17-channel-h3/result.json)** — Newly labels the after-climb remainder of 71 J as available instead of 191 J before the climb. The full numerical chain and positive-remainder YES conclusion are correct.
- **[expanded-017-h3](answers/expanded-017-h3/result.json)** — Newly evaluates one-half times 2 times 8 squared as 128 rather than 64 J. Later reports its after-climb 72 J as available and zero as required.
- **[expanded-077-h1](answers/expanded-077-h1/result.json)** — Newly subtracts the 100 J descent gain from the 40 J kinetic energy, then treats descent as an energy barrier. The final 40/100/NO addresses this fictitious descent barrier rather than the actual climb.

### round one: originally wrong

- **[cart-base](answers/cart-base/result.json)** — Evaluates one-half times 2 times 6 squared as 18 rather than 36 J. Also subtracts the 40 J descent contribution and propagates negative energy to -38 J.
- **[cart-changed](answers/cart-changed/result.json)** — Evaluates one-half times 2 times 6 squared as 18 rather than 36 J. Also subtracts the descent contribution. NO is correct for the actual problem but is reached with wrong available energy.
- **[bead-changed](answers/bead-changed/result.json)** — Subtracts the 30 J downhill contribution from 8 J instead of adding it. Then drops the descent entirely, obtains zero after friction, and adds uphill potential energy to claim 40/40/YES.
- **[trolley-base](answers/trolley-base/result.json)** — Computes one-half times 2 times 8 squared as 128 rather than 64 J. The subsequent descent addition is correct in sign, but the wrong initial energy yields 124 instead of 60 J and the wrong YES verdict.
- **[trolley-changed](answers/trolley-changed/result.json)** — Computes one-half times 2 times 8 squared as 128 rather than 64 J. Subsequent operations give 124 instead of 60 J; verdict YES happens to be right.
- **[block-m2-base](answers/block-m2-base/result.json)** — Computes one-half times 2 times 6 squared as 18 rather than 36 J. The descent is added and friction subtracted correctly thereafter, yielding 22 instead of 40 J.
- **[confirm-carriage-1](answers/confirm-carriage-1/result.json)** — Treats 1,344 J of mechanical energy measured from sea-level zero as energy available above the 20 m climb foot, rather than subtracting its 1,200 J baseline potential. Absolute energy 1,368 before losses and 1,344 afterward are valid in their datum; mixing that datum with a 60 J relative climb is the error. No explicit final verdict.
- **[confirm-carriage-3](answers/confirm-carriage-3/result.json)** — Subtracts the downhill potential-energy change from total mechanical energy despite a smooth conservative descent. Also evaluates 6*10*(22-20) as 600 instead of 120 J, then compares absolute energy 744 with a relative 180 J climb requirement.
- **[confirm-disc-2](answers/confirm-disc-2/result.json)** — Evaluates one-half times 2 times 8 squared as 128 instead of 64 J. Then subtracts the 20 J descent contribution and 24 J loss to obtain 84 instead of 60.
- **[confirm-disc-4](answers/confirm-disc-4/result.json)** — Evaluates one-half times 2 times 8 squared as 128 instead of 64 J. Also subtracts descent, producing 84 instead of 60 and an incorrect YES verdict.
- **[confirm-runner-3](answers/confirm-runner-3/result.json)** — Initially subtracts gravity's work in an equation between total mechanical energies, then explicitly corrects this to conservation. Capped before the required field/verdict. Distinguish the repaired descent equation from the later unresolved datum/velocity errors. First unresolved issue: After friction, retains the pre-friction velocity 10.20 m/s and calls absolute mechanical energy 720 J available at a 20 m foot; the available kinetic energy is 120 J.
- **[confirm-runner-5](answers/confirm-runner-5/result.json)** — Subtracts the 60 J descent potential change from total mechanical energy instead of conserving that total. Also initially writes the opposite rearrangement; later treats absolute energy 660 J as available from a 20 m climb foot. Capped.
- **[new-sledge-1](answers/new-sledge-1/result.json)** — Assigns gravity negative work during the descent and subtracts 50 J from the initial kinetic energy. Repeats the negative-energy calculation and stops at the cap.
- **[new-sledge-2](answers/new-sledge-2/result.json)** — Sets the initial potential energy to zero while taking the lower climb foot as zero, omitting the 50 J descent gain from available energy. Required energy and NO verdict happen to be right; available should be 70 J.

### round one: originally correct

- **[block-base](answers/block-base/result.json)** — Newly labels the correctly calculated 30 J after-climb remainder as available rather than 60 J before the climb. The scorer's 'descent omitted' label is not the written calculation. Step 4 also calls the remainder potential energy before using it as kinetic energy.
- **[block-m2-changed](answers/block-m2-changed/result.json)** — Newly evaluates one-half times 2 times 6 squared as 18 rather than 36 J. The descent is added afterward; required energy and NO verdict remain correct.
- **[block-m4-base](answers/block-m4-base/result.json)** — Newly reports 20 J after the climb as available rather than the correctly computed 60 J at its start. Numerical chain and positive-remainder YES decision are right. 'Descent omitted' is a coincidence because the climb cost equals the descent gain.
- **[confirm-parcel-1](answers/confirm-parcel-1/result.json)** — Newly subtracts 20 J from the initial 18 J during descent instead of obtaining 38 J. Loops over explanations of the fictitious negative kinetic energy and reaches the token cap.

