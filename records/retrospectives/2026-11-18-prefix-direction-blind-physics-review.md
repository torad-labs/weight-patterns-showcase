# Independent blind physics audit

I scored the 16 anonymous responses in packet `7b249645f58077c41f8bc0b34124895faf0ee69731ee868b98bfc9fe435b5542` before viewing assignments or identified answers. The table groups the 13 distinct response hashes. Each repeated answer has identical physics parameters and termination. **C** means correct, **I** incorrect and **U** unresolved. The verdict column checks the physical answer independently of comparison on the model's own stated operands. “Capped” records termination separately. [Frozen audit protocol](../../studies/04-round-2-overlap/runs/prefix-direction-generalization/audit-protocol.md)

| SHA-256 prefix | Opaque response IDs | End | Initial | Descent | Bottom | Friction | Available | Climb | Comparison | Verdict | Full chain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dbeef1e4258d` | `0d31618faf600858` | Capped | C | I | U | C | I | I | U | U | I |
| `dcd938f1adbe` | `09e95e3884f5b71f`, `1fedbc1234556d74` | Capped | C | I | U | C | I | I | U | U | I |
| `235a6a188308` | `064a5b688b81bbcb` | Capped | C | I | I | C | I | C | U | U | I |
| `dadb2e20b996` | `71c5ce7db001b39a` | Capped | C | I | U | I | I | C | C | I | I |
| `d7c5605e138b` | `ca5681888a4ae55b` | Capped | C | I | U | C | I | U | U | U | I |
| `1a07ec9d2177` | `acad092df8d805d6` | Capped | C | I | U | C | I | I | C | C | I |
| `6c7a0cec9c6c` | `a0ef43901066fb7b` | Capped | C | I | U | I | I | C | I | C | I |
| `54d1298e4239` | `dcf3b4c237158caf`, `4792ccabdc889490` | Ended | C | I | U | C | I | C | C | I | I |
| `ebbb0a5b05c5` | `7b1b798a4a725588` | Ended | C | I | U | C | I | C | C | C | I |
| `85ee961b81f6` | `1f69c27e141d4914`, `c932750dca3da2fd` | Capped | C | I | U | C | I | I | U | U | I |
| `90c422a44fcd` | `4b7e3a2524fcbc30` | Capped | C | I | U | C | I | I | U | U | I |
| `b0b0facac09b` | `797462c65056004a` | Ended | C | I | U | C | I | C | C | I | I |
| `1bb46cb6444c` | `b3219eafd021b2df` | Capped | C | I | C | C | I | C | U | U | I |

All initial kinetic-energy calculations give the correct 25 J or 49 J. The four physically correct sequences, in the order initial, bottom, available, required and verdict, are 25, 45, 39, 40, NO for `v=5,d=1,Q=6`, 25, 65, 41, 40, YES for `v=5,d=2,Q=24`, 49, 69, 61, 60, YES for `v=7,d=1,Q=8`, and 49, 89, 59, 60, NO for `v=7,d=2,Q=30`. Energy quantities are in joules.

| SHA-256 prefix | Evidence and judgment |
| --- | --- |
| `dbeef1e4258d` | Uses initial 49 as rough-section input and subtracts 8 once, obtaining 41. It sets the correctly signed top elevation to `-1+3=2 m` relative to the original start and calls the absolute potential 40 J the climb requirement. This can be used with 41 J in a consistent global reference calculation. It omits the bottom kinetic-energy gain and supplies neither requested local endpoint. The comparison cuts off while listing operands. |
| `dcd938f1adbe` | Carries 25 into a whole-route balance, adds -24 once and cancels elevation changes for a resulting 1 J. It gives no bottom kinetic energy or local available amount. In the incomplete required-energy step it explicitly places the top 2 m above the initial point, contradicting its preceding zero net elevation. The climb judgment reflects the incorrect datum. No numerical comparison or verdict follows. |
| `235a6a188308` | Calculates bottom as `49-40=9` and applies friction as `9-30=-21`. The explicit climb substitution `2×10×3` establishes 60 J despite the cut in typography. A preliminary negative-energy comment calls for reevaluation, and the answer ends before comparison against its own computed demand or a settled verdict. |
| `dadb2e20b996` | Skips the descent gain and computes 41 from `49-8`. The numerical friction operation is locally valid, but it defines `W_friction=-8` and then explicitly subtracts that signed quantity in the displayed formula. I marked the unretracted symbolic contradiction under the frozen rule. The climb gives 60 J and `41<60` is correct on its own figures. Its stated failure to reach the top is physically wrong. |
| `d7c5605e138b` | Starts with erroneous `49-40=9` and `9-30=-21`, then explicitly replaces that calculation with `49-30=19` and claims friction precedes descent. The retained route explicitly omits the downhill gain and produces no retained bottom amount. The numeric friction loss is applied once to the energy it assigns before the imagined rough segment. No completed climb calculation, comparison or settled verdict follows. |
| `1a07ec9d2177` | Uses 49 as its rough input, subtracts 8 and compares 41 with 40 from the absolute top elevation `-1+3=2 m`. That consistently referenced global comparison gives the correct YES. The required local 60 J and locally available 61 J do not appear. The descent geometry enters the global threshold instead of a bottom kinetic-energy computation. |
| `6c7a0cec9c6c` | Uses `49-30=19` with the descent gain omitted. Its stated negative friction work and symbolic subtraction conflict with the valid numeric subtraction. The alternate balance also inserts -30 into the requirement comparison and infers failure from `19=30` being false. The climb calculation explicitly evaluates the 3 m gain as 60 J, despite ambiguity in the earlier label “start point.” The alternative equality makes the comparison incorrect. The stated insufficient energy gives the physically correct NO. |
| `54d1298e4239` | Uses 25 as the rough input and subtracts 24 once to obtain 1 J. Gives the required 40 J and correctly compares its own amounts using `1<40`. Its NO is physically wrong. |
| `ebbb0a5b05c5` | Uses 25 as the rough input and subtracts 6 once to obtain 19 J. Gives 40 J and `19<40`. Its NO is physically correct. |
| `85ee961b81f6` | Uses 25 as the rough input and subtracts 6 once to obtain 19 J. It identifies the rise cost as 40 J but labels `40-25+6=21` the required energy, and leaves that treatment unrevised. The inequality check `19≠21` is part of a stated reevaluation and does not settle the reachability comparison. Capped before verdict. |
| `90c422a44fcd` | Subtracts 30 once from its own 49 J to produce 19 J. Uses `2+3=5 m` for the climb requirement and reports 100 J. The comparison and verdict trail off. |
| `b0b0facac09b` | Subtracts 8 once from its own 49 J and obtains 41 J. Gives the required 60 J and correctly compares `41<60`. Its NO is physically wrong. |
| `1bb46cb6444c` | Explicitly states that gravitational work is negative for this descent. It then solves the signed energy equation `25=v²-20`, obtaining the correct 45 J of bottom kinetic energy. The unretracted sign statement conflicts with the numerical descent and yields an incorrect descent judgment under the frozen contradiction rule. From its own total energy 25 J it removes 6 once and calls 19 J available, combining an absolute datum with a local 40 J climb demand. The cap leaves comparison and verdict unresolved. |

The two signed-friction formula contradictions leave 14 strict friction passes, even though all 16 responses show a subtraction of the correct amount from the energy used immediately before their reported friction calculation. The two globally referenced 41-versus-40 answers use a legitimate energy datum for reachability. Their local energy fields remain incorrect under the frozen rubric. The capped `1bb46cb6444c` answer performs the correct numerical descent, with an explicit conflicting sign statement. These distinctions leave every full chain incorrect because each has at least one incorrect operation. Comparison and verdict are each unresolved in nine responses.
