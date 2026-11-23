# Checkpoint-blind physics audit

The panel tests whether an edit to the existing checkpoint transfers during free generation. The four configurations are paired by an identical route described in chronological or endpoint-first order. Neither form supplies a worked answer. The public novelty inventory defines the scope of the unused-case claim. Each physical configuration counts once when summarizing the number of configurations. The paired wording outcomes remain visible.

An independent reader receives one globally shuffled collection of responses with per-case physics scalars and opaque response IDs. Prompt wording, endpoint names, acquisition order, trace readings and the pairing key are withheld through the manual audit. The reader computes the expected quantities from the parameters and records the model's actual equations, operands and values. Score every operation as **correct**, **incorrect** or **unresolved**. Mark an explicit contradiction incorrect unless the response clearly corrects and abandons the earlier attempt. If the reasoning ends before an operation, score that operation unresolved. Record termination and cap status separately from arithmetic.

| Field | Rule |
| --- | --- |
| initial | The model obtains $K_0=mv^2/2$. A rounded or wrong formula that happens to land on the right number still counts as incorrect when it is explicit. |
| descent | The model adds $mgd$ to its **own** incoming energy, with $d$ the positive descent height. Accept subtracting a negative potential change. A signed symbolic balance with no mistaken numerical application may establish the operation. A correct isolated magnitude or printed sign is insufficient. |
| bottom | The model derives or makes explicit the physically correct bottom energy $K_0+mgd$, including an unambiguous additive expression if the intermediate value is implicit. |
| friction | Given the model's own energy before the rough segment, it removes exactly $Q$ once. A combined balance may establish this when its application is explicit. A coherent subtraction survives an earlier descent error, and a coincidentally correct post-friction number does not repair an incorrect friction operation. |
| available | The response obtains the physically correct $K_0+mgd-Q$ as energy available at the start of the climb. |
| climb | The response obtains a required energy of $mgh$ for the final climb, using the given rise measured from the start of that climb. It must use this rise once. |
| comparison | The response compares its own available and required quantities with the correct inequality. |
| verdict | The reported physical verdict is YES when $K_0+mgd-Q \geq mgh$ and NO otherwise. Record comparison and verdict independently. |
| full_chain | Correct when every listed operation is correct. Incorrect when any listed operation is incorrect. Otherwise unresolved. |

Return a JSON object keyed by opaque ID. Each entry has one key for each field in the table. An operation is an object with a `status` and brief `evidence`; `full_chain` has its derived status and can omit evidence. Include `response_sha256` from the reading packet when practical. Record every response. When inference or ambiguity matters, cite the expression that supports the call. Never infer the computation solely from a correct final number.

The evaluation pairs exact prompt IDs after the complete blind audit. Report every pre/post status, including resolved omissions, regressions, preservation and persistent failures. A repair from an explicit wrong computation and resolution of an unobserved computation are distinct. An increase in valid descents with preservation of previously valid friction and climb would match the directional transfer prediction. A fair comparison retains any losses in other operations and any contrast between wording forms. Any lack of baseline errors limits observable repair. The operation audit is independent of trace discovery, model fitting, checkpoint eligibility and checkpoint selection.

The native read records two boundaries at feed-forward merge 20 for reproducibility and later diagnostics. That acquisition does not establish the held-out full trace or full-trace loss. Physics alone supplies the operation judgments in this record. The panel remains outside candidate ranking and model selection.
