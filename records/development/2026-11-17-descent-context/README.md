# Does the model's generated context change the sign of the descent?

At exact frozen checkpoint 175 on the same GPU, the fitted problem naturally generated a correct addition and the prospectively selected development problem naturally generated a subtraction. The fitted case begins with 64 J and gains 40 J on the descent. The unfitted case begins with 96 J and gains 30 J. Their energy after friction should be 91 J and 115 J, respectively.

Both original responses stopped at exactly the same boundary before choosing the sign. Each had generated 141 tokens, ending immediately after writing the initial kinetic energy symbol in the equation for the bottom of the descent. The fitted history called this quantity “Total Energy before rough section” and named it E₁. The unfitted history called it “Kinetic Energy after descent” and named it KE_descent. Both histories calculated the positive magnitude of the potential-energy change and described it as “Potential Energy gained”. Their headings also differed.

The intervention crosses these two naturally occurring framings with each unchanged tokenized physics problem. Only the receiving problem's correct numeric inputs are supplied in the prefix, including its mass, starting energy, drop height and descent-energy magnitude. No prefix supplies a sign for combining the energies, any downstream energy, or a verdict. Each model uses greedy native generation from the sign to its natural ending. The two original framings serve as replay controls, allowing comparison of the complete logits at the sign and every subsequent generated token against the natural same-device acquisitions. All four conditions use the same reconstructed, write-sealed checkpoint. Four prefixes, with text and token IDs · Prospective measurement.

Each receiver retains its original native callback settings. Within each problem, the experimental intervention changes the already-generated context as a bundle. The contrast can identify dependence on that bundle while preserving the problem and native weights. It cannot identify which heading, variable label or individual word drives any effect, or why the model spontaneously chose a particular framing.

## The full pre-sign context reversed both calculations

| Original problem | Pre-sign framing | Addition minus subtraction logit | Free bottom calculation | After friction | Required | Verdict |
|---|---|---:|---|---:|---:|---|
| Fitted | Total energy before rough section | +6.031 | 64 + 40 = 104 J | 91 J | 20 J | YES |
| Fitted | Kinetic energy after descent | −4.558 | 64 − 40 = 24 J | 11 J | 20 J | NO |
| Unfitted | Total energy before rough section | +6.042 | 96 + 30 = 126 J | 115 J | 30 J | YES |
| Unfitted | Kinetic energy after descent | −4.681 | 96 − 30 = 66 J | 55 J | 30 J | YES |

Both unmodified diagonal prefixes reproduced the full vocabulary score hash at the first sign and their entire original freely generated histories. At fixed weights and with the same unfitted prompt and operands, supplying the fitted frame changed the addition-minus-subtraction logit margin by +10.723 and repaired the arithmetic. Supplying the unfitted framing to the fitted problem changed the margin by −10.589 and induced the reciprocal error. The two unfitted outcomes both answer YES because even the erroneous 55 J exceeds the 30 J needed. The intermediate calculations distinguish the genuine repair. [Native decisions and signed arithmetic](native-result.json).

The frame comparison changes two lines in each case: the descent section heading and the line that names the unknown bottom energy. The positive descent-magnitude line is identical within each receiving problem. At frozen 175 the unfitted problem can therefore perform the correct step with its own numbers; its natural answer enters a preceding context that favors the opposite sign. The direction of spontaneous frame choice and the reach of this effect across other problems remain separate questions.
